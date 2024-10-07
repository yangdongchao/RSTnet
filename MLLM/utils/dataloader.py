
import json
import os
import sys
import torch
import copy
import random
import logging
import torch.distributed as dist

from tools.tokenizer.MimiCodec.mimi_tokenizer import MimiTokenizer
from tools.tokenizer.Text2ID.moshi_text_tokenizer import Text2IDTokenizer

def print_log(content: str):
    logging.info(content)
    print(content)

def load_jsons(json_files):
    """output: {id: tensor(cpu)}"""

    data_dict = {}
    for json_file in json_files:
        dataset_json = json.load(open(json_file)) 
        # Moshi only have one key called 'audio_seq'
        this_pt_path = dataset_json['keys']['audio_seq']
        this_data_dict = torch.load(this_pt_path, map_location='cpu')
        data_dict.update(this_data_dict)

        logging.info(f"done loading {this_pt_path}: {len(this_data_dict)} valid examples")
        print(f"done loading {this_pt_path}: {len(this_data_dict)} valid examples")
    
    logging.info(f"from all json files, we have {len(data_dict)} examples")
    print(f"from all json files, we have {len(data_dict)} examples")

    return data_dict

def build_data_iterator(
        data_dict,
        delay_step=1,
        max_length=-1,
        min_length=-1,
        batch_scale=1000,
        is_train=True,
        n_worker=1,
        seed=999,
        minibatch_debug=-1,
    ):
    valid_utts = filter_data(data_dict, max_length, min_length) 
    batches = batchfy(data_dict, valid_utts, batch_scale) # prepare batch

    logging.info(f"Finish pre-process all data. {len(valid_utts)} examples and {len(batches)} batches")
    print(f"Finish pre-process all data. {len(valid_utts)} examples and {len(batches)} batches")

    if minibatch_debug > 0:
        batches = batches[:min(minibatch_debug, len(batches))]
        logging.info(f"only use {len(batches)} as this is a debug mode")
    dataset = Dataset(batches, data_dict)
    sampler = DDPSyncSampler(size=len(batches), seed=seed, is_train=is_train)
    # Build iterator. No multi-process when debug
    collate_fn = Collate_Fn_Factory(
            max_length=max_length if max_length > 0 else 15000,
            delay_step=delay_step,
    )
    if minibatch_debug != -1:
        iterator = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
            collate_fn=collate_fn,
        )
        logging.info("disable multi-processing data loading: debug mode")
    else:
        # debug 
        iterator = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=n_worker,
            prefetch_factor=min(100, len(batches)),
            collate_fn=collate_fn,
        )
    return iterator

def filter_data(data_dict, max_length, min_length):
    # we find the valid key rather than remove the whole exmaple as the invalid exmaples can 
    # also work as the prompt
    print('filter data ', max_length, min_length)
    keys = list(data_dict.keys())
    if max_length <= 0 and min_length <= 0:
        return keys

    valid_keys = []
    if max_length > 0:
        for k in keys:
            if  (data_dict[k].shape[-1] <= max_length or max_length <= 0) \
            and (data_dict[k].shape[-1] >= min_length or min_length <= 0):
                valid_keys.append(k)
    logging.info(f"you requires length between [{min_length}, {max_length}] so only {len(valid_keys)}/{len(keys)} examples are reserved.")
    print(f"you requires length between [{min_length}, {max_length}] so only {len(valid_keys)}/{len(keys)} examples are reserved.")
    return valid_keys

def batchfy(data_dict, batch_utts, batch_scale):
    batch_utts.sort(key=lambda x: data_dict[x].shape[-1]) # sort 
    batch_lengths = [data_dict[k].shape[-1] for k in batch_utts]
    print('batch_scale ', batch_scale)
    # TODO: maybe length**2 is a better measure of computing complexity

    # Only take care of the uttid rather than the whole example
    batches, batch, summed_tokens = [], [], 0
    for utt, l in zip(batch_utts, batch_lengths):
        if l + summed_tokens > batch_scale:
            assert len(batch) > 0, f"batch_tokens should be larger: {batch_scale}"
            batches.append(copy.deepcopy(batch))
            batch, summed_tokens = [], 0

        summed_tokens += l
        batch.append(utt)

    if len(batch) > 0:
        batches.append(copy.deepcopy(batch))

    # TODO: maybe report statistics
    logging.info(f'After batchfy, there are {len(batches)} batches')
    print(f'After batchfy, there are {len(batches)} batches')
    return batches 

class Dataset(torch.utils.data.Dataset):
    """ Dataset. Each example is exactly a batch """
    def __init__(self, data_split, data_dict):
        self.data_split = data_split # batches
        self.data_dict = data_dict

    def __getitem__(self, index):
        uttids = self.data_split[index]
        return [(uttid, self.data_dict[uttid]) for uttid in uttids]

    def __len__(self):
        return len(self.data_split)

class SequentialSampler(object):
    def __init__(self, sequence):
        self.seq = sequence

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)

    def refresh(self):
        pass

class DDPSyncSampler(object):
    def __init__(self, size, seed, is_train=True):
        self.size = size
        self.seed = seed
        self.epoch = 0
        self.is_train = is_train

        # Ensure that data iterator aross all GPUs has the same number of batches
        if dist.is_initialized() and torch.cuda.is_available():
            local_rank = int(os.environ["LOCAL_RANK"])
            device = torch.device(f"cuda:{local_rank}")
            size = torch.Tensor([size]).to(device).int()
            dist.all_reduce(size, dist.ReduceOp.MAX)

            self.pad_number = size.item() - self.size
            self.rank = dist.get_rank()
        else:
            logging.warning("torch.distributed is not available!")
            self.pad_number = 0
            self.rank = 0

        self.refresh()

    def refresh(self):
        seq = list(range(self.size))

        if self.is_train:
            # Assume the batches are sorted from shortest to longest
            # This introduces local randomness by local random shuffling
            # otherwise each global batch will be identical across epochs
            chunk_size, start = 10, 0
            random.seed(self.rank + self.seed + self.epoch)
            while start < self.size:
                seg = seq[start: min(self.size, start + chunk_size)]
                local_random_order = random.sample(list(range(len(seg))), len(seg))
                seg = [seg[i] for i in local_random_order]
                seq[start: min(self.size, start + chunk_size)] = seg
                start += len(seg)

            # even after this shuffle, the batch lengths across GPUs 
            # are very similar
            random.seed(self.seed + self.epoch)
            random.shuffle(seq)

        # so the #batches are identical across GPUs
        if self.pad_number > 0:
            seq = list(range(self.pad_number)) + seq

        self.seq = seq
        self.epoch += 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)

    def get_state_dict(self):
        state_dict = {
            'epoch': self.epoch,
            'seed': self.seed,
        }
        return state_dict

    def load_state_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)

class Collate_Fn_Factory(object):
    def __init__(self, 
                 max_length=15000,
                 delay_step=1,
    ):
        self.max_length = max_length
        self.delay_step = delay_step

        self.text_empty_token = 32000
        self.semantic_empty_token = 2048
        self.acoustic_empty_token = 2048
        self.padding_token = 0

    def delay(self, d):
        """[17, T] -> [17, T+self.delay_step]"""
        # d = d[:, :self.max_length - self.delay_step]
        original_length = d.shape[-1]
        delay_length = original_length + self.delay_step # we really need the last frame? I am not sure.
        
        # prepare delay sequences
        sequence = torch.ones((17, delay_length)).long() # 
        # layer 0/1/9: right padding
        sequence[0, -self.delay_step:] = self.text_empty_token
        sequence[1, -self.delay_step:] = self.semantic_empty_token
        sequence[9, -self.delay_step:] = self.semantic_empty_token
        # layer 2-8/10-16: left padding
        sequence[2:9, :self.delay_step] = self.acoustic_empty_token
        sequence[10:17, :self.delay_step] = self.acoustic_empty_token

        sequence[0, :original_length] = d[0] # text do not change
        sequence[1, :original_length] = d[1] # semantic do not change
        sequence[9, :original_length] = d[9] # semantic
        sequence[2:9, self.delay_step:delay_length] = d[2:9]
        sequence[10:17, self.delay_step:delay_length] = d[10:17]

        return sequence

    def delay_collate_fn(self, batch):
        """Output: data and mask [B, 17, T] """

        batch_size = len(batch)
        # print('batch ', batch_size)
        sequences = torch.ones((batch_size, 17, self.max_length+1)).long() * self.padding_token
        masks = torch.zeros((batch_size, 17, self.max_length+1)).bool() # 0 denotes mask

        lengths, example_ids= [], []
        for idx, (example_id, d) in enumerate(batch):
            sequence = self.delay(d) # we first set delay pattern for each data
            #print('sequence ', sequence.shape, sequence)
            sequences[idx, :, :sequence.shape[-1]] = sequence
            masks[idx, :, :sequence.shape[-1]] = True # donot 
            lengths.append(sequence.shape[-1])
            example_ids.append(example_id)

        sequences = sequences[:, :, :max(lengths)].long()
        masks = masks[:, :, :max(lengths)]
        lengths = torch.Tensor(lengths).long()
        return sequences, masks, lengths, example_ids

    def __call__(self, batch):
        assert len(batch) == 1, "batch size should only be 1"
        batch = batch[0] # a list of data
        return self.delay_collate_fn(batch)

def get_data_iterator_tokenizer_vocabulary(
        args,
        train_jsons,
        valid_jsons,
        batch_scale=3000,
        delay_step=1,
        minibatch_debug=-1,
        max_length=-1,
        min_length=-1,
        non_acoustic_repeat=1,
        n_worker=4,
        decoder_only=True,
        seed=999,
    ):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format=f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    # (1) load all data in the raw format
    logging.info(f"loading train: {train_jsons}")
    print(f"loading train: {train_jsons}")
    train_data_dict = load_jsons(train_jsons)

    logging.info(f"loading valid:  {valid_jsons}")
    print(f"loading valid:  {valid_jsons}")
    valid_data_dict = load_jsons(valid_jsons)

    # (2) build data iterator
    valid_iterator = build_data_iterator(
        valid_data_dict,
        delay_step=delay_step, 
        max_length=max_length,
        min_length=min_length,
        batch_scale=batch_scale,
        is_train=False,
        n_worker=n_worker,
        seed=seed,
        minibatch_debug=minibatch_debug,
    )
    train_iterator = build_data_iterator(
        train_data_dict, 
        delay_step=delay_step, 
        max_length=max_length,
        min_length=min_length,
        batch_scale=batch_scale, 
        is_train=True,
        n_worker=n_worker,
        seed=seed,
        minibatch_debug=minibatch_debug,
    )
    logging.info('all iterator built')
    return train_iterator, valid_iterator

if __name__ == "__main__":
    get_data_iterator_tokenizer_vocabulary(sys.argv[1:2], sys.argv[2:3], n_worker=1) 
