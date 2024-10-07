import torch
import logging
import json
import random
from pathlib import Path

# For some data types that are large and can hardly be fully stored in memory,
# We do offline tokenization and save them as codec sequences. e.g., audio
def load_pt_data(f):
    return torch.load(f, map_location='cpu')

def load_text_data(f):
    lines = open(f, encoding='utf-8').readlines()
    lines = [line.strip().split() for line in lines]
    ret = {}
    for line in lines:
        if len(line) < 2:
            logging.warning(f"find an empty entry: {line}")
            continue
        example_id, ctx = line[0], " ".join(line[1:])
        ret[example_id] = ctx
    return ret

def unified_loading(f):
    """ allow both format """
    if f.endswith('.pt'):
        return load_pt_data(f)
    else:
        return load_text_data(f)

loading_methods = {
    'audio': load_pt_data,
    'audio_prompt': unified_loading,
    'text': unified_loading,
    'content': unified_loading,
    'semantic': load_pt_data,
}
        
# 2. This part defines all valid task format.
# The data format of each task is defined as below:
# (1)   keys: data keys in order. This determines the order of the components in the sequences
# (2)   type: type of each data key. It determines the tokenizer for each data key
# (3)   features: some features belong to the examples but are not in the training sequence. e.g., speaker-id
# (4)   loss_key: key to predict. it determines which data key the loss should be computed on.
# (5)   encoder_keys: keys that are placed in encoder input when using the encoder-decoder format. 
#         Should always be the first several entry in "keys"
#         If this is set to None or [], it means encoder-decoder format is not supported, e.g., LM
# Note: you may need to add more type beside the current ones. However, to support a new type, you should
# provide a new tokenizer inherited from the AbsTokenizer
# Maybe some TODO: (1) features are text-only -> maybe more types
#                  (2) only one loss_key -> maybe support more than one


"""
Moshi format
"""
moshi_format = {
    'keys': ["audio_seq"],    # Stacked sequence of text, moshi_semantic, moshi_acoustic, user_semantic, user_acoustic
    'type': ["audio"],
    'max_len': [None],
    'tokenizer': ['audio'],
    'sp_token': [False],
    'default': [None],
    'features': [],
    'loss_key': 'audio_seq',
}

task_formats = {
    'moshi_ft': moshi_format,
}

for task_name, fmt in task_formats.items():
    assert len(fmt['keys']) == len(fmt['type']) \
        == len(fmt['max_len']) == len(fmt['tokenizer']) \
        == len(fmt['sp_token']) == len(fmt['default']), \
        task_name

# 3. This part defins how data is loaded in the data_dict at the loading stage
# It load all data into the memory according to the task format definition.
# It roughly compute the length of each data key, along with the length of the
# whole sequence that can be used for batchfy.
# However, it doesn't do any tokenization and data combination: they are done
# in the collate_fn
# Note, since all data is fully stored in the memory during training, the data
# should only be in light format: e.g., Text / Codec.
# Other raw data are not supported since volume is large: e.g., raw audio
# / image / SSL model embeddings (they are computed on-the-fly in the tokenizers).
def load_data_for_all_tasks(json_files, tokenizers):
    """ accept and parse multiple json_files, each of which represents a task dataset"""
    data_dict = {}
    for json_file in json_files:
        dataset_json = json.load(open(json_file)) 
        logging.info(f"loading dataset file: {json_file} for {dataset_json['task']} task") 
        print(f"loading dataset file: {json_file} for {dataset_json['task']} task")                  
        task_data = load_data_for_one_task(dataset_json, tokenizers)
        data_dict.update(task_data)
    logging.info(f"from all json files, we have {len(data_dict)} examples")
    print(f"from all json files, we have {len(data_dict)} examples")
    return data_dict

def load_data_for_one_task(dataset_json, tokenizers):
    task_type = dataset_json['task']
    task_format = task_formats[task_type]

    # load default token seq options
    defaults = {}
    for key, dft, tk_type in zip(task_format['keys'], task_format['default'], task_format['tokenizer']):
        if dft is not None:
            defaults[key] = [ tokenizers[tk_type].tokenize(i) for i in dft ]

    # load data for each data key
    data_dict = {}
    for key, data_type, max_len in zip(task_format['keys'], task_format['type'], task_format['max_len']):
        if key not in defaults:
            if key in dataset_json['keys']:
                logging.info(f"loading file: {dataset_json['keys'][key]} as key: {key}")
                print(f"loading file: {dataset_json['keys'][key]} as key: {key}")
                this_data_dict = loading_methods[data_type](dataset_json['keys'][key])
                this_data_dict = {f"{dataset_json['task']}_{k}": v # TODO(optional): distinguish different json file via prefix
                        for k, v in this_data_dict.items()
                }
                for example_id, data in this_data_dict.items():
                    if example_id not in data_dict:
                        data_dict[example_id] = {}
                    data_dict[example_id][key] = data if max_len is None else data[:max_len]
            else:
                raise ValueError(f"For task {task_type}, data key {key} is needed but missing.")
    
    # (random) sample token seq from defaults for each example
    total_samples = len(data_dict)
    for key, candidates in defaults.items():
        for example_id, dft_tk in zip(data_dict, random.choices(candidates, k=total_samples)):
            data_dict[example_id][key] = dft_tk.clone()

    # load data for each feature
    for feat in task_format['features']:
        if feat not in dataset_json['features']:
            raise ValueError(f"For task {task_type}, data feature {feat} is needed but missing")

        feature_file = dataset_json['features'][feat]
        logging.info(f"loading file: {feature_file} as a feature: {feat}")

        feature_dict = open(feature_file).readlines()
        feature_dict = [line.strip().split() for line in feature_dict]
        feature_dict = {line[0]: line[1:] for line in feature_dict}

        for example_id, data in feature_dict.items():
            if example_id not in data_dict:
                data_dict[example_id] = {}
            data_dict[example_id][feat] = data

    # Validate the data: remove the examples when some entries are missing.
    # add the task label after validation
    example_ids = list(data_dict.keys())
    for example_id in example_ids:
        for key in task_format['keys'] + task_format['features']:
            if key not in data_dict[example_id]:
                del data_dict[example_id]
                logging.warning(f"{task_type} example {example_id} is removed since {key} is missing")
                #print(f"{task_type} example {example_id} is removed since {key} is missing")
                break

    example_ids = list(data_dict.keys())
    for example_id in example_ids:
        data_dict[example_id]['task'] = task_type
        data_dict[example_id]['loss_key'] = task_format['loss_key']

    logging.info(f"done loading this raw data dict: {len(data_dict)} valid examples")
    print(f"done loading this raw data dict: {len(data_dict)} valid examples")

    return data_dict

if __name__ == "__main__":
    pass

