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

''' for the audio-only format. We add zero_text PADDING to the text streaming. 
    For the zero_text token, we add loss scale to reducing it's influence. e.g. 1/N
    Speech: Youdos, 200k hours ~9B frames. The actually tokens should be 9B*8. 
    Audio: Stable Audio Open+ about 10k hours. about 0.45M frames.
'''
audio_only_format = {
    'keys': ["audio_seq"],   
    'type': ["audio"],
    'sp_token': ['zero_text'],
    'features': [],
    'loss_key': ['audio_seq']
}
''' for the text-only format. We add zero_audio (zero-semantic and zero-acoustic) PADDING to the audio streaming. 
    For the zero_audio token, we add loss scale to reducing it's influence. e.g. 1/N
'''
text_only_format = {
    'keys': ["text_seq"],   
    'type': ["text"],
    'sp_token': ['zero_audio'],
    'features': [],
    'loss_key': ['text_seq']
}

''' for the text-audio interleaved format. We add zero_text PAD and zero_semantic, zero_acoustic to non-part
    And we donot calculate loss for the text streaming.
'''
setence_level_text_audio_interleaved_format = {
    'keys': ["text_seq", "audio_seq"],
    'type': ["text", "audio"],
    'sp_token': ['zero_text', 'zero_audio'],
    'features': [],
    'loss_key': ['text_seq', 'audio_seq']
}


''' segment-level text-audio interleaved: whisperx will split the audio into several segments. we make them as interleavel
'''
segment_level_audio_text_interleaved_format = {
    'keys': ["audio_seq", "text_seq"],
    'type': ["audio", "text"],
    'sp_token': ['zero_text', 'zero_audio'],
    'features': [],
    'loss_key': ['text_seq', 'audio_seq']
}

''' word-level text-audio interleaved: we will choose the short audio
'''
word_level_audio_text_interleaved_format = {
    'keys': ["audio_seq", "text_seq"],
    'type': ["audio", "text"],
    'sp_token': ['zero_text', 'zero_audio'],
    'features': [],
    'loss_key': ['text_seq', 'audio_seq']
}

''' word-alignemt text-audio. We force the alignment. Only for speech-text
    PAD the text in word level.
'''
word_level_audio_text_alignment_format = {
    'keys': ["audio_seq", "text_seq"],
    'type': ["audio", "text"],
    'sp_token': ['zero_text', 'zero_audio'],
    'features': [],
    'loss_key': ['audio_seq']
}


task_formats = {
    'text_only': text_only_format,
    'audio_only': audio_only_format,
    'setence_level_text_audio_interleaved': setence_level_text_audio_interleaved_format,
    'segment_level_audio_text_interleaved': segment_level_audio_text_interleaved_format,
    'word_level_audio_text_interleaved': word_level_audio_text_interleaved_format,
    'word_level_audio_text_alignment': word_level_audio_text_alignment_format
}

# 3. This part defins how data is loaded in the data_dict at the loading stage
# It load all data into the memory according to the task format definition.
# It roughly compute the length of each data key, along with the length of the
# whole sequence that can be used for batchfy.
# However, it doesn't do any tokenization and data combination: they are done
# in the collate_fn
# Note, since all data is fully stored in the memory during training, the data
# should only be in light format: e.g., Text / Codec.
# Other raw data are not supported since volume is large: e.g., raw audio
def load_data_for_all_tasks(json_files):
    """ accept and parse multiple json_files, each of which represents a task dataset"""
    data_dict = {}
    text_dict = {}
    for json_file in json_files:
        dataset_json = json.load(open(json_file)) 
        logging.info(f"loading dataset file: {json_file} for {dataset_json['task']} task") 
        print(f"loading dataset file: {json_file} for {dataset_json['task']} task") 
        task_data = load_data_for_one_task(dataset_json)     
        if dataset_json['task'] == 'text_only':
            text_dict.update(task_data)
        else:
            data_dict.update(task_data)
    logging.info(f"from all json files, we have {len(data_dict)} examples and {len(text_dict)} text only examples")
    return data_dict, text_dict

def load_data_for_one_task(dataset_json):
    task_type = dataset_json['task']
    task_format = task_formats[task_type]
    # load data for each data key
    data_dict = {}
    for key, data_type in zip(task_format['keys'], task_format['type']):
        if key not in dataset_json['keys']:
            raise ValueError(f"For task {task_type}, data key {key} is needed but missing.")
        logging.info(f"loading file: {dataset_json['keys'][key]} as key: {key}")
        print(f"loading file: {dataset_json['keys'][key]} as key: {key}")
        this_data_dict = loading_methods[data_type](dataset_json['keys'][key])
        this_data_dict = {f"{dataset_json['task']}_{k}": v 
                for k, v in this_data_dict.items()}
        for example_id, data in this_data_dict.items():
            if example_id not in data_dict:
                data_dict[example_id] = {}
            data_dict[example_id][key] = data
    # Validate the data: remove the examples when some entries are missing.
    # add the task label after validation
    example_ids = list(data_dict.keys())
    for example_id in example_ids:
        for key in task_format['keys']:
            if key not in data_dict[example_id]:
                del data_dict[example_id]
                logging.warning(f"{task_type} example {example_id} is removed since {key} is missing")
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

