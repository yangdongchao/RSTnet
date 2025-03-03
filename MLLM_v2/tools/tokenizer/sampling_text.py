''' this code provides by Marcob. It used to sampling some text.
'''
import json
from transformers import AutoTokenizer
from megatron_tokenizer_lite import indexed_dataset # pip install megatron_tokenizer_lite
import numpy as np
from tqdm import tqdm
from tools.tokenizer.abs_tokenizer import AbsTokenizer
import random
import sys
import torch
import argparse
import logging

def get_parser():
    parser = argparse.ArgumentParser(
        description="convert a data list, do tokenization and save as a torch .pt file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-file", type=str, help="dict")
    parser.add_argument("--rank", type=int, help="local GPU rank, if applicable")
    parser.add_argument("--target", type=int, help="the number of saming tokens")
    return parser

class TextChooseTokenizer(AbsTokenizer):
    def __init__(self, target, idx):
        self.paths = [
            "/weka2/proj-stablelm/dataset_hub/stablelm_2_arcade_tokenized/the_pile_v2/arxiv/stablelm_2_arcade_tokenized-the_pile_v2-arxiv-1-merged",
            "/weka2/proj-stablelm/dataset_hub/stablelm_2_arcade_tokenized/the_pile_v2/pubmed/stablelm_2_arcade_tokenized-the_pile_v2-pubmed-1-merged",
            "/weka2/proj-stablelm/dataset_hub/stablelm_2_arcade_tokenized/the_pile_v2/s2orc/stablelm_2_arcade_tokenized-the_pile_v2-s2orc-1-merged",
            "/weka2/proj-stablelm/dataset_hub/stablelm_2_arcade_tokenized/the_pile_v1/philpapers/stablelm_2_arcade_tokenized-the_pile_v1-philpapers-1-merged",
            "/weka2/proj-stablelm/dataset_hub/stablelm_2_arcade_tokenized/the_pile_v1/bookcorpus/stablelm_2_arcade_tokenized-the_pile_v1-bookcorpus-1-merged",
            "/weka2/proj-stablelm/dataset_hub/stablelm_2_arcade_tokenized/the_pile_v2/gutenberg/stablelm_2_arcade_tokenized-the_pile_v2-gutenberg-1-merged",
            "/weka2/proj-stablelm/dataset_hub/stablelm_2_arcade_tokenized/openwebmath/stablelm_2_arcade_tokenized-openwebmath-1-merged",
            "/weka2/proj-stablelm/dataset_hub/stablelm_2_arcade_tokenized/redpajama/wiki/stablelm_2_arcade_tokenized-redpajama-wiki-1-merged",
            "/weka2/proj-stablelm/dataset_hub/stablelm_2_arcade_tokenized/redpajama/stackexchange/stablelm_2_arcade_tokenized-redpajama-stackexchange-1-merged",
            "/weka/home-marcob/edu_filter/tokenized/rw_filtered_merged_text_document",
            "/weka2/proj-stablelm/dataset_hub/stablelm_2_arcade_tokenized/c4_rephrased_ctx_with_qa/c4_rephrased_ctx_with_qa_part0_text_document",
            "/weka2/proj-rephrasing_llm_data/data/fineweb-edu_100BT_tokenized/fineweb-edu_100BT_text_document",
            "/weka2/proj-stablelm/dataset_hub/stablelm_2_arcade_tokenized/cc_2024/2024_0_text_document",
            "/weka2/proj-stablelm/dataset_hub/stablelm_2_arcade_tokenized/cc_2024/2024_1_text_document",
            "/weka2/proj-stablelm/dataset_hub/stablelm_2_arcade_tokenized/cc_2024/2024_2_text_document",
            "/weka2/proj-stablelm/dataset_hub/stablelm_2_arcade_tokenized/cc_2024/2024_3_text_document",
            "/weka2/proj-stablelm/dataset_hub/stablelm_2_arcade_tokenized/cc_2024/2024_4_text_document",
        ]
        self.idx = idx % (len(self.paths))
        self.data = [indexed_dataset.make_dataset(path) for path in self.paths[self.idx:self.idx+1]]
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-2-1_6b")
        self.target = target # 
    
    def sampling(self):
        sampling_data, total = [], 0
        pbar = tqdm()
        np.random.seed(42)
        while total < self.target:
            source_id = np.random.choice(range(len(self.data))) # sample source
            source = self.data[source_id]
            doc_id = np.random.randint(0, len(source)) # sample doc
            con = self.tokenizer.decode(source[doc_id])
            con = con.replace('<|endoftext|>', '')
            word_list = con.split(' ')
            if len(word_list) > 250:
                du = random.randint(10, 250) # randomly choose a length
                st = random.randint(0,len(word_list)-du)
                ch_con = ' '.join(word_list[st:st+du])
                sampling_data.append(ch_con)
                total += du
            else:
                ch_con = ' '.join(word_list)
                sampling_data.append(ch_con)
                total += len(word_list)
            #pbar.set_description(f"Processed {total}{ '/' + str(target)} tokens")
        return sampling_data

def main(args):
    args = get_parser().parse_args(args)
    args.rank -= 1 # run.pl starts from 1 but the exact jobid / gpuid starts from 0   
    t_chooser = TextChooseTokenizer(target=int(args.target), idx=args.rank)
    sampling_data = t_chooser.sampling()
    f = open(args.output_file, 'w')
    with open(args.output_file, "w") as f:
        json.dump(sampling_data, f, indent=4)
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
    
