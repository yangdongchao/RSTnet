import json
import math
import sys
#sys.path.append('/weka2/home-dongchao/code3/RSTnet_private/MLLM')
from pathlib import Path
from typing import Union

import torch
from tools.tokenizer.common import fix_and_load_json
from tools.tokenizer.abs_tokenizer import AbsTokenizer


class TextTokenizer(AbsTokenizer):
    def __init__(self, checkpoint_dir: Union[Path, str], max_length=500) -> None:
        super(TextTokenizer, self).__init__()
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise NotADirectoryError(f"The checkpoint directory does not exist: {str(checkpoint_dir)}")
        # some checkpoints have both files, `.json` takes precedence
        if (vocabulary_path := checkpoint_dir / "tokenizer.json").is_file():
            from tokenizers import Tokenizer as HFTokenizer
            self.model = HFTokenizer.from_file(str(vocabulary_path))
            self.backend = "huggingface"
            # get BOS and EOS ids
            if (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
                with open(special_tokens_path, encoding="utf-8") as fp:
                    config = json.load(fp)
                bos_token = config.get("bos_token")
                eos_token = config.get("eos_token")
                if bos_token is not None and isinstance(bos_token, dict):
                    bos_token = bos_token.get("content")
                if eos_token is not None and isinstance(eos_token, dict):
                    eos_token = eos_token.get("content")
                self.bos_id = self.token_to_id(bos_token) if bos_token is not None else None
                self.eos_id = self.token_to_id(eos_token) if eos_token is not None else None
            if (special_tokens_path := checkpoint_dir / "generation_config.json").is_file():
                try:
                    with open(special_tokens_path, encoding="utf-8") as fp:
                        config = json.load(fp)
                except json.JSONDecodeError:  # Some files like the Llama 3.2 one have bugs
                    with open(special_tokens_path, encoding="utf-8") as fp:
                        json_string = fp.read()
                        config = fix_and_load_json(json_string)
                if self.bos_id is None:
                    self.bos_id = config.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = config.get("eos_token_id")
        else:
            vocabulary_path = next(checkpoint_dir.glob("tokenizer*.model"), None)
            assert vocabulary_path is not None, f"No vocabulary file found in {str(checkpoint_dir)}"
            from sentencepiece import SentencePieceProcessor
            self.model = SentencePieceProcessor(model_file=str(vocabulary_path))
            self.backend = "sentencepiece"
            self.bos_id = self.model.bos_id()
            self.eos_id = self.model.eos_id()
        # Set special token ids. we use the reversed token from llama 3 tokenizer
        self.pad_id = 128004    # 0: <unk> / <epad>
        self.epad_id = 128005    # 3: <pad>
        self.use_bos = True
        self.use_eos = True
        self.max_length = max_length
    
    def tokenize2(self, token):
        if isinstance(token, torch.Tensor):
            return token.to(torch.int64)
        else:
            raise NotImplementedError

    def find_length(self, x):
        return x.shape[-1]

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        if len(tokens) == 1 and self.apply_decoding_fix:
            dummy_token_id = 33  # \x1e
            dummy_token = self.model.decode([dummy_token_id])
            return self.model.decode([dummy_token_id] + tokens)[len(dummy_token) :]
        return self.model.decode(tokens)

    def token_to_id(self, token: str) -> int:
        if self.backend == "huggingface":
            id_ = self.model.token_to_id(token)
        elif self.backend == "sentencepiece":
            id_ = self.model.piece_to_id(token)
        else:
            raise RuntimeError
        if id_ is None:
            raise ValueError(f"token {token!r} not found in the collection.")
        return id_
    
    def get_word_to_subword_mapping(self, tokens, ids):
        word_to_subword = []
        current_word = ""
        current_subwords = []
        for i, token in enumerate(tokens):
            # SentencePiece 使用 "▁" 作为词的开始标记
            # tiktorch 使用 "Ġ" 作为词的开始标记
            if token.startswith("▁") or token.startswith("Ġ"):  
                if current_word:
                    word_to_subword.append({
                        "word": current_word, 
                        "tokens": current_subwords
                    })
                current_word = token[1:]  # 去掉 "▁"
                current_subwords = [ids[i]]
            else:
                current_word += token
                current_subwords.append(ids[i])
        if current_word:
            word_to_subword.append({
                        "word": current_word, 
                        "tokens": current_subwords
                    })
        return word_to_subword

    def pad_tokens(self, word_list, duration, frame_rate=12.5):
        EPAD = self.epad_id    # 0: <unk> / <epad>
        PAD = self.pad_id # 3: <pad>
        length = math.ceil(duration * frame_rate)
        text_tokens = torch.ones(length, dtype=torch.long) * PAD    # initialize with <pad>
        for idx, word in enumerate(word_list):
            # Skip words without timestep
            if "start" not in word:
                continue

            # Convert seconds to frames
            start = round(word["start"] * frame_rate)
            end = round(word["end"] * frame_rate)
            
            # Shift back 1 frame for PAD if it is the first word
            if start == 0:
                start += 1
                end += 1

            # insert <epad> only if not overlapped with previous word
            if text_tokens[start-1] == PAD:
                text_tokens[start-1] = EPAD
            for i, token in enumerate(word["tokens"]):
                if start + i >= length:
                    break
                text_tokens[start + i] = token
        return text_tokens

    def tokenize_text(self, text):
        ''' input the text setence. output the ID sequence.
        '''
        if self.backend == "huggingface":
            encodings = self.model.encode(text) # this returns a `Encoding` object
            tokens = encodings.tokens
            ids = encodings.ids
        elif self.backend == "sentencepiece":
            tokens = self.model.encode_as_pieces(text)   # this returns a list of tokens
            ids = [self.model.piece_to_id(token) for token in tokens]
        else:
            raise RuntimeError(f"`{self.backend}` is not supported.")
        if self.use_bos:
            if ids[0] != self.bos_id:
                ids = [self.bos_id] + ids
        if self.use_eos:
            if ids[-1] != self.eos_id:
                ids = ids + [self.eos_id]
        if self.max_length > 0:
            ids = ids[:self.max_length]
        return ids
    
    def tokenize_segment(self, segments):
        word_list = []
        for segment in segments:
            if self.backend == "huggingface":
                encodings = self.model.encode(segment["text"]) # this returns a `Encoding` object
                tokens = encodings.tokens
                ids = encodings.ids
            elif self.backend == "sentencepiece":
                tokens = self.model.encode_as_pieces(segment["text"])   # this returns a list of tokens
                ids = [self.model.piece_to_id(token) for token in tokens]
            else:
                raise RuntimeError(f"`{self.backend}` is not supported.")
            # remove BOS token if it exists
            if ids[0] == self.bos_id:
                tokens = tokens[1:]
                ids = ids[1:]
            # print('tokens ', tokens, ids)
            word_to_subword = self.get_word_to_subword_mapping(tokens, ids)
            # print('word_to_subword ', word_to_subword)
            for word, tokens in zip(segment["words"], word_to_subword):
                # assert word["word"] == tokens["word"], "tokenized word does not match original word!"
                word["tokens"] = tokens["tokens"]
                word_list.append(word)
        return word_list
        

if __name__ == "__main__":
    tx_tokenizer = TextTokenizer('/home-dongchao/data/checkpoints/meta-llama/Meta-Llama-3-8B')
    j_data = '/home-dongchao/data/source/p225_001.json'
    with open(j_data) as f:
        metadata = json.load(f)
    word_list = tx_tokenizer.tokenize_segment(metadata["segments"])
    print('word_list ', word_list)
    text_tokens = tx_tokenizer.pad_tokens(word_list, metadata["duration"])
    print('text_tokens ', text_tokens.shape)
    
