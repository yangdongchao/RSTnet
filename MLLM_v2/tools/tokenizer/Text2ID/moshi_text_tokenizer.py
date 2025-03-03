import math
import os
import sentencepiece
from huggingface_hub import hf_hub_download
import torch
from tools.tokenizer.abs_tokenizer import AbsTokenizer


class Text2IDTokenizer(AbsTokenizer):
    def __init__(self):
        super(Text2IDTokenizer, self).__init__()
        ckpt_path = "Moshi/ckpts/moshiko-pytorch-bf16/tokenizer_spm_32k_3.model"
        if not os.path.exists(ckpt_path):
            ckpt_path = hf_hub_download("kyutai/moshiko-pytorch-bf16", "tokenizer_spm_32k_3.model")
        self.model = sentencepiece.SentencePieceProcessor(ckpt_path)  # type: ignore
    
    def get_word_to_subword_mapping(self, tokens):
        word_to_subword = []
        current_word = ""
        current_subwords = []
        for token in tokens:
            if token.startswith("▁"):  # SentencePiece 使用 "▁" 作为词的开始标记
                if current_word:
                    word_to_subword.append({
                        "word": current_word, 
                        "tokens": current_subwords
                    })
                current_word = token[1:]  # 去掉 "▁"
                current_subwords = [self.model.piece_to_id(token)]
            else:
                current_word += token
                current_subwords.append(self.model.piece_to_id(token))
        if current_word:
            word_to_subword.append({
                        "word": current_word, 
                        "tokens": current_subwords
                    })
        return word_to_subword

    def debug(self, first, second, token):
        # first["tokens"] = [self.model.id_to_piece(token) for token in first["tokens"]]
        # second["tokens"] = [self.model.id_to_piece(token) for token in second["tokens"]]
        print(first)
        print(second)
        print(self.model.id_to_piece(token))
    
    def pad_tokens(self, word_list, duration, frame_rate=12.5):
        EPAD = 0    # 0: <unk> / <epad>
        PAD = 3 # 3: <pad>
        length = math.ceil(duration * frame_rate)
        text_tokens = torch.ones(length, dtype=torch.long) * PAD
        for idx, word in enumerate(word_list):
            if "start" not in word:
                continue
            start = round(word["start"] * frame_rate)
            end = round(word["end"] * frame_rate)
            
            if start == 0:
                start += 1
                end += 1
            # insert <epad> only if not overlapped with previous word
            if text_tokens[start-1] == PAD:
                text_tokens[start-1] = EPAD
            for i, token in enumerate(word["tokens"]):
                if start + i >= length:
                    # print("WARNING: word exceeds duration")
                    # print(word)
                    # print(idx, len(word_list), start+len(word["tokens"])-length)
                    break
                # if text_tokens[start + i] != PAD:
                #     print("WARNING: word overlapped with previous word")
                #     self.debug(word_list[idx-1], word, token)
                text_tokens[start + i] = token
        return text_tokens

    def tokenize(self, segments):
        word_list = []
        for segment in segments:
            tokens = self.model.encode_as_pieces(segment["text"])
            word_to_subword = self.get_word_to_subword_mapping(tokens)
            for word, tokens in zip(segment["words"], word_to_subword):
                assert word["word"] == tokens["word"], "tokenized word does not match original word!"
                word["tokens"] = tokens["tokens"]
                word_list.append(word)
        return word_list
