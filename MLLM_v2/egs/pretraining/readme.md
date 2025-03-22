## Main idea

1. preprocess the dataset

    First, prepare environment following [Emilia](https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia). The Emilia codes are at `exripts/emilia`.

    Then, modify paths in `prepare_broadcast_data.sh` and run. The dataloader will yield a token sequence with the following [B, 9, t_text+t_audio+2] shape:
```
    <|begin_of_text|>[<spk_id1>] <text> ··· [<spk_id1>] <text><|text_emply_token|>···
                                                              <semantic_tokens>···<|semantic_emply_token|>
                                                              <|semantic_emply_token|><acoustic_tokens>···
```

2. Pre-training

3. Post-training

4. inference

