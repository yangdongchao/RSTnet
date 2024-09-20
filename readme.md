<img src="moshi.png" width="400px"></img>

# Reproduce the training process of Moshi

Pytorch implementation of [Moshi](https://kyutai.org/Moshi.pdf), "Moshi: a speech-text foundation model for real-time dialogue", from Kyutai Lab.

In this repo, we will try to reproduce the training process of Moshi, including their audio codec (Mimi), and their hierarchical LM for text and audio.

## Install

```
conda create -n open-moshi python=3.12
conda activate open-moshi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tqdm
pip install librosa==0.9.1
pip install matplotlib
pip install omegaconf 
pip install einops
pip install vector_quantize_pytorch
pip install tensorboard

```

## Plan
- [x] Release the training code of Mimi codec
- [ ] Release the first version training code of hierarchical LM for text and audio modeling
- [ ] Release the final version training code of hierarchical LM for text and audio modeling

## MimiCodec
The detail please refer to MimiCodec part. The paper gives a lot of useful tricks during training. All of these tricks has been used in our reproduced version, including:
- [x] Removing reconstruction loss
- [x] High compression rate (12.5hz)
- [x] not applying quantization with a certain probability during training
- [x] Semantic guidance

## Moshi LM
The details will be updated in the following days.

## Reference
The implements of audio codec and hierarchical LM are based on previous codebase:
https://github.com/yangdongchao/AcademiCodec 
https://github.com/yangdongchao/LLM-Codec
https://github.com/yangdongchao/UniAudio
https://github.com/kyutai-labs/moshi

## Citations

```bibtex
@techreport{kyutai2024moshi,
    author = {Alexandre D\'efossez and Laurent Mazar\'e and Manu Orsini and Am\'elie Royer and
			  Patrick P\'erez and Herv\'e J\'egou and Edouard Grave and Neil Zeghidour},
    title = {Moshi: a speech-text foundation model for real-time dialogue},
    institution = {Kyutai},
    year={2024},
    month={September},
    url={http://kyutai.org/Moshi.pdf},
}
```
```bibtex
@article{yang2023hifi,
  title={HiFi-Codec: Group-residual Vector quantization for High Fidelity Audio Codec},
  author={Yang, Dongchao and Liu, Songxiang and Huang, Rongjie and Tian, Jinchuan and Weng, Chao and Zou, Yuexian},
  journal={arXiv preprint arXiv:2305.02765},
  year={2023}
}
```
```bibtex
@article{yang2023uniaudio,
  title={UniAudio: An Audio Foundation Model Toward Universal Audio Generation},
  author={Dongchao Yang, Jinchuan Tian, Xu Tan, Rongjie Huang, Songxiang Liu, Xuankai Chang, Jiatong Shi, Sheng Zhao, Jiang Bian, Xixin Wu, Zhou Zhao, Helen Meng},
  journal={arXiv preprint arXiv:2310.00704},
  year={2023}
}
```
