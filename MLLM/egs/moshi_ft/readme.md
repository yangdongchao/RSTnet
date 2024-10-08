# Recipe for Fine-tuning Moshi on Multi-stream Speech Data

Here we provide an example of fine-tuning Moshi RQ-Transformer on Fisher dataset, containing both data pre-processing and training pipelines.

## Data Preprocessing

### 1. Environment

You may need to create a separate conda environment for data preprocessing.

```bash
conda create -n moshi-data python=3.9
conda activate moshi-data
conda install ffmpeg
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118
pip install pyannote.audio
pip install git+https://github.com/m-bain/whisperx.git --upgrade
pip install deepfilternet
```

If you need VAD to segment long conversation or need super-resolution, there are two additional things to do:

- For PyAnnote VAD, remember to apply on [HuggingFace](https://huggingface.co/pyannote/voice-activity-detection) first.
- For AERO, we made a few modification on the official repo to support batched 8kHz->24kHz SR. Please clone and install [this repository](https://github.com/Ching-Yee-Chan/aero.git) and download the original checkpoint. Finally, fill in aero_root in run.sh.

### 2. Download Dataset

Download your multi-stream audio dataset to `$db_root` in `run.sh`. All wavs under this directory will be processed.

### 3. Run run.sh

Note that stage 1-3 need `moshi-data` env, whereas stage 4-6 need `open-moshi`. You may need to manually switch the conda env.