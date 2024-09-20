import os
import glob

f = open('train.scp', 'w')
names = glob.glob('LibriSpeech/train-clean-100/*/*/*.flac')

for name in names:
    bs_name = os.path.basename(name)
    f.write(name+'\n')

names = glob.glob('LibriSpeech/train-clean-360/*/*/*.flac')
for name in names:
    bs_name = os.path.basename(name)
    f.write(name+'\n')

names = glob.glob('LibriSpeech/train-other-500/*/*/*.flac')
for name in names:
    bs_name = os.path.basename(name)
    f.write(name+'\n')



