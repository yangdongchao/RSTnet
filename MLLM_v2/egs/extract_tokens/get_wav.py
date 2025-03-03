import os 
import glob

names = glob.glob("/home-dongchao/data/source/*.wav")
f = open('/home-dongchao/code3/RSTnet_private/MLLM/egs/extract_tokens/wav.scp', 'w')
for name in names:
    bs_name = os.path.basename(name)
    f.write(bs_name+' '+name+'\n')


    