## Open-MimiCodec

## Begin to training
```
bash run_train.sh
```

## Begin to inference
```
bash test.sh
```

## Discussion: semantic_feature_type
The paper use wavlm as semantic teacher, we can also try to use other models, such as whisper, hubert, and so on. 
We support to choose different semantic teacher by change the parameter semantic_feature_type.

## Discussion: reconstruction performance
From the results, the codec reconstruction performance still has improvement room. Thus, reproduce it and improve it is meanfully.

