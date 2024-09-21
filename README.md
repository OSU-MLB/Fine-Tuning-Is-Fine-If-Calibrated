# TPDD (Training under Partial Domain Data)

## Official Releases

This repository details the official releases of following work:

_Fine-Tuning is Fine with Partial Target Data_

## Training Instructions

### Regular Mode
Run the training in regular mode using the following command:

```bash
python -m py.main --device "cuda:0" --dataset "OfficeHome" --source "Ar" --target "Rw"
```

Evaluation
```bash
python -m py.main --device "cuda:0" --dataset "OfficeHome" --source "Ar" --target "Rw" --eval --pretrained_model_path "/research/nfs_chao_209/chowdhury/TPDD/out/19.pth" 
```

## Stay Tuned

More algorithms, example scripts and detailed reproducing guides are on the way to facilitate easier adoption and experimentation.
