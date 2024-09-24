# TPDD (Training under Partial Domain Data)

## Official Releases

This repository details the official releases of following work:

_Fine-Tuning is Fine, if Calibrated_

## Training Instructions

### Regular Mode
Run the training in regular mode using the following command:

```bash
python -m py.main --train --device "cuda:0" --dataset "OfficeHome" --source "Ar" --target "Rw"
python -m py.main --train --device "cuda:0" --dataset "OfficeHome" --source "Ar" --target "Cl"
python -m py.main --train --device "cuda:0" --dataset "OfficeHome" --source "Ar" --target "Pr"
python -m py.main --train --device "cuda:0" --dataset "OfficeHome" --source "Rw" --target "Ar"
python -m py.main --train --device "cuda:0" --dataset "OfficeHome" --source "Rw" --target "Cl"
python -m py.main --train --device "cuda:0" --dataset "OfficeHome" --source "Rw" --target "Pr"

python -m py.main --train --device "cuda:0" --dataset "ImageNet" --target "R"
python -m py.main --train --device "cuda:0" --dataset "ImageNet" --target "S"
```

Evaluation
```bash
python -m py.main --eval --device "cuda:0" --dataset "OfficeHome" --source "Ar" --target "Rw" --eval_model_path out/paper_ckpt/OfficeHome/Ar_Rw.pth
python -m py.main --eval --device "cuda:0" --dataset "OfficeHome" --source "Ar" --target "Cl" --eval_model_path out/paper_ckpt/OfficeHome/Ar_Cl.pth
python -m py.main --eval --device "cuda:0" --dataset "OfficeHome" --source "Ar" --target "Pr" --eval_model_path out/paper_ckpt/OfficeHome/Ar_Pr.pth
python -m py.main --eval --device "cuda:0" --dataset "OfficeHome" --source "Rw" --target "Ar" --eval_model_path out/paper_ckpt/OfficeHome/Rw_Ar.pth
python -m py.main --eval --device "cuda:0" --dataset "OfficeHome" --source "Rw" --target "Cl" --eval_model_path out/paper_ckpt/OfficeHome/Rw_Cl.pth
python -m py.main --eval --device "cuda:0" --dataset "OfficeHome" --source "Rw" --target "Pr" --eval_model_path out/paper_ckpt/OfficeHome/Rw_Pr.pth

python -m py.main --eval --device "cuda:0" --dataset "ImageNet" --target "R" --eval_model_path out/paper_ckpt/ImageNet/ImageNet-R.pth
python -m py.main --eval --device "cuda:0" --dataset "ImageNet" --target "S" --eval_model_path out/paper_ckpt/ImageNet/ImageNet-S.pth

Evaluation (Cross-Val)
python -m py.main --eval --device "cuda:0" --dataset "OfficeHome" --source "Ar" --target "Rw" --eval_model_path out/paper_ckpt/OfficeHome/Ar_Rw.pth --cross_val_config "asset/OfficeHome/cross_val_config/Ar_Rw.json"
```
