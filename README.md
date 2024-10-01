# Fine-Tuning is Fine, if Calibrated

__Official implementation of the following work:__

*Mai, Z., Chowdhury, A., Zhang, P., Tu, C.H., Chen, H.Y., Pahuja, V., Berger-Wolf, T., Gao, S., Stewart, C., Su, Y., & Chao, W.L. Fine-Tuning is Fine, if Calibrated. NeurIPS 2024.*  

## Preparation

### Environment

Please ensure you have `conda` installed. Then run `./env_setup.sh`.

### Data and Checkpoints
Download and overwrite `asset` and `out` folders using the following link:

[Download Folder](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/zhang_14217_buckeyemail_osu_edu/EtFPrq4qjclMobShSjD1SlQBwW5_own1xctxN6U1gifGxA?e=O4Tmwl)

Once overwritten, go through `asset` and extract all the `.tar.gz` files to their respective locations. After extraction, the `asset` folder structure should look like this:

```
├── ImageNet
│   ├── cross_val_config
│   │   ├── R.json
│   │   └── S.json
│   ├── data
│   │   ├── imagenet-r
│   │   │   ├── n01443537
│   │   │   ├── n01484850
│   │   │   ├── n01494475
│   │   │   ├── n01498041
│   │   │   └── ...
│   │   └── imagenet-s
│   │       ├── n01440764
│   │       ├── n01443537
│   │       ├── n01484850
│   │       ├── n01491361
│   │       └── ...
│   └── image_list
│       ├── R_testing.txt
│       ├── R_training.txt
│       ├── S_testing.txt
│       └── S_training.txt
└── OfficeHome
    ├── cross_val_config
    │   ├── Ar_Cl.json
    │   ├── Ar_Pr.json
    │   ├── Ar_Rw.json
    │   ├── Rw_Ar.json
    │   ├── Rw_Cl.json
    │   └── Rw_Pr.json
    ├── data
    │   ├── Art
    │   │   ├── Alarm_Clock
    │   │   ├── Backpack
    │   │   ├── Batteries
    │   │   ├── Bed
    │   │   └── ...
    │   ├── Clipart
    │   │   ├── Alarm_Clock
    │   │   ├── Backpack
    │   │   ├── Batteries
    │   │   ├── Bed
    │   │   └── ...
    │   ├── Product
    │   │   ├── Alarm_Clock
    │   │   ├── Backpack
    │   │   ├── Batteries
    │   │   ├── Bed
    │   │   └── ...
    │   └── Real_World
    │       ├── Alarm_Clock
    │       ├── Backpack
    │       ├── Batteries
    │       ├── Bed
    │       └── ...
    └── image_list
        ├── Art_testing.txt
        ├── Art_training.txt
        ├── Clipart_testing.txt
        ├── Clipart_training.txt
        ├── Product_testing.txt
        ├── Product_training.txt
        ├── Real_World_testing.txt
        └── Real_World_training.txt
```

## Reproduction

### Evaluation

For evaluating the checkpoints we reported in paper, run the following commands:

```bash
# OfficeHome
# Source Ar
python -m py.main --eval --device "cuda:0" --dataset "OfficeHome" --source "Ar" --target "Rw" --eval_model_path out/paper_ckpt/OfficeHome/Ar_Rw.pth
python -m py.main --eval --device "cuda:0" --dataset "OfficeHome" --source "Ar" --target "Cl" --eval_model_path out/paper_ckpt/OfficeHome/Ar_Cl.pth
python -m py.main --eval --device "cuda:0" --dataset "OfficeHome" --source "Ar" --target "Pr" --eval_model_path out/paper_ckpt/OfficeHome/Ar_Pr.pth
# Source Rw
python -m py.main --eval --device "cuda:0" --dataset "OfficeHome" --source "Rw" --target "Ar" --eval_model_path out/paper_ckpt/OfficeHome/Rw_Ar.pth
python -m py.main --eval --device "cuda:0" --dataset "OfficeHome" --source "Rw" --target "Cl" --eval_model_path out/paper_ckpt/OfficeHome/Rw_Cl.pth
python -m py.main --eval --device "cuda:0" --dataset "OfficeHome" --source "Rw" --target "Pr" --eval_model_path out/paper_ckpt/OfficeHome/Rw_Pr.pth

# ImageNet
python -m py.main --eval --device "cuda:0" --dataset "ImageNet" --target "R" --eval_model_path out/paper_ckpt/ImageNet/ImageNet-R.pth
python -m py.main --eval --device "cuda:0" --dataset "ImageNet" --target "S" --eval_model_path out/paper_ckpt/ImageNet/ImageNet-S.pth
```

### Training

If you are intested in training your own model:

```bash
# OfficeHome
## Source Ar
python -m py.main --train --device "cuda:0" --dataset "OfficeHome" --source "Ar" --target "Rw"
python -m py.main --train --device "cuda:0" --dataset "OfficeHome" --source "Ar" --target "Cl"
python -m py.main --train --device "cuda:0" --dataset "OfficeHome" --source "Ar" --target "Pr"
## Source Rw
python -m py.main --train --device "cuda:0" --dataset "OfficeHome" --source "Rw" --target "Ar"
python -m py.main --train --device "cuda:0" --dataset "OfficeHome" --source "Rw" --target "Cl"
python -m py.main --train --device "cuda:0" --dataset "OfficeHome" --source "Rw" --target "Pr"

# ImageNet
python -m py.main --train --device "cuda:0" --dataset "ImageNet" --target "R"
python -m py.main --train --device "cuda:0" --dataset "ImageNet" --target "S"
```

There are other parameters you can play with. Please check `py/main.py`.

## Citation

```
@misc{mai2024finetuningfinecalibrated,
      title={Fine-Tuning is Fine, if Calibrated}, 
      author={Zheda Mai and Arpita Chowdhury and Ping Zhang and Cheng-Hao Tu and Hong-You Chen and Vardaan Pahuja and Tanya Berger-Wolf and Song Gao and Charles Stewart and Yu Su and Wei-Lun Chao},
      year={2024},
      eprint={2409.16223},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.16223}, 
}
```