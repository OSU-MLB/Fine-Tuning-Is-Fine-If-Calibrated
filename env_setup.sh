conda create -n HT python=3.6.13
conda activate HT

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt
