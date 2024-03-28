# conda create -n clip4sbsr python=3.10 -y
# conda activate clip4sbsr
# conda install pytorch=2.1.2=py3.10_cuda11.8_cudnn8.7.0_0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# pip install torch==2.1 torchvision==0.16 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.2
pip install peft==0.10.0
pip install adapters==0.1.2
pip install numpy==1.26.4
pip install easydict==1.13
pip install wandb==0.15.12
pip install scipy==1.12.0
pip install scikit-learn==1.3.2

