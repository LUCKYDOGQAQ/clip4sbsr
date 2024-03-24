conda create -n clip4sbsr python=3.10
conda init
conda activate clip4sbsr
# conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda install -c huggingface transformers
conda install -y numpy
pip install peft
pip install adapters

