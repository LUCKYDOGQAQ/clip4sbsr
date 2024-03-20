conda create -n clip4sbsr python=3.10
conda activate clip4sbsr
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c huggingface transformers

