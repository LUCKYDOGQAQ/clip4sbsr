conda create -n clip4sbsr python=3.10
conda init
conda activate clip4sbsr

pip install torch==2.1 torchvision==0.16 torchaudio==2.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.36.2
pip install peft
pip install adapters
pip install numpy

