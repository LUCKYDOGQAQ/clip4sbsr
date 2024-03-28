cd /lizhikai/workspace/clip4sbsr
/lizhikai/anaconda3/envs/clip4sbsr/bin/python script/hf_download.py \
 --model openai/clip-vit-base-patch32 \
 --save_dir hf_model/ \
 --exclude *.msgpack \
 --exclude *.h5 \