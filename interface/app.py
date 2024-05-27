import sys
sys.path.append('.')

import numpy as np
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from model.sketch_model import SketchModel
from model.classifier import Classifier
from model.view_model import MVCNN
from pathlib import Path
from PIL import Image

from torchvision import datasets
from dataset.view_dataset import MultiViewDataSet
from torchvision import transforms
import torch.nn.functional as F
import os

top_k = 16
backbone = "./hf_model/models--openai--clip-vit-base-patch32"
# view_dir = "/lizhikai/workspace/clip4sbsr/data/SHREC13_ZS2/13_view_render_test_img"
view_dir = "/lizhikai/workspace/clip4sbsr/data/SHREC14_ZS2/14_view_render_train_img"
feature_path = "/lizhikai/workspace/clip4sbsr/output/feature.mat"
# ckpt_path = "/lizhikai/workspace/clip4sbsr/ckpt/lora-unshared_prompt-ams_tcl"
ckpt_path = "/lizhikai/workspace/clip4sbsr/ckpt/Epoch5"

view_data = MultiViewDataSet(root=view_dir, transform=transforms.Resize(224))
feature = torch.load(feature_path)
for key in feature.keys():
    feature[key] = torch.tensor(feature[key]).to("cuda")

use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
if torch.cuda.is_available(): device = "cuda"
elif torch.backends.mps.is_available(): device = "mps"
else: device = "cpu"
print(f"Currently using device: {device}")

sketch_model = SketchModel(backbone=backbone)
view_model = MVCNN(backbone=backbone)
classifier = Classifier(12, 512, 50)
if use_gpu:
    sketch_model =sketch_model.to(device)
    view_model = view_model.to(device)
    classifier = classifier.to(device)

classifier.load_state_dict(torch.load(Path(ckpt_path) / 'mlp_layer.pth'))
sketch_model.load(Path(ckpt_path) /'sketch_lora')
view_model.load(Path(ckpt_path) / 'view_lora')
sketch_model.eval()
view_model.eval()
classifier.eval() 

def convert_transparent_to_white(input_image):
    # 检查图片是否为RGBA模式，即带有透明度通道
    image=input_image
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # 分割图片的RGBA通道
    r, g, b, a = image.split()

    # 创建一个新的白色背景图片，大小与原始图片相同
    white_background = Image.new('RGB', image.size, (255, 255, 255))

    # 将原始图片的RGB通道复制到新的白色背景图片上
    white_background.paste(image, mask=a)

    # 保存转换后的图片
    return white_background

def topk_result(sketch_feature, k):
    cosine_similarities = F.cosine_similarity(sketch_feature, feature['view_feature'], dim=1)
    top_scores, top_indices = torch.topk(cosine_similarities, k)

    res_images = []
    res_labels = []
    for i in top_indices:
        res_images.append(view_data[i][0][5])
        res_labels.append(view_data[i][1])

    return res_images, res_labels


def sbsr(input_img):
    # 对输入数据进行处理：
    input_img=input_img['composite']
    pil_img = Image.fromarray(np.uint8(input_img))
    pil_img=convert_transparent_to_white(pil_img)
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像调整为指定大小，如果不为正方形，会压缩成正方形，而不是裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711])])
    input = image_transforms(pil_img).unsqueeze(0).to(device)

    # print(input.shape)
    with torch.no_grad():
        output = sketch_model.forward(input)
        mu_embeddings = classifier.forward(output)

    # 提前计算好，数据库中的view embedding，进行比对排名，得到最相近的model
    sketch_feature = nn.functional.normalize(mu_embeddings, dim=1)
    res_images, res_labels = topk_result(sketch_feature, top_k)

    # return zip(res_images, [view_data.idx_to_class[index] for index in res_labels])
    return res_images


def get_gallery_images():
    return  [ (f"{view_dir}/race_car/4/M000827_00{i}.jpg", 'race_car') for i in range(1, 10)] #[(image0, label0), (image1, label1), ...]
    # return  [ (f"{view_dir}/ant/0/m0_{i}.png", 'ant') for i in range(12)] #[(image0, label0), (image1, label1), ...]

with gr.Blocks(title="3D Shape Model Retrieval using Sketch") as demo:
    gr.Markdown("""
                # 3D Shape Model Retrieval using Sketch
                Upload or Draw a Sketch to find the most similar 3D Shape Models!
                """)
    
    with gr.Row():
        gr.Gallery(
            value=get_gallery_images(),
            label="Example 3D Objects", 
            show_label=True, 
            elem_id="gallery", 
            columns=[10], 
            rows=[1], 
            object_fit="contain", 
            height=256,
            preview=True)
        # gr.Image("/lizhikai/workspace/clip4sbsr/data/SHREC13_ZS2/13_view_render_test_img/ant/0/m0_0.png", scale=0)
        # gr.Image("/lizhikai/workspace/clip4sbsr/data/SHREC13_ZS2/13_view_render_test_img/ant/0/m0_1.png", scale=0)

    with gr.Row():
        with gr.Column(scale=2):
            # inp=gr.Image(label="Sketch", 
            #              height='512px',)
            inp=gr.ImageEditor(image_mode='RGBA',label='Sketch',
                                # value={
                                #     "background": np.full((600, 800), 255, dtype=np.uint8),
                                #     "layers": None,
                                #     "composite": np.full((600, 800), 255, dtype=np.uint8),
                                # },
                               height='512px',
                               brush=gr.Brush(colors=["#000000"])),
        with gr.Column(scale=2):
            out=gr.Gallery(label="3D Shape Model", 
                           columns=[4], 
                           show_label=True, 
                           object_fit="contain", 
                           height='512px',
                           interactive=False,)
    
    btn = gr.Button("Search!")
    btn.click(fn=sbsr, inputs=inp[0], outputs=out)


demo.launch()



   

