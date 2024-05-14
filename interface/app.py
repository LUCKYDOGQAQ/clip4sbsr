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
view_dir = "/lizhikai/workspace/clip4sbsr/data/SHREC13_ZS2/13_view_render_test_img"
feature_path = "/lizhikai/workspace/clip4sbsr/output/feature.mat"
ckpt_path = "/lizhikai/workspace/clip4sbsr/ckpt/lora-unshared_prompt-ams_tcl"

# id2label = dict()
# id = 0
# for label in sorted(os.listdir(view_dir)): # Label
#     for _ in range( len(os.listdir(view_dir + '/' + label))):
#         id2label[id] = str(label)
#         id += 1

feature = torch.load(feature_path)
for key in feature.keys():
    feature[key] = torch.tensor(feature[key]).to("cuda")
view_data = MultiViewDataSet(root=view_dir, transform=transforms.Resize(224))

def topk_result(sketch_feature, k):
    cosine_similarities = F.cosine_similarity(sketch_feature, feature['view_feature'], dim=1)
    top_scores, top_indices = torch.topk(cosine_similarities, k)

    # sketch_data = datasets.ImageFolder(root="/lizhikai/workspace/clip4sbsr/data/SHREC13_ZS2/13_sketch_test_picture", transform=transforms.Resize(224))
    res_images = []
    res_labels = []
    for i in top_indices:
        res_images.append(view_data[i][0][0])
        res_labels.append(view_data[i][1])

    return res_images, res_labels


def sbsr(input_img):

    use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"
    print(f"Currently using device: {device}")

    sketch_model = SketchModel(backbone=backbone)
    view_model = MVCNN(backbone=backbone)
    classifier = Classifier(12, 512, 133)

    if use_gpu:
        sketch_model =sketch_model.to(device)
        view_model = view_model.to(device)
        classifier = classifier.to(device)     

    sketch_model.load(Path(ckpt_path) /'sketch_lora')
    view_model.load(Path(ckpt_path) / 'view_lora')
    classifier.load_state_dict(torch.load(Path(ckpt_path) / 'mlp_layer.pth'))

    sketch_model.eval()
    view_model.eval()
    classifier.eval() 

    # 对输入数据进行处理：
    pil_img = Image.fromarray(np.uint8(input_img * 255))
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像调整为指定大小，如果不为正方形，会压缩成正方形，而不是裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711])])
    input = image_transforms(pil_img).unsqueeze(0).to(device)

    # print(input.shape)
    with torch.no_grad():
        output = sketch_model.forward(input)
        mu_embeddings= classifier.forward(output)
        #mu_embeddings,logits = classifier.forward(output)

    # 提前计算好，数据库中的view embedding，进行比对排名，得到最相近的model
    sketch_feature = nn.functional.normalize(mu_embeddings, dim=1)
    res_images, res_labels = topk_result(sketch_feature, top_k)

    return zip(res_images, [view_data.idx_to_class[index] for index in res_labels])

def get_gallery_images():
    return  [ (f"/lizhikai/workspace/clip4sbsr/data/SHREC13_ZS2/13_view_render_test_img/ant/0/m0_{i}.png", 'ant') for i in range(12)] #[(image0, label0), (image1, label1), ...]


with gr.Blocks(title="3D Shape Model Retrieval using Sketch") as demo:
    gr.Markdown("""
                # 3D Shape Model Retrieval using Sketch
                Upload a Sketch to find the most similar 3D Shape Models!
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
            inp=gr.Image(label="Sketch", 
                         height='512px',)
            
        with gr.Column(scale=2):
            out=gr.Gallery(label="3D Shape Model", 
                           columns=[4], 
                           show_label=True, 
                           object_fit="contain", 
                           height='512px',
                           interactive=False,)
    
    btn = gr.Button("Search!")
    btn.click(fn=sbsr, inputs=inp, outputs=out)


demo.launch()



   

