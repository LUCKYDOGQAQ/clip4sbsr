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

view_dir = "/lizhikai/workspace/clip4sbsr/data/SHREC13_ZS2/13_view_render_test_img"

id2label = dict()
id = 0
for label in sorted(os.listdir(view_dir)): # Label
    for _ in range( len(os.listdir(view_dir + '/' + label))):
        id2label[id] = str(label)
        id += 1
print()

def topk_result(sketch_feature, k):
    feature = torch.load("/lizhikai/workspace/clip4sbsr/output/feature.mat")
    for key in feature.keys():
        feature[key] = torch.tensor(feature[key]).to("cuda")

    cosine_similarities = F.cosine_similarity(sketch_feature, feature['view_feature'], dim=1)
    top_scores, top_indices = torch.topk(cosine_similarities, k)

    # sketch_data = datasets.ImageFolder(root="/lizhikai/workspace/clip4sbsr/data/SHREC13_ZS2/13_sketch_test_picture", transform=transforms.Resize(224))
    view_data = MultiViewDataSet(root=view_dir, transform=transforms.Resize(224))
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

    sketch_model = SketchModel(backbone="./hf_model/models--openai--clip-vit-base-patch32")
    view_model = MVCNN(backbone="./hf_model/models--openai--clip-vit-base-patch32")
    classifier = Classifier(12, 512, 133)

    if use_gpu:
        sketch_model =sketch_model.to(device)
        view_model = view_model.to(device)
        classifier = classifier.to(device)     

    # Load model
    # sketch_model.load(args.ckpt_dir+'sketch_lora')
    # view_model.load(args.ckpt_dir + 'view_lora')
        
    classifier.load_state_dict(torch.load(Path('./ckpt/Epoch29') / 'mlp_layer.pth'))
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
    res_images, res_labels = topk_result(sketch_feature, 10)

    return zip(res_images, [id2label[index] for index in res_labels])

demo = gr.Interface(fn=sbsr, 
                    inputs=gr.Image(label="Sketch"), 
                    outputs=gr.Gallery(label="3D Shape Model", 
                                       show_label=True, 
                                    #    elem_id="gallery", 
                                       object_fit="contain", 
                                       height="auto"),
                    title="3D Shape Model Retrieval using Sketch",
                    description="Upload a Sketch to find the most similar 3D Shape Models!")

demo.launch()



   

