# -*- coding: utf-8 -*-

import os
import sys
sys.path.append('.')

import argparse
import numpy as np
import scipy.io as sio
#sys.path.append('../mobilenet')

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from model.sketch_model import SketchModel
from model.classifier import Classifier
from model.view_model import MVCNN
from dataset.view_dataset_reader import MultiViewDataSet
from utils.metric import evaluation_metric, cal_cosine_distance
#from sketch_dataset import SketchDataSet
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics.pairwise import pairwise_distances

from tqdm import tqdm
from pathlib import Path
import yaml
from easydict import EasyDict

# parser = argparse.ArgumentParser("feature extraction of sketch images")
# # SHREC13
# # parser.add_argument('--sketch-datadir', type=str, default='E:/3d_retrieval/Dataset/Shrec13_ZS2/13_sketch_test_picture')
# # parser.add_argument('--view-datadir', type=str, default='E:/3d_retrieval/Dataset/Shrec13_ZS2/13_view_render_test_img')
# # SHREC14
# parser.add_argument('--sketch-datadir', type=str, default='E:/3d_retrieval/Dataset/Shrec14_ZS2/14_sketch_test_picture')
# parser.add_argument('--view-datadir', type=str, default='E:/3d_retrieval/Dataset/Shrec14_ZS2/14_view_render_test_img')
# parser.add_argument('--workers', default=5, type=int,
#                     help="number of data loading workers (default: 0)")

# # test
# parser.add_argument('--batch-size', type=int, default=32)
# parser.add_argument('--num-classes', type=int, default=133)

# # misc
# parser.add_argument('--gpu', type=str, default='0')
# parser.add_argument('--seed', type=int, default=1)
# parser.add_argument('--use-cpu', action='store_false')
# parser.add_argument('--model', default="/lizhikai/workspace/clip4sbsr/hf_model/models--openai--clip-vit-base-patch32")
# parser.add_argument('--pretrain', type=bool, choices=[True, False], default=True)
# parser.add_argument('--uncer', type=bool, choices=[True, False], default=False)
# # features
# parser.add_argument('--cnn-feat-dim', type=int, default=512)
# parser.add_argument('--feat-dim', type=int, default=256)
# parser.add_argument('--test-feat-dir', type=str, default='feature.mat')
# parser.add_argument('--train-feat-dir', type=str, default='/home/daiweidong/david/strong_baseline/sketch_modality/shrec_14/train_sketch_picture')
# parser.add_argument('--model-dir', type=str, default='./ckpt_14_3Epoch10/')

# parser.add_argument('--pattern', type=bool, default=False,
#                     help="Extract training data features or test data features,'True' is for train dataset")

# args = parser.parse_args()

parser = argparse.ArgumentParser("feature extraction of sketch images")
parser.add_argument('-c', '--config', help="running configurations", type=str, required=True)
with open(parser.parse_args().config, 'r', encoding='utf-8') as r:
    config = EasyDict(yaml.safe_load(r))


def get_test_data(sketchdir,viewdir):
    """Image reading, but no image augmentation
       Args:
         traindir: path of the traing picture
    """
    image_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711])])  # Imagenet standards
    sketch_data = datasets.ImageFolder(root=sketchdir, transform=image_transforms)
    view_data = MultiViewDataSet(root=viewdir, transform=image_transforms)
    sketch_dataloaders = DataLoader(sketch_data, batch_size=config.dataset.batch_size, shuffle=False, num_workers=config.dataset.workers)
    view_dataloaders = DataLoader(view_data, batch_size=config.dataset.batch_size, shuffle=False, num_workers=config.dataset.workers)
    return sketch_dataloaders,view_dataloaders,len(sketch_data),len(view_data)


def main():
    # torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.train.gpu
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"
    print(f"Currently using device: {device}")

    # sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + '.txt'))
    # if use_gpu:
    #     print("Currently using GPU: {}".format(args.gpu))
    #     cudnn.benchmark = True
    #     # torch.cuda.manual_seed_all(args.seed)
    # else:
    #     print("Currently using CPU")

    sketchloader,viewloader,sketch_num,view_num = get_test_data(config.dataset.sketch_datadir,config.dataset.view_datadir)
    sketch_model = SketchModel(backbone=config.model.model)
    view_model = MVCNN(backbone=config.model.model)
    classifier = Classifier(12,config.features.cnn_feat_dim,config.train.num_classes)

    if use_gpu:
        sketch_model =sketch_model.to(device)
        view_model = view_model.to(device)
        classifier = classifier.to(device)     

    # Load model
    sketch_model.load(Path(config.model.ckpt_dir) /'sketch_lora')
    view_model.load(Path(config.model.ckpt_dir) / 'view_lora')
    classifier.load_state_dict(torch.load(Path(config.model.ckpt_dir) / 'mlp_layer.pth'))
    sketch_model.eval()
    view_model.eval()
    classifier.eval() 


    # Define two matrices to store extracted features
    sketch_feature = None
    sketch_labels = None
    view_feature = None
    view_labels = None

    for batch_idx, (data, labels) in tqdm((enumerate(sketchloader)), total = len(sketchloader), leave=True, desc="Extracting sketch features"):
        # if use_gpu:
        data, labels = data.to(device), labels.to(device)

        # print(batch_idx)
        with torch.no_grad():
            output = sketch_model.forward(data)
            mu_embeddings= classifier.forward(output)
        #mu_embeddings,logits = classifier.forward(output)

        outputs = nn.functional.normalize(mu_embeddings, dim=1)

        #logits = classifier.forward(outputs)
        labels_numpy = labels.detach().cpu().clone().numpy()
        outputs_numpy = outputs.detach().cpu().clone().numpy()
        if sketch_feature is None:
            sketch_labels = labels_numpy
            sketch_feature = outputs_numpy
        else:
            sketch_feature=np.concatenate((sketch_feature,outputs_numpy),axis=0)
            sketch_labels=np.concatenate((sketch_labels,labels_numpy),axis=0)
        # print("==> test samplses [%d/%d]" % (batch_idx+1, np.ceil(sketch_num / args.batch_size)))
    
    for batch_idx, (data, labels) in tqdm((enumerate(viewloader)), total = len(viewloader), leave=True, desc="Extracting view features"):
        data = np.stack(data, axis=1)
        data = torch.from_numpy(data)
        if use_gpu:
            data, labels = data.to(device), labels.to(device)
            
        with torch.no_grad():
            output = view_model.forward(data)
            mu_embeddings= classifier.forward(output)
        #mu_embeddings,logits = classifier.forward(output)

        outputs = nn.functional.normalize(mu_embeddings, dim=1)

        #logits = classifier.forward(outputs)
        labels_numpy = labels.detach().cpu().clone().numpy()
        outputs_numpy = outputs.detach().cpu().clone().numpy()
        if view_feature is None:
            view_labels = labels_numpy
            view_feature = outputs_numpy
        else:
            view_feature=np.concatenate((view_feature,outputs_numpy),axis=0)
            view_labels=np.concatenate((view_labels,labels_numpy),axis=0)
        # print("==> test samplses [%d/%d]" % (batch_idx+1, np.ceil(view_num / args.batch_size)))

    feature_data = {'sketch_feature': sketch_feature, 'sketch_labels': sketch_labels,
                    'view_feature': view_feature, 'view_labels': view_labels}
    distance_matrix = cal_cosine_distance(sketch_feature, view_feature)
    Av_NN, Av_FT, Av_ST, Av_E, Av_DCG, Av_Precision = evaluation_metric(distance_matrix, sketch_labels, view_labels,
                                                                        'cosine')
    print("NN:", Av_NN.mean())
    print("FT:", Av_FT.mean())
    print("ST:", Av_ST.mean())
    print("E:", Av_E.mean())
    print("DCG:", Av_DCG.mean())
    print("mAP", Av_Precision.mean())
    torch.save(feature_data,config.features.test_feat_dir)
    #torch.save(sketch_uncer,"sketch_uncertainty.mat")
    #torch.save(dist,'baseline_dist.mat')


if __name__ == '__main__':
    main()
