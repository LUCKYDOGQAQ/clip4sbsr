# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import scipy.io as sio
sys.path.append('../')
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
from reader.view_dataset_reader import MultiViewDataSet
#from sketch_dataset import SketchDataSet
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances

from tqdm import tqdm

parser = argparse.ArgumentParser("feature extraction of sketch images")
# SHREC13
# parser.add_argument('--sketch-datadir', type=str, default='E:/3d_retrieval/Dataset/Shrec13_ZS2/13_sketch_test_picture')
# parser.add_argument('--view-datadir', type=str, default='E:/3d_retrieval/Dataset/Shrec13_ZS2/13_view_render_test_img')
# SHREC14
parser.add_argument('--sketch-datadir', type=str, default='E:/3d_retrieval/Dataset/Shrec14_ZS2/14_sketch_test_picture')
parser.add_argument('--view-datadir', type=str, default='E:/3d_retrieval/Dataset/Shrec14_ZS2/14_view_render_test_img')
parser.add_argument('--workers', default=5, type=int,
                    help="number of data loading workers (default: 0)")

# test
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--num-classes', type=int, default=133)

# misc
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_false')
parser.add_argument('--model', default="openai/clip-vit-base-patch32")
parser.add_argument('--pretrain', type=bool, choices=[True, False], default=True)
parser.add_argument('--uncer', type=bool, choices=[True, False], default=False)
# features
parser.add_argument('--cnn-feat-dim', type=int, default=512)
parser.add_argument('--feat-dim', type=int, default=256)
parser.add_argument('--test-feat-dir', type=str, default='feature.mat')
parser.add_argument('--train-feat-dir', type=str, default='/home/daiweidong/david/strong_baseline/sketch_modality/shrec_14/train_sketch_picture')
parser.add_argument('--model-dir', type=str, default='./saved_model_14_3Epoch10/')

parser.add_argument('--pattern', type=bool, default=False,
                    help="Extract training data features or test data features,'True' is for train dataset")

args = parser.parse_args()

def cal_cosine_distance(sketch_feat,view_feat):
    distance_matrix = cosine_similarity(sketch_feat,view_feat)

    return distance_matrix

def evaluation_metric(distance_matrix, sketch_label, view_label,dist_type):
    """ calculate the evaluation metric

    Return:
        Av_NN:the precision of top 1 retrieval list
        Av_FT:Assume there are C relavant models in the database,FT is the
        recall of the top C-1 retrieval list
        Av_ST: recall of the top 2(C-1) retrieval list
        Av_E:the retrieval performance of the top 32 model in a retrieval list
        Av_DCG:normalized summed weight value related to the positions of related models
        Av_Precision:mAP1

    """
    from collections import Counter
    np.set_printoptions(suppress=True)
    index_label = np.zeros((view_label.shape[0],))
    # Get the number of samples for each category of 3D models
    view_label_count = {}
    sketch_num = len(sketch_label)
    view_num = len(view_label)
    view_label_list = list(np.reshape(view_label, (view_num,)))
    view_label_set = set(view_label_list)
    count = 0
    for i in view_label_set:
        view_label_count[i] = view_label_list.count(i)
        #print(np.arange(view_label_count[i]))
        index_label[count:count+view_label_count[i]] = np.arange(view_label_count[i])
        #print(index_label[0:315])
        count+=view_label_count[i]
    #print(view_label_count)
    # sketch_num = args.num_testsketch_samples
    # view_num = args.num_view_samples

    P_points = np.zeros((sketch_num, 632));
    Av_Precision = np.zeros((sketch_num, 1));
    Av_NN = np.zeros((sketch_num, 1));
    Av_FT = np.zeros((sketch_num, 1));
    Av_ST = np.zeros((sketch_num, 1));
    Av_E = np.zeros((sketch_num, 1));
    Av_DCG = np.zeros((sketch_num, 1));

    for j in tqdm(range(sketch_num), leave=True, desc="Evaluating"):
        true_label = sketch_label[j]
        view_label_num = view_label_count[true_label]
        # print(view_label_num)
        dist_sort_index = np.zeros((view_num, 1), dtype=int)
        count = 0
        if dist_type == 'euclidean':
            dist_sort_index = np.argsort(distance_matrix[j], axis=0)
        elif dist_type == 'cosine':
            dist_sort_index = np.argsort(-distance_matrix[j],axis = 0)
        dist_sort_index = np.reshape(dist_sort_index, (view_num,))

        view_label_sort = view_label[dist_sort_index]
        index_label_sort = index_label[dist_sort_index]
        #print(view_label_sort)

        b = np.array([[0]])
        view_label_sort = np.insert(view_label_sort, 0, values=b, axis=0)

        G = np.zeros((view_num + 1, 1))
        for i in range(1, view_num + 1):
            if true_label == view_label_sort[i]:
                G[i] = 1
        G_sum = G.cumsum(0)

        NN = G[1]
        FT = G_sum[view_label_num] / view_label_num
        ST = G_sum[2 * view_label_num] / view_label_num

        P_32 = G_sum[32] / 32
        R_32 = G_sum[32] / view_label_num
        if (P_32 == 0) and (R_32 == 0):
            Av_E[j] = 0
        else:
            Av_E[j] = 2 * P_32 * R_32 / (P_32 + R_32)

        # 计算DCG
        NORM_VALUE = 1 + np.sum(1. / np.log2(np.arange(2, view_label_num + 1)))

        m = 1. / np.log2(np.arange(2, view_num + 1))
        m = np.reshape(m, [m.shape[0], 1])

        dcg_i = m * G[2:]
        dcg_i = np.vstack((G[1], dcg_i))
        Av_DCG[j] = np.sum(dcg_i) / NORM_VALUE;

        R_points = np.zeros((view_label_num + 1, 1), dtype=int)

        for n in range(1, view_label_num + 1):
            for k in range(1, view_num + 1):
                if G_sum[k] == n:
                    R_points[n] = k
                    break

        R_points_reshape = np.reshape(R_points, (view_label_num + 1,))

        P_points[j, 0:view_label_num] = np.reshape(G_sum[R_points_reshape[1:]] / R_points[1:], (view_label_num,))

        Av_Precision[j] = np.mean(P_points[j, 0:view_label_num])
        Av_NN[j] = NN
        Av_FT[j] = FT
        Av_ST[j] = ST
        #print(Av_Precision[j])

        #if Av_Precision[j] <=0.99:
            #print(j)
            #print(Av_Precision[j])
            #print("++++++++++++++++++++++++++++")
            #time.sleep(1)
        # if j % 100 == 0:
        #     print("==> test samplses [%d/%d]" % (j, view_num))

    return Av_NN, Av_FT, Av_ST, Av_E, Av_DCG, Av_Precision


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
    sketch_dataloaders = DataLoader(sketch_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    view_dataloaders = DataLoader(view_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    return sketch_dataloaders,view_dataloaders,len(sketch_data),len(view_data)


def main():
    # torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"

    # sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + '.txt'))
    # if use_gpu:
    #     print("Currently using GPU: {}".format(args.gpu))
    #     cudnn.benchmark = True
    #     # torch.cuda.manual_seed_all(args.seed)
    # else:
    #     print("Currently using CPU")
    print(f"Currently using device: {device}")

    sketchloader,viewloader,sketch_num,view_num = get_test_data(args.sketch_datadir,args.view_datadir)
    sketch_model = SketchModel(backbone=args.model)
    view_model = MVCNN(backbone=args.model)
    classifier = Classifier(12,args.cnn_feat_dim,args.num_classes)

    if use_gpu:
        sketch_model =sketch_model.to(device)
        view_model = view_model.to(device)
        classifier = classifier.to(device)     

    # Load model
    # sketch_model.load(args.model_dir+'sketch_lora')
    # view_model.load(args.model_dir + 'view_lora')
    classifier.load_state_dict(torch.load(args.model_dir+'mlp_layer.pth'))
    sketch_model.eval()
    view_model.eval()
    classifier.eval() 


    # Define two matrices to store extracted features
    sketch_feature = None
    sketch_labels = None
    view_feature = None
    view_labels = None

    for batch_idx, (data, labels) in tqdm((enumerate(sketchloader)), total = len(viewloader), leave=True, desc="Extracting sketch features"):
        if use_gpu:
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
    torch.save(feature_data,args.test_feat_dir)
    #torch.save(sketch_uncer,"sketch_uncertainty.mat")
    #torch.save(dist,'baseline_dist.mat')


if __name__ == '__main__':
    main()