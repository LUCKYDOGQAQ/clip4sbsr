# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import sys
sys.path.append('.')

import numpy as np
import time
import torch
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from utils.metric import cal_euc_distance, cal_cosine_distance, evaluation_metric

from pathlib import Path
import yaml
from easydict import EasyDict

# parser = argparse.ArgumentParser("Retrieval Evaluation")

# parser.add_argument('--class-sorting-file', type=str, default='../class_sorting/sketch_class_sorting.mat',
#                     help="class sorting  flie of test sketches, .mat file")

# parser.add_argument('--distance-type', type=str, choices=['cosine','euclidean'],default='cosine')
# parser.add_argument('--num-testsketch-samples', type=int, default=171*30)
# # parser.add_argument('--num-classes', type=int, default=171)
# parser.add_argument('--num-view-samples', type=int, default=8987)

# parser.add_argument('--feat-file', type=str,
#                     default='feature.mat',
#                     help="features flie of test sketches, .mat file")

# args = parser.parse_args()

parser = argparse.ArgumentParser("Retrieval Evaluation")
parser.add_argument('-c', '--config', help="running configurations", type=str, required=True)
with open(parser.parse_args().config, 'r', encoding='utf-8') as r:
    config = EasyDict(yaml.safe_load(r))


def get_feat_and_labels(feat_file):
    """" read the features and labels of sketches and 3D models
    Args:
        test_sketch_feat_file: features flie of test sketches, it is .mat file
        view_feat_flie: features flie of view images of 3d models
    """
    features = torch.load(feat_file)


    sketch_feature = features['sketch_feature']
    sketch_label = features['sketch_labels']
    #sketch_predict_label = sket_data_features['predict_label']
    """
    sketch_feature = sket_data_features['view_feature']
    print(sketch_feature.shape)
    sketch_label = sket_data_features['view_labels']
    """
    view_feature = features['view_feature']
    view_label = features['view_labels']


    return sketch_feature, sketch_label, view_feature, view_label


def main():
    sketch_feature, sketch_label, view_feature, view_label = get_feat_and_labels(config.feat_file)
    # sketch_label=np.expand_dims(sketch_label,1)
    # view_label = np.expand_dims(view_label, 1)
    if config.distance_type == 'euclidean':
        distance_matrix = cal_euc_distance(sketch_feature,view_feature)
    elif config.distance_type == 'cosine':
        distance_matrix = cal_cosine_distance(sketch_feature,view_feature)

    # print(sketch_label)
    # if MODE=='CLF':
    #     for i in range(distance_matrix.shape[0]):
    #         distance_matrix[i,np.where(view_label==sketch_predict_label[i])[0]]+=1000
    # elif MODE=='UP':
    #     for i in range(distance_matrix.shape[0]):
    #         distance_matrix[i,np.where(view_label==sketch_label[i])[0]]+=1000


    distance_matrix_data = {"distance_matrix":distance_matrix}
    torch.save(distance_matrix_data,"./output/distance_matrix.mat")

    Av_NN, Av_FT, Av_ST, Av_E, Av_DCG, Av_Precision = evaluation_metric(distance_matrix,sketch_label, view_label,config.distance_type)

    torch.save(Av_Precision,'./output/precison.mat')
    print("NN:", Av_NN.mean())
    print("FT:", Av_FT.mean())
    print("ST:", Av_ST.mean())
    print("E:", Av_E.mean())
    print("DCG:", Av_DCG.mean())
    print("mAP", Av_Precision.mean())


if __name__ == '__main__':
    main()