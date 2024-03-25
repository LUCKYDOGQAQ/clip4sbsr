import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances


def cal_euc_distance(sketch_feat,view_feat):
    distance_matrix = pairwise_distances(sketch_feat,view_feat)

    return distance_matrix
    
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