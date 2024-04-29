import torch 
import torch.nn as nn 

class TripletCenterLoss:
    def __init__(self, margin=0):
        self.margin = margin 
        self.ranking_loss = nn.MarginRankingLoss(margin=margin) 
   
    def __call__(self, inputs, targets, centers): 
        batch_size = inputs.size(0) 
        centers_batch = centers[targets] # shape=[batch_size, feature_dim]

        # compute pairwise distances between input features and corresponding centers 
        dist = torch.cdist(inputs, centers_batch) # shape=[batch_size, batch_size]

        # for each anchor, find the hardest positive and negative 
        mask = targets.unsqueeze(0) == targets.unsqueeze(1)
        dist_ap, dist_an = [], [] 
        for i in range(batch_size): # for each sample, we compute distance 
            dist_ap.append(dist[i][mask[i] == True].max()) # mask[i]: positive samples of sample i
            dist_an.append(dist[i][mask[i] == False].min()) # mask[i]==0: negative samples of sample i 

        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        # generate a new label y
        # compute ranking hinge loss 
        y = torch.ones_like(dist_an)
        # y_i = 1, means dist_an > dist_ap + margin will cause loss be zero
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # prec = (dist_an.data > dist_ap.data).sum() * 1. / batch_size # normalize data by batch size 
        return loss