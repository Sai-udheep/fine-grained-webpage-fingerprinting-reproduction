import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np



class MultiLoss(torch.nn.Module):
    def __init__(self, nb_classes, emb_size, threshold = 0.1, beta = 4.5):
        torch.nn.Module.__init__(self)
        # Hyp Loss Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, emb_size).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.threshold = threshold
        self.beta = beta
        
    def forward(self, embedding, labels):
        beta = self.beta
        threshold = self.threshold       
        pos_one_hot = labels

        ### proxy-based loss
        cos_simi = F.normalize(embedding, p = 2, dim = 1).mm(F.normalize(self.proxies, p = 2, dim = 1).T)
        pos_loss = 1 - cos_simi
        neg_loss = F.relu(cos_simi - threshold)


        pos_num = len(pos_one_hot.nonzero())
        neg_num = len((pos_one_hot == 0).nonzero())
        pos_term = torch.where(pos_one_hot  ==  1, pos_loss.to(torch.float32), torch.zeros_like(cos_simi).to(torch.float32)).sum() / pos_num
        neg_term = torch.where(pos_one_hot  ==  0, neg_loss.to(torch.float32), torch.zeros_like(cos_simi).to(torch.float32)).sum() / neg_num
        # return pos_term + neg_term   ### merely proxy-based loss

        ### sample-based loss
        if beta > 0:
            index = labels.sum(dim = 1) > 1
            selected_sample_labels = labels[index].float()
            selected_samples = embedding[index]
            label_sim = selected_sample_labels.mm(selected_sample_labels.T)
            if len((label_sim == 0).nonzero()) == 0:
                sample_term = 0
            else:
                sample_simi = F.normalize(selected_samples, p = 2, dim = 1).mm(F.normalize(selected_samples, p = 2, dim = 1).T)
                neg_sample = beta * F.relu(sample_simi - threshold)
                sample_term = torch.where(label_sim == 0, neg_sample, torch.zeros_like(sample_simi)).sum() / len((label_sim == 0).nonzero())
        else:
            sample_term = 0
        # return sample_term  ### merely sample-based loss
        return pos_term + neg_term + sample_term
    
    def get_proxies(self):
        return self.proxies
