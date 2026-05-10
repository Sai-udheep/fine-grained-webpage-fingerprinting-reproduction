
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pickle
import math
from sklearn import metrics

NB_CLASSES =  1000
THRESHOLD = 0.3
path = "result/closed-world/Oscar_augment/res-proxy40-40neighbor"

with open(path, "r") as ff:
    lines = ff.readlines()
    all_res = np.zeros((len(lines), NB_CLASSES))
    all_pred = np.zeros((len(lines), NB_CLASSES))
    for j in range(len(lines)):
        line  = lines[j][:-1]
        res = line.split(" ")
    
        predict = {}
        for i in range(len(res)):
            if res[i] == "label:":
                continue
            if res[i] == "predict:":
                ss = i+1
                break
            all_res[j][int(res[i])] = 1
        maxx = float(res[ss].split("-")[1])
        for i in range(ss, len(res)):
            temp = res[i].split("-")
            all_pred[j][int(temp[0])] = float(temp[1])
print(metrics.f1_score(all_res, all_pred > THRESHOLD, average='micro'))