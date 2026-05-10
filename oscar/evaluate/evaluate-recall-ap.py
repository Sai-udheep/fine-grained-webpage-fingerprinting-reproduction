import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pickle
import math

NB_CLASSES =  1000

### load the result file that you want to evaluate
path = f"result/closed-world/Oscar_augment/res-proxy40-40neighbor"

def to_vector(labels):
    label_vector = np.zeros(NB_CLASSES, dtype=int)
    for i in labels:
        label_vector[i] = 1
    return label_vector

def countpk(k, label, predict_dict):
    return len(set(label) & set(predict_dict)) / k


def count_apk(k, true_labels, predict):
    apk = 0
    for j in range(k):
        apk += countpk(j+1, true_labels, predict[:(j+1)])
    apk = apk / min(len(true_labels), k)
    return apk



recall5 = 0
recall10 = 0
recall15 = 0
recall20 = 0
recall25 = 0
recall30 = 0
ap1 = 0
ap2 = 0
ap3 = 0
ap4 = 0
ap5 = 0
with open(path, "r") as ff:
    lines = ff.readlines()
    for line in lines:
        line  = line[:-1]
        res = line.split(" ")
    
        true_labels = []
        predict = []
        for i in range(len(res)):
            if res[i] == "label:":
                continue
            if res[i] == "predict:":
                ss = i+1
                break
            true_labels.append(int(res[i]))
        for i in range(ss, len(res)):
            temp = res[i].split("-")
            predict.append(int(temp[0]))
        recall5 += countpk(len(true_labels), true_labels, predict[:5])
        recall10 += countpk(len(true_labels), true_labels, predict[:10])
        recall15 += countpk(len(true_labels), true_labels, predict[:15])
        recall20 += countpk(len(true_labels), true_labels, predict[:20])
        recall25 += countpk(len(true_labels), true_labels, predict[:25])
        recall30 += countpk(len(true_labels), true_labels, predict[:30])
        ap1 += count_apk(1, true_labels, predict)
        ap2 += count_apk(2, true_labels, predict)
        ap3 += count_apk(3, true_labels, predict)
        ap4 += count_apk(4, true_labels, predict)
        ap5 += count_apk(5, true_labels, predict)
        
        
print("recall@5:", recall5 / len(lines)) 
print("recall@10:", recall10 / len(lines)) 
print("recall@15:", recall15 / len(lines)) 
print("recall@20:", recall20 / len(lines)) 
print("recall@25:", recall25 / len(lines)) 
print("recall@30:", recall30 / len(lines)) 
print("ap@1:", ap1 / len(lines)) 
print("ap@2:", ap2 / len(lines)) 
print("ap@3:", ap3 / len(lines)) 
print("ap@4:", ap4 / len(lines)) 
print("ap@5:", ap5 / len(lines)) 
