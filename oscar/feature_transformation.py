import torch
import os
import json
import sys
import pandas as pd
import numpy as np
from core.model import gen_model
import pickle
from sklearn.model_selection import train_test_split
import time

start_time = time.time()
### load hyperparameter settings
test_config_file = sys.argv[1]
print(">>> load config:", test_config_file)
with open(test_config_file) as fin:
    test_config = json.load(fin)


model_name = test_config["model_name"]
train_file = test_config["train_file"]
test_file = test_config["test_file"]
outpath = test_config["outpath"]
PADDING = test_config["padding"]
emb_size = test_config["emb_size"]
description = test_config["description"]
batch_size = test_config["batch_size"]

outpath = os.path.join(outpath, description)
network_path = os.path.join(outpath, f"{model_name}.pth")
model_config_file = f"config/model/{model_name}.json"



with open(model_config_file) as fin:
    model_config = json.load(fin)



def feature_transformation(all_samples):
    sample_represent = np.zeros((all_samples.shape[0], emb_size), dtype=float)
    ss = 0
    while True:
        ee = ss + batch_size
        if ee >= all_samples.shape[0]:
            ee = all_samples.shape[0]
        input_samples = all_samples[ss:ee]
        input_samples = torch.tensor(input_samples[:,np.newaxis], dtype=torch.float32)

        input_sample_embs = model(input_samples)
        input_sample_embs = input_sample_embs.detach().numpy()
        sample_represent[ss:ee] = input_sample_embs
        if ee == all_samples.shape[0]:
            break
        ss += batch_size
    return sample_represent



### load the trained model
model = gen_model(**model_config)
model.load_state_dict(torch.load(network_path, map_location='cpu'))
print("model loaded...")


### prepare data
train_data = np.load(train_file)
train_X = train_data['direction']
train_Y = train_data['Y']
test_data = np.load(test_file)
test_X = test_data['direction']
test_Y = test_data['Y']
print("train data:", train_X.shape, train_Y.shape)
print("test data:", test_X.shape, test_Y.shape)

### convert samples to the transformed feature space
repre_train = feature_transformation(train_X)
repre_test = feature_transformation(test_X)

train_file = os.path.join(outpath, f"train.npz")
np.savez(train_file, X=repre_train, Y=train_Y)

test_file = os.path.join(outpath, f"test.npz")
np.savez(test_file, X=repre_test, Y=test_Y)


end_time = time.time()
print(">>> time elapsed: ", end_time - start_time)