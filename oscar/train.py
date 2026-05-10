import torch, math, time, argparse, os, sys
import json
from core.model import gen_model
import random, losses
import numpy as np
import pandas as pd
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import pickle
from torch.optim.lr_scheduler import StepLR
from tqdm import trange

seed = 1
random.seed(seed)
np.random.seed(seed)


### load hyperparameter settings
train_config_file = sys.argv[1]
print(">>> load config:", train_config_file)
with open(train_config_file) as fin:
    train_config = json.load(fin)

model_name = train_config["model_name"]
valid_file = train_config["valid_file"]
train_file = train_config["train_file"]
outpath = train_config["outpath"]
batch_size = train_config["batch_size"]
learning_rate = train_config["learning_rate"]
weight_decay = train_config["weight_decay"]
momentum = train_config["momentum"]
num_epoch = train_config["epoch"]
num_webs = train_config["num_webs"]
gpu_id = train_config["gpu_id"]
emb_size = train_config["emb_size"]
threshold = train_config["threshold"]
beta = train_config["beta"]
description = train_config["description"]
scenario = outpath.split("/")[-1]
outpath = os.path.join(outpath, description)
os.makedirs(outpath, exist_ok=True)


os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)    


def load_array(features, labels, batch_size, is_train = True):
    dataset = torch.utils.data.TensorDataset(features, labels)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=is_train, num_workers=8)




## load model
model_config_file = f"config/model/{model_name}.json"
with open(model_config_file) as fin:
    model_config = json.load(fin)

model = gen_model(**model_config)
model.cuda()
print(">>>model loaded...")
print("parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


criterion = losses.MultiLoss(nb_classes=num_webs, emb_size=emb_size, threshold=threshold, beta=beta).cuda()
param_groups = [
    {'params': model.parameters(), 'lr':float(learning_rate) * 1},
]
param_groups.append({'params': criterion.proxies, 'lr':float(learning_rate) * 100})    ### Proxies are added to the parameters and updated with the model's parameters through the same optimizer. 


### prepare data in npz format
train_data = np.load(train_file)
train_X = train_data['direction']
train_Y = train_data['Y']
valid_data = np.load(valid_file)
valid_X = valid_data['direction']
valid_Y = valid_data['Y']
print("train data:", train_X.shape, train_Y.shape)
print("valid data:", valid_X.shape, valid_Y.shape)

train_iter = load_array(torch.from_numpy(train_X), torch.from_numpy(train_Y), batch_size, True)
valid_iter = load_array(torch.from_numpy(valid_X), torch.from_numpy(valid_Y), batch_size, False)
LOSS = 1000


optimizer = optim.SGD(param_groups, lr=learning_rate, weight_decay=weight_decay, momentum=momentum, nesterov=True)
writer = SummaryWriter(os.path.join("runs", f"{scenario}/{description}"))

### model training
for epoch in trange(num_epoch):
    model.train()
    train_loss = []
    valid_loss = []
    for batch_idx, (x, y) in enumerate(train_iter):
        batch_X = torch.tensor(x[:, np.newaxis], dtype=torch.float32).cuda()
        m = model(batch_X)
        loss = criterion(m, y.cuda())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)
        optimizer.step()
        train_loss.append(loss.data.cpu().numpy())

    
    for batch_idx, (x, y) in enumerate(valid_iter):
        batch_X = torch.tensor(x[:, np.newaxis], dtype=torch.float32).cuda()
        m = model(batch_X)
        loss = criterion(m, y.cuda())
        valid_loss.append(loss.data.cpu().numpy())
    
    print(
        'Train Epoch: {} train Loss: {:.6f}   valid Loss: {:.6f}'.format(
            epoch,  np.mean(train_loss),  np.mean(valid_loss)))
    writer.add_scalar("train loss", np.mean(train_loss), epoch)
    writer.add_scalar("valid loss", np.mean(valid_loss), epoch)


    ### save the models with the best loss on validation datasets
    if np.mean(valid_loss) < LOSS:
        LOSS = np.mean(valid_loss)
        torch.save(model.state_dict(), os.path.join(outpath, f'{model_name}.pth'))
        proxies = criterion.proxies.data

### save proxies
proxies =  proxies.cpu().detach().numpy()
proxies_file = os.path.join(outpath, "proxies.pickle")
with open(proxies_file, 'wb') as file:
    pickle.dump(proxies, file)