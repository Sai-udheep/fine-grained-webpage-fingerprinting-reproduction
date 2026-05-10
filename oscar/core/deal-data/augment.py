import pandas as pd
import random
import numpy as np
import pickle
import os
from tqdm import trange
import argparse

random.seed(74)
np.random.seed(74)
PADDING = 2000


### read original data
parser = argparse.ArgumentParser(description='Data augmentation')
parser.add_argument("-s", '--scenario', default="closed-world", choices=['closed-world', 'open-world', '700', '800', '900'], type=str, help='Evaluation scenario')
args = parser.parse_args()
base_dir = f"datasets/{args.scenario}"
data = np.load(os.path.join(base_dir, "train.npz"))
direction = data['direction']
time = data['time']
Y = data['Y']
print("original data shape:", direction.shape, time.shape, Y.shape)
if args.scenario == "closed-world":
    num_webs = 1000
elif args.scenario == "open-world":
    num_webs = 1001
else:
    num_webs = int(args.scenario)


#### Intra-sample augmentation based on packet exchanging
exchanged_direction = np.zeros((direction.shape[0], 10000), dtype=np.int8)
average_burst = 0
for i in trange(direction.shape[0]):
    burst_sequence = []
    burst_dir = []
    last_index = 0
    ### burst extraction
    for j in range(1, 10000):
        if direction[i][j] != direction[i][j-1]:
            burst_sequence.append(j - last_index)
            burst_dir.append(direction[i][j-1])
            last_index = j
        if direction[i][j] == 0:
            break
    if direction[i][9999] != 0:
        burst_sequence.append(10000 - last_index)
        burst_dir.append(direction[i][9999])
    average_burst += len(burst_sequence)

    ### random sample bursts for augmentation
    ex_pos = random.sample(range(len(burst_sequence)), int(len(burst_sequence)*0.05))
    ex_pos = sorted(ex_pos)

    ### exchange the selected bursts with their subsequent bursts
    for p in ex_pos:
        try:
            burst_sequence[p], burst_sequence[p+1] = burst_sequence[p+1], burst_sequence[p]
            burst_dir[p], burst_dir[p+1] = burst_dir[p+1], burst_dir[p] 
        except:
            pass
    index = 0
    for j in range(len(burst_sequence)):
        exchanged_direction[i][index:index+burst_sequence[j]] = burst_dir[j]
        index += burst_sequence[j]

# print("average burst number:", average_burst/data.shape[0])
new_data = np.concatenate([direction, exchanged_direction], axis=0)
new_Y = np.concatenate([Y, Y], axis=0)


### inter-sample augmentation based traffic combining
combined_direction = np.zeros((direction.shape[0]//2,10000), dtype=np.int8)
### shuffle data
indices = np.random.permutation(len(direction))
direction = direction[indices]
time = time[indices]
Y = Y[indices]
combined_Y = np.zeros((direction.shape[0]//2, num_webs), dtype=np.int8)
for i in trange(0,direction.shape[0],2):
    if i == direction.shape[0]-1:
        break
    index1 = 0
    index2 = 0
    for j in range(10000):
        if time[i][index1] > time[i+1][index2]:
            combined_direction[i//2][j] = direction[i+1][index2]
            index2 += 1
        else:
            combined_direction[i//2][j] = direction[i][index1]
            index1 += 1
    combined_Y[i//2] = np.maximum(Y[i], Y[i+1])
new_data = np.concatenate([new_data, combined_direction], axis=0)
new_Y = np.concatenate([new_Y, combined_Y], axis=0)

combined_direction = np.zeros((direction.shape[0]//2,10000), dtype=np.int8)
### shuffle data
indices = np.random.permutation(len(direction))
direction = direction[indices]
time = time[indices]
Y = Y[indices]
combined_Y = np.zeros((direction.shape[0]//2, num_webs), dtype=np.int8)
for i in trange(0,direction.shape[0],2):
    if i == direction.shape[0]-1:
        break
    index1 = 0
    index2 = 0
    for j in range(10000):
        if time[i][index1] > time[i+1][index2]:
            combined_direction[i//2][j] = direction[i+1][index2]
            index2 += 1
        else:
            combined_direction[i//2][j] = direction[i][index1]
            index1 += 1
    combined_Y[i//2] = np.maximum(Y[i], Y[i+1])
new_data = np.concatenate([new_data, combined_direction], axis=0)
new_Y = np.concatenate([new_Y, combined_Y], axis=0)

print(new_data.dtype, new_Y.dtype)
print("augmented data shape:", new_data.shape, new_Y.shape)


### save augmented data
outpath = os.path.join(base_dir, "train_augment.npz")
np.savez(outpath, direction=new_data, Y=new_Y)