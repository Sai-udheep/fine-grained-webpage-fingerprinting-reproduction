import numpy as np
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import argparse

feat_length = 512
PADDING = 2000
nb_num = 40

parser = argparse.ArgumentParser(description='Classification')
parser.add_argument("-d", '--description', default="Oscar_augment", type=str, help='Model name')
parser.add_argument("-s", '--scenario', default="closed-world", choices=['closed-world', 'open-world', '700', '800', '900'], type=str, help='Evaluation scenario')
args = parser.parse_args()
base_dir = f"result/{args.scenario}/{args.description}"

if args.scenario == "closed-world":
    num_labels = 1000
elif args.scenario == "open-world":
    num_labels = 1001
else:
    num_labels = int(args.scenario)



### extract proxy label, note that each proxy has a single label
def extract_proxy_label(original_label):
    temp_label = np.zeros((len(original_label), num_labels), dtype=np.int8)
    for i in range(len(original_label)):
        temp_label[i][original_label[i]] = 1
    return temp_label


### metric calculation
def countpk(k, label, predict_dict):
    return len(set(label) & set(predict_dict)) / k


def count_apk(k, real_label, predict):
    apk = 0
    for j in range(k):
        apk += countpk(j+1, real_label, predict[:(j+1)])
    apk = apk / min(len(real_label), k)
    return apk


### load proxies and data
proxy_file = os.path.join(base_dir, "proxies.pickle")
train_file = os.path.join(base_dir, "train.npz")
test_file = os.path.join(base_dir, "test.npz")
train_data = np.load(train_file)
train_X = train_data['X']
train_Y = train_data['Y']
test_data = np.load(test_file)
test_X = test_data['X']
test_Y = test_data['Y']
proxy_X = pd.read_pickle(proxy_file)
proxy_X = np.array(pd.DataFrame(proxy_X).astype(float))
proxy_labels = extract_proxy_label(np.array(range(0, num_labels)))
print("train_instances:", train_X.shape, train_Y.shape)
print("test_instances:", test_X.shape, test_Y.shape)


### k-NN model training
model1 = KNeighborsClassifier(n_neighbors=nb_num, weights='distance', p=2, metric='cosine', algorithm='brute')  # sample model
model2 = KNeighborsClassifier(n_neighbors=nb_num, weights='distance', p=2, metric='cosine', algorithm='brute')  # proxy label
model1.fit(train_X, train_Y)
model2.fit(proxy_X, proxy_labels)
predicts1 = model1.predict_proba(test_X)
predicts2 = model2.predict_proba(test_X)


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
outfile = os.path.join(base_dir, f"res-proxy{nb_num}-{nb_num}neighbor")

with open(outfile, "w") as file:
    for i in range(test_X.shape[0]):
        label = test_Y[i]
        real_label = np.nonzero(label)[0]
        file.write("label: ")
        for ll in real_label:
            file.write(f"{ll} ")
        if len(real_label) == 0:
            continue
        temp_res = {}
        for j in range(num_labels):
            # temp_res[j] = 2 * predicts1[j][i][1] + predicts2[j][i][1]
            temp_res[j] = 2 * (1-predicts1[j][i][0]) + predicts2[j][i][1]
        res = sorted(temp_res.items(), key=lambda x: x[1], reverse=True)
        predict_top_labels = []
        file.write("predict:")
        for prob in res:
            temp_label = prob[0]
            predict_top_labels.append(prob[0])
            file.write(f" {prob[0]}-{prob[1]}")
        file.write("\n")
        recall5 += countpk(len(real_label), real_label, predict_top_labels[:5])
        recall10 += countpk(len(real_label), real_label, predict_top_labels[:10])       
        recall15 += countpk(len(real_label), real_label, predict_top_labels[:15])
        recall20 += countpk(len(real_label), real_label, predict_top_labels[:20])
        recall25 += countpk(len(real_label), real_label, predict_top_labels[:25])
        recall30 += countpk(len(real_label), real_label, predict_top_labels[:30])
        ap1 += count_apk(1, real_label, predict_top_labels)
        ap2 += count_apk(2, real_label, predict_top_labels)
        ap3 += count_apk(3, real_label, predict_top_labels)
        ap4 += count_apk(4, real_label, predict_top_labels)
        ap5 += count_apk(5, real_label, predict_top_labels)
        

print("recall@5:", recall5 / test_X.shape[0]) 
print("recall@10:", recall10 / test_X.shape[0]) 
print("recall@15:", recall15 / test_X.shape[0]) 
print("recall@20:", recall20 / test_X.shape[0]) 
print("recall@25:", recall25 / test_X.shape[0]) 
print("recall@30:", recall30 / test_X.shape[0])
print("ap@1:", ap1 / test_X.shape[0]) 
print("ap@2:", ap2 / test_X.shape[0]) 
print("ap@3:", ap3 / test_X.shape[0]) 
print("ap@4:", ap4 / test_X.shape[0]) 
print("ap@5:", ap5 / test_X.shape[0]) 
