# Towards Fine-Grained Webpage Fingerprinting at Scale
This repository contains the source code for our paper "Towards Fine-Grained Webpage Fingerprinting at Scale".

### Prerequisites
We prototype a WPF attack Oscar using Torch 1.9.0 and Python 3.8. The required packets and the corresponding versions are listed in the file `requirements.txt`. You can install the necessary packages with the following command:
```
pip install -r requirements.txt
```
The training of the feature trasformation model is done on GPU, requiring around 6G of memory.
The feature transformation and webpage identification can be done on CPU.
The data augmentation takes around an hour; the training of the feature transformation model takes around 12h; the feature transformation can be done within an hour; and the webpage identification can be done within a few minutes. 


### Datasets
We collect our webpage datasets under the multi-tab setting. You can decompress the datasets and put them under the `datasets` folder.
Our datasets involve three settings: closed-world, open-world and small-scale (700, 800, 900 webpages) evaluation.
Under the closed-world setting, there are 1,000 monitored webpages, which are represented as 0 to 999. Under the open-world setting, the unmonitored webpages are represented as 1,000.
The traffic data is contained in a npz file, including the direction sequence, the time sequence and the label in the binarized format. The data can be loaded through the following codes.
```
data = np.load(train_file)
direction = data['direction']
time = data['time']
Y = data['Y']
```


Our datasets are divided into the training, validation and testing datasets. The training and validtation sets are used to train a DF-based feature transformation model. Then we use the trained model to transform the samples in the testing set and conduct webpage classification based on the updated proxies and the transformed samples in the training set.
We put the trained model and the proxies under the folder `result`.


We provide the scripts for our experiments in Section 6, and the details are as follows.
### Training
We take the training of Oscar on the closed-world dataset as an example.
First, we generate augmented samples.
```
python core/deal-data/augment.py -s closed-world
```
After data augmentation, we use the augmented datasets to train a feature transformation model based on the multi-label metric learning loss.
```
python train.py config/train/closed-world.json
```
After this, we save the trained feature transformation model and the updated proxies.


### Evaluation
For evaluation, we first load the trained model, transform samples in the training and testing sets and save the transformed samples.
```
python feature_transformation.py config/feature_transformation/closed-world.json
```
Then we achieve webpage identification based on the proxy and sample combined k-NN classifiers. We use multi-label metrics Recall@k and AP@k to evaluate the performance of Oscar.
```
python knn.py -s closed-world
```

### Metric calculation
You can also calculate Recall@k and AP@k from a result file.
```
python evaluate/evaluate-recall-ap.py
```


### Baseline
**k-FP**: https://github.com/jhayes14/k-FP
**DF**: https://github.com/deep-fingerprinting/df
**Tik-Tok**: https://github.com/msrocean/Tik_Tok/
**NetCLR**: https://github.com/SPIN-UMass/Realistic-Website-Fingerprinting-By-Augmenting-Network-Traces
**BAPM**: https://github.com/Xinhao-Deng/Website-Fingerprinting-Library/blob/master/WFlib/models/BAPM.py
**TMWF**: https://github.com/jzx-bupt/TMWF
For baseline methods, we utilized the source codes given by the authors with minor modifications, and the details can be found in our submitted paper.