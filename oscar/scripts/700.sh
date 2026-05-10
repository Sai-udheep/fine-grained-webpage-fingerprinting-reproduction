python core/deal-data/augment.py -s 700
python train.py config/train/700.json
python feature_transformation.py config/feature_transformation/700.json
python knn.py -s 700