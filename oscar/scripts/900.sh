python core/deal-data/augment.py -s 900
python train.py config/train/900.json
python feature_transformation.py config/feature_transformation/900.json
python knn.py -s 900