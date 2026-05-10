python core/deal-data/augment.py -s 800
python train.py config/train/800.json
python feature_transformation.py config/feature_transformation/800.json
python knn.py -s 800