python core/deal-data/augment.py -s closed-world
python train.py config/train/closed-world.json
python feature_transformation.py config/feature_transformation/closed-world.json
python knn.py -s closed-world