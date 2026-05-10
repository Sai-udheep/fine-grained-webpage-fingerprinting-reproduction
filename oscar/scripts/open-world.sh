python core/deal-data/augment.py -s open-world
python train.py config/train/open-world.json
python feature_transformation.py config/feature_transformation/open-world.json
python knn.py -s open-world