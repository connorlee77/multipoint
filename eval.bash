python predict_align_image_pair.py --yaml-config=configs/config_image_pair_dataset_prediction.yaml --model-dir=model_weights/seasonal_multipoint_no_id_mask --version=e2490 --index=-1 --radius=1 -e -tk=4 -th=10

python predict_align_image_pair.py --yaml-config=configs/config_image_pair_dataset_prediction.yaml --model-dir=model_weights/seasonal_multipoint_id_mask --version=e2490 --index=-1 --radius=1 -e -tk=4 -th=10

python predict_align_image_pair.py --yaml-config=configs/config_image_pair_dataset_prediction.yaml --model-dir=model_weights/seasonal_multipoint_id_mask_valid_mask --version=e2490 --index=-1 --radius=1 -e -tk=4 -th=10

python predict_align_image_pair.py --yaml-config=configs/config_image_pair_dataset_prediction.yaml --model-dir=model_weights/superpoint --index=-1 --radius=1 -e -tk=4 -th=10 --version v1

python predict_align_image_pair.py --yaml-config=configs/config_image_pair_dataset_prediction.yaml --model-dir=model_weights/sift --index=-1 --radius=1 -e -tk=4 -th=10 --version none

