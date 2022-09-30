import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
import yaml
import tqdm

import multipoint.datasets as datasets
import multipoint.models as models
import multipoint.utils as utils

def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def main():
    parser = argparse.ArgumentParser(description='Predict the keypoints of an image')
    parser.add_argument('-y', '--yaml-config', default='configs/config_image_pair_dataset_prediction.yaml', help='YAML config file')
    parser.add_argument('-m', '--model-dir', default='model_weights/multipoint', help='Directory of the model')
    parser.add_argument('-v', '--version', default='latest', help='Model version (name of the param file), none for no weights')
    parser.add_argument('-i', '--index', default=0, type=int, help='Index of the sample to predict and show')
    parser.add_argument('-r', '--radius', default=4, type=int, help='Radius of the keypoint circle')
    parser.add_argument('-p', dest='plot', action='store_true', help='If set the prediction the results are displayed')
    parser.add_argument('-e', dest='evaluation', action='store_true', help='If set the evaluation metrics are computed')
    parser.add_argument('-tk', dest='threshold_keypoints', default=4, type=int, help='Distance below which two keypoints are considered a match')
    parser.add_argument('-th', dest='threshold_homography', default=1, type=int, help='Homography correctness threshold')
    parser.add_argument('-s', '--seed', default=0, type=int, help='Seed of the random generators')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.yaml_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(args.model_dir, 'params.yaml'), 'r') as f:
        # overwrite the model params
        config['model'] = yaml.load(f, Loader=yaml.FullLoader)['model']

    # check training device
    device = torch.device("cpu")
    if config['prediction']['allow_gpu']:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print('Predicting on device: {}'.format(device))

    # dataset
    dataset = getattr(datasets, config['dataset']['type'])(config['dataset'])
    loader_dataset = torch.utils.data.DataLoader(dataset, batch_size=config['prediction']['batchsize'],
                                                 shuffle=False, num_workers=config['prediction']['num_worker'])

    # network
    net = getattr(models, config['model']['type'])(config['model'])
    if args.version != 'none':
        weights = torch.load(os.path.join(args.model_dir, args.version + '.model'), map_location=torch.device('cpu'))
        weights = utils.fix_model_weigth_keys(weights)
        net.load_state_dict(weights)
        del weights
    net.to(device)

    # put the network into the evaluation mode
    net.eval()

    with torch.no_grad():
        write_dir = os.path.join('landmark_exp', os.path.basename(os.path.normpath(args.model_dir)))
        os.makedirs(write_dir, exist_ok=True)
        os.makedirs(os.path.join(write_dir, 'debug'), exist_ok=True)
        os.makedirs(os.path.join(write_dir, 'landmarks'), exist_ok=True)

        for idx in tqdm.tqdm(list(range(len(dataset)))):
            data = dataset[idx]
            data = utils.data_to_device(data, device)
            data = utils.data_unsqueeze(data, 0)

            # predict
            out_optical = net(data['optical'])
            out_thermal = net(data['thermal']) # could be optimized to move into one forward pass

            # compute the nms probablity
            if config['prediction']['nms'] > 0:
                out_optical['prob'] = utils.box_nms(out_optical['prob'] * data['optical']['valid_mask'],
                                                    config['prediction']['nms'],
                                                    config['prediction']['detection_threshold'],
                                                    keep_top_k=config['prediction']['topk'],
                                                    on_cpu=config['prediction']['cpu_nms'])
                out_thermal['prob'] = utils.box_nms(out_thermal['prob'] * data['thermal']['valid_mask'],
                                                    config['prediction']['nms'],
                                                    config['prediction']['detection_threshold'],
                                                    keep_top_k=config['prediction']['topk'],
                                                    on_cpu=config['prediction']['cpu_nms'])


            # add homography to data if not available
            if 'homography' not in data['optical'].keys():
                data['optical']['homography'] =  torch.eye(3, dtype=torch.float32).to(device).view(data['optical']['image'].shape[0],3,3)

            if 'homography' not in data['thermal'].keys():
                data['thermal']['homography'] =  torch.eye(3, dtype=torch.float32).to(device).view(data['optical']['image'].shape[0],3,3)

            for i, (optical, thermal,
                    prob_optical, prob_thermal,
                    mask_optical, mask_thermal,
                    H_optical, H_thermal,
                    desc_optical, desc_thermal) in enumerate(zip(data['optical']['image'],
                                                                data['thermal']['image'],
                                                                out_optical['prob'],
                                                                out_thermal['prob'],
                                                                data['optical']['valid_mask'],
                                                                data['thermal']['valid_mask'],
                                                                data['optical']['homography'],
                                                                data['thermal']['homography'],
                                                                out_optical['desc'],
                                                                out_thermal['desc'],)):

                # get the keypoints
                pred_optical = torch.nonzero((prob_optical.squeeze() > config['prediction']['detection_threshold']).float())
                pred_thermal = torch.nonzero((prob_thermal.squeeze() > config['prediction']['detection_threshold']).float())
                kp_optical = [cv2.KeyPoint(c[1], c[0], args.radius) for c in pred_optical.cpu().numpy().astype(np.float32)]
                kp_thermal = [cv2.KeyPoint(c[1], c[0], args.radius) for c in pred_thermal.cpu().numpy().astype(np.float32)]

                # get the descriptors
                if desc_optical.shape[1:] == prob_optical.shape[1:]:
                    # classic descriptors, directly take values
                    desc_optical_sampled = desc_optical[:, pred_optical[:,0], pred_optical[:,1]].transpose(0,1)
                    desc_thermal_sampled = desc_thermal[:, pred_thermal[:,0], pred_thermal[:,1]].transpose(0,1)
                else:
                    H, W = data['optical']['image'].shape[2:]
                    desc_optical_sampled = utils.interpolate_descriptors(pred_optical, desc_optical, H, W)
                    desc_thermal_sampled = utils.interpolate_descriptors(pred_thermal, desc_thermal, H, W)

                # match the keypoints
                desc1 = desc_optical_sampled.cpu().numpy()
                desc2 = desc_thermal_sampled.cpu().numpy()

                if True:
                    # draw predictions and ground truth on image
                    out_optical = cv2.cvtColor((np.clip(optical.cpu().numpy().squeeze(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)
                    out_thermal = cv2.cvtColor((np.clip(thermal.cpu().numpy().squeeze(), 0.0, 1.0) * 255.0).astype(np.uint8),cv2.COLOR_GRAY2RGB)

                    out_optical = cv2.drawKeypoints(out_optical,
                                                    kp_optical,
                                                    outImage=np.array([]),
                                                    color=(0, 255, 0),
                                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    out_thermal = cv2.drawKeypoints(out_thermal,
                                                    kp_thermal,
                                                    outImage=np.array([]),
                                                    color=(0, 255, 0),
                                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    cv2.imwrite(os.path.join(write_dir, 'debug', os.path.basename(data['on_path'])), out_optical)
                    cv2.imwrite(os.path.join(write_dir, 'debug', os.path.basename(data['off_path'])), out_thermal)
                print(data['on_path'], data['off_path'])
                print(optical.shape, thermal.shape)
                print(pred_optical.shape, desc1.shape)

if __name__ == "__main__":
    main()
