import numpy as np
import random
import sys
import torch
from torch.utils.data.dataset import Dataset
import h5py
import os
import glob
import multipoint.utils as utils
from .augmentation import augmentation
import cv2
import pandas as pd
import skimage.morphology

def get_percentile_value(percentile_df, feature_layer, p):
    col_name = 'feature_layer_{}'.format(feature_layer)
    p_val = percentile_df.loc[percentile_df['percentiles'] == p, col_name].values[0]
    return p_val

def binarize_mask(mask, p_val, out_shape, dilation_radius=11):
    h, w = out_shape

    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
    mask = mask > p_val
    mask = skimage.morphology.binary_dilation(mask, skimage.morphology.disk(dilation_radius))
    return mask

def xywh2wywy(x, y, w, h):
    return x, y, x + w, y + h

def get_bounding_boxes(mask, min_width):

    bboxes = []
    contours, hierarchies = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)

        if min(w, h) < min_width:
            continue
            
        x1, y1, x2, y2 = xywh2wywy(x, y, w, h)
        bboxes.append([x1, y1, x2, y2])
    
    return bboxes

class SeasonPairDataset(Dataset):
    '''
    Class to load a sample from a given hdf5 file.
    '''
    default_config = {
        'filename': None,
        'keypoints_filename': None,
        'height': -1,
        'width': -1,
        'raw_thermal': False,
        'single_image': True,
        'random_pairs': False,
        'return_name' : True,
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'border_reflect': True,
                'valid_border_margin': 0,
                'mask_border': True,
            },
        }
    }

    def __init__(self, config):
        if config:
            import copy
            self.config = utils.dict_update(copy.copy(self.default_config), config)
        else:
            self.config = self.default_config

        if self.config['dataset_dir'] is None:
            raise ValueError('SeasonsDataset: The dataset filename needs to be present in the config file')

        if self.config['single_image'] and self.config['random_pairs']:
            print('INFO: random_pairs has no influence if single_image is true')

        self.on_files = sorted(glob.glob(os.path.join(self.config['dataset_dir'], 'on', '*')))
        self.off_files = sorted(glob.glob(os.path.join(self.config['dataset_dir'], 'off', '*')))

        self.use_valid_mask = self.config['mask']['use_as_valid_mask']

        self.enable_mask = self.config['mask']['enable']
        if self.enable_mask:
            self.id_mask_dir = self.config['mask']['id_mask_dir']
            self.mask_feature_layer = str(self.config['mask']['feature_layer'])
            mask_percentile = self.config['mask']['percentile']

            mask_percentile_mapping_path = self.config['mask']['percentile_map_path']
            mask_percentile_mapping = pd.read_csv(mask_percentile_mapping_path)
            self.p_val = get_percentile_value(
                mask_percentile_mapping, 
                feature_layer=self.mask_feature_layer, 
                p=mask_percentile)

        self.label_dir = self.config['keypoint_dir']
        missing_labels = []
        self.memberslist = []
        for x, y in zip(self.on_files, self.off_files):
            x_name = os.path.basename(x).replace('_on.png', '')
            y_name = os.path.basename(y).replace('_off.png', '')

            self.memberslist.append(x_name)
            assert x_name == y_name, (x, y)
            if not os.path.exists(os.path.join(self.label_dir, '{}.npy'.format(x_name))):
                missing_labels.append(x_name)

        if len(missing_labels) > 0:
            print('Labels ({}) for the following samples not available: {}'.format(len(missing_labels), missing_labels))

        # extract info from the h5 file
        self.num_files = len(self.on_files)
        print('The dataset ' + self.config['dataset_dir'] + ' contains {} samples'.format(self.num_files))


    def __getitem__(self, index):
        on_img_path = self.on_files[index]
        off_img_path = self.off_files[index]
        # off_img_path = self.on_files[index]
        place_name = self.memberslist[index]
        
        on_img = cv2.imread(on_img_path, 0) / 255.0
        off_img = cv2.imread(off_img_path, 0) / 255.0

        if on_img.shape != off_img.shape:
            raise ValueError('ImagePairDataset: The optical and thermal image must have the same shape')

        # Just assign optical/thermal to integrate with existing code
        optical = on_img 
        thermal = off_img

        kp_path =  os.path.join(self.label_dir, '{}.npy'.format(place_name))
        if os.path.exists(kp_path):
            keypoints = np.load(kp_path)
            if len(keypoints.shape) == 2:
                keypoints = np.flip(keypoints, axis=1)
            else:
                keypoints = np.zeros((0, 2)) 
        else:
            keypoints = np.zeros((0, 2)) 

        # Mask via landmark id network
        bin_on_mask = None
        bin_off_mask = None
        combined_mask = np.ones_like(optical)
        if self.enable_mask:
            on_mask_path = os.path.join(self.id_mask_dir, self.mask_feature_layer, '{}_on.npy'.format(place_name))
            off_mask_path = os.path.join(self.id_mask_dir, self.mask_feature_layer, '{}_off.npy'.format(place_name))

            on_mask = np.load(on_mask_path)
            off_mask = np.load(off_mask_path)

            bin_on_mask = binarize_mask(on_mask, self.p_val, on_img.shape, dilation_radius=7)
            bin_off_mask = binarize_mask(off_mask, self.p_val, off_img.shape, dilation_radius=7)

            combined_mask = np.logical_and(bin_on_mask, bin_off_mask)
            keypoints_mask = utils.generate_keypoint_map(keypoints, on_img.shape)

            keypoints = np.argwhere(np.logical_and(keypoints_mask, combined_mask))

            # if True:
            #     bin_on_mask = np.uint8(255 * bin_on_mask)
            #     bin_off_mask = np.uint8(255 * bin_off_mask)

            #     bin_on_mask = np.stack([bin_on_mask]*3, axis=2)
            #     bin_off_mask = np.stack([bin_off_mask]*3, axis=2)

            #     rgb_img_on = cv2.imread(on_img_path, 1)
            #     rgb_img_off = cv2.imread(off_img_path, 1)

            #     rgb_img_off = cv2.addWeighted(rgb_img_off, 0.6, bin_off_mask, 0.3, 0, None)
            #     rgb_img_on = cv2.addWeighted(rgb_img_on, 0.6, bin_on_mask, 0.3, 0, None)

            #     print(keypoints.shape)
            #     for pt in keypoints:
            #         y, x = pt[:2].astype(int)    
            #         cv2.circle(rgb_img_off, (x, y), 1, (255, 0, 0), -1, lineType=16)
            #         cv2.circle(rgb_img_on, (x, y), 1, (0, 255, 0), -1, lineType=16)

            #     cv2.imwrite('test_imgs_no_mask/{}.png'.format(place_name), np.hstack([rgb_img_off, rgb_img_on]))
            #     exit(0)

        # subsample images if requested
        if self.config['height'] > 0 or self.config['width'] > 0:
            if self.config['height'] > 0:
                h = self.config['height']
            else:
                h = thermal.shape[0]

            if self.config['width'] > 0:
                w = self.config['width']
            else:
                w = thermal.shape[1]

            if w > thermal.shape[1] or h > thermal.shape[0]:
                raise ValueError('ImagePairDataset: Requested height/width exceeds original image size')

            # subsample the image
            i_h = random.randint(0, thermal.shape[0]-h)
            i_w = random.randint(0, thermal.shape[1]-w)

            optical = optical[i_h:i_h+h, i_w:i_w+w]
            thermal = thermal[i_h:i_h+h, i_w:i_w+w]

            ### Custom valid mask
            bin_off_mask = bin_off_mask[i_h:i_h+h, i_w:i_w+w]
            bin_on_mask = bin_on_mask[i_h:i_h+h, i_w:i_w+w]
            combined_mask = combined_mask[i_h:i_h+h, i_w:i_w+w]

            # print(bin_off_mask.sum(), bin_on_mask.sum())

            if keypoints is not None:
                # shift keypoints
                keypoints = keypoints - np.array([[i_h,i_w]])

                # filter out bad ones
                keypoints = keypoints[np.logical_and(
                                      np.logical_and(keypoints[:,0] >=0,keypoints[:,0] < h),
                                      np.logical_and(keypoints[:,1] >=0,keypoints[:,1] < w))]

        else:
            h = thermal.shape[0]
            w = thermal.shape[1]

        out = {}
        out['has_viable_matching_regions'] = combined_mask.sum() > 0.05*h*w
        if self.config['single_image']:
            is_optical = bool(random.randint(0,1))

            if is_optical:
                image = optical
            else:
                image = thermal

            # augmentation
            if self.config['augmentation']['photometric']['enable']:
                image = augmentation.photometric_augmentation(image, **self.config['augmentation']['photometric'])

            if self.config['augmentation']['homographic']['enable']:
                image, keypoints, valid_mask = augmentation.homographic_augmentation(image, keypoints, **self.config['augmentation']['homographic'])
            else:
                valid_mask = augmentation.dummy_valid_mask(image.shape)

            # add channel information to image and mask
            image = np.expand_dims(image, 0)
            valid_mask = np.expand_dims(valid_mask, 0)

            # add to output dict
            out['image'] = torch.from_numpy(image.astype(np.float32))
            out['valid_mask'] = torch.from_numpy(valid_mask.astype(np.bool))
            out['is_optical'] = torch.BoolTensor([is_optical])
            if keypoints is not None:
                keypoints = utils.generate_keypoint_map(keypoints, (h,w))
                out['keypoints'] = torch.from_numpy(keypoints.astype(np.bool))

        else:
            # initialize the images
            out['optical'] = {}
            out['thermal'] = {}

            optical_is_optical = True
            thermal_is_optical = False
            if self.config['random_pairs']:
                tmp_optical = optical
                tmp_thermal = thermal
                if bool(random.randint(0,1)):
                    optical = tmp_thermal
                    optical_is_optical = False
                if bool(random.randint(0,1)):
                    thermal = tmp_optical
                    thermal_is_optical = True

            # augmentation
            if self.config['augmentation']['photometric']['enable']:
                optical = augmentation.photometric_augmentation(optical, **self.config['augmentation']['photometric'])
                thermal = augmentation.photometric_augmentation(thermal, **self.config['augmentation']['photometric'])

            if self.config['augmentation']['homographic']['enable']:
                # randomly pick one image to warp
                if bool(random.randint(0,1)):

                    valid_mask_thermal = augmentation.dummy_valid_mask(thermal.shape)
                    if self.use_valid_mask:
                        # Custom valid mask
                        valid_mask_thermal = combined_mask
                        
                    keypoints_thermal = keypoints
                    optical, keypoints_optical, valid_mask_optical, H = augmentation.homographic_augmentation(optical,
                                                                                                              keypoints,
                                                                                                              return_homography = True,
                                                                                                              **self.config['augmentation']['homographic'])
                    out['optical']['homography'] = torch.from_numpy(H.astype(np.float32))
                    out['thermal']['homography'] = torch.eye(3, dtype=torch.float32)
                else:
                    valid_mask_optical = augmentation.dummy_valid_mask(optical.shape)
                    if self.use_valid_mask:
                        # Custom valid mask
                        valid_mask_optical = combined_mask
                    keypoints_optical = keypoints
                    thermal, keypoints_thermal, valid_mask_thermal, H = augmentation.homographic_augmentation(thermal,
                                                                                                              keypoints,
                                                                                                              return_homography = True,
                                                                                                              **self.config['augmentation']['homographic'])
                    out['thermal']['homography'] = torch.from_numpy(H.astype(np.float32))
                    out['optical']['homography'] = torch.eye(3, dtype=torch.float32)
            else:
                keypoints_optical = keypoints
                keypoints_thermal = keypoints

                valid_mask_optical = valid_mask_thermal = augmentation.dummy_valid_mask(optical.shape)
                if self.use_valid_mask:
                    # Custom valid mask
                    valid_mask_optical = combined_mask
                    valid_mask_thermal = combined_mask

            # add channel information to image and mask
            optical = np.expand_dims(optical, 0)
            thermal = np.expand_dims(thermal, 0)
            valid_mask_optical = np.expand_dims(valid_mask_optical, 0)
            valid_mask_thermal = np.expand_dims(valid_mask_thermal, 0)

            out['optical']['image'] = torch.from_numpy(optical.astype(np.float32))
            out['optical']['valid_mask'] = torch.from_numpy(valid_mask_optical.astype(np.bool))
            out['optical']['is_optical'] = torch.BoolTensor([optical_is_optical])
            if keypoints_optical is not None:
                keypoints_optical = utils.generate_keypoint_map(keypoints_optical, (h,w))
                out['optical']['keypoints'] = torch.from_numpy(keypoints_optical.astype(np.bool))

            out['thermal']['image'] = torch.from_numpy(thermal.astype(np.float32))
            out['thermal']['valid_mask'] = torch.from_numpy(valid_mask_thermal.astype(np.bool))
            out['thermal']['is_optical'] = torch.BoolTensor([thermal_is_optical])
            if keypoints_optical is not None:
                keypoints_thermal = utils.generate_keypoint_map(keypoints_thermal, (h,w))
                out['thermal']['keypoints'] = torch.from_numpy(keypoints_thermal.astype(np.bool))

        if self.config['return_name']:
            out['name'] = self.memberslist[index]
            out['on_path'] = on_img_path
            out['off_path'] = off_img_path
            
        return out

    def get_name(self, index):
        return self.memberslist[index]

    def returns_pair(self):
        return not self.config['single_image']

    def __len__(self):
        return self.num_files
