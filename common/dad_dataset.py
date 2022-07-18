#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*- Line 2
#----------------------------------------------------------------------------
# Created By  : Victoria BRAMI   Line 3
# Created Date: 2022/07/time ..etc
# version ='1.0'
# ---------------------------------------------------------------------------
''' Details about the module and for what purpose it was built for'''  #Line 4
# ---------------------------------------------------------------------------
# Imports Line 5

import numpy as np
import argparse
import os
import copy
from .lib.dad_to_coco import DadDataset
from .lib.dataset import H36mTypeDataset
from common.skeleton import Skeleton
# ---------------------------------------------------------------------------

def normalize_screen_coordinates(X, w, h): 
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - [1, h/w]

    
def image_coordinates(X, w, h):
    assert X.shape[-1] == 2
    
    # Reverse camera frame normalization
    return (X + [1, h/w])*w/2


datapath = "/datasets_local/DriveAndAct"
dad_data = DadDataset(datapath=datapath, point_of_view='inner_mirror')
    
EXTRINSIC_PARAMS = {
        "vp1": [
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp1/run1b_2018-05-29-14-02-47.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp1/run1b_2018-05-29-14-02-47.ids_1.openpose.3d.csv"), True)[1]
            }, 
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp1/run2_2018-05-29-14-33-44.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp1/run2_2018-05-29-14-33-44.ids_1.openpose.3d.csv"), True)[1]
            }
            ],
        "vp2":[
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp2/run1_2018-05-03-14-08-31.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp2/run1_2018-05-03-14-08-31.ids_1.openpose.3d.csv"), True)[1]
            }, 
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp2/run2_2018-05-24-17-22-26.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp2/run2_2018-05-24-17-22-26.ids_1.openpose.3d.csv"), True)[1]
            }
            ],
        "vp3":[
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp3/run1b_2018-05-08-08-46-01.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp3/run1b_2018-05-08-08-46-01.ids_1.openpose.3d.csv"), True)[1]
            }, 
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp3/run2_2018-05-29-16-03-37.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp3/run2_2018-05-29-16-03-37.ids_1.openpose.3d.csv"), True)[1]
            }
            ],
        "vp4":[
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp4/run1_2018-05-22-13-28-51.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp4/run1_2018-05-22-13-28-51.ids_1.openpose.3d.csv"), True)[1]
            }, 
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp4/run2_2018-05-22-14-25-04.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp4/run2_2018-05-22-14-25-04.ids_1.openpose.3d.csv"), True)[1]
            }
            ],
        "vp5":[
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp5/run1_2018-05-22-15-10-41.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp5/run1_2018-05-22-15-10-41.ids_1.openpose.3d.csv"), True)[1]
            }, 
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp5/run2b_2018-05-22-15-50-07.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp5/run2b_2018-05-22-15-50-07.ids_1.openpose.3d.csv"), True)[1]
            }
            ],
        "vp6":[
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp6/run1_2018-05-23-10-21-45.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp6/run1_2018-05-23-10-21-45.ids_1.openpose.3d.csv"), True)[1]
            }, 
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp6/run2_2018-05-23-11-05-00.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp6/run2_2018-05-23-11-05-00.ids_1.openpose.3d.csv"), True)[1]
            }
            ],
        "vp7":[
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp7/run1_2018-05-23-13-16-52.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp7/run1_2018-05-23-13-16-52.ids_1.openpose.3d.csv"), True)[1]
            }, 
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp7/run2b_2018-05-23-13-54-07.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp7/run2b_2018-05-23-13-54-07.ids_1.openpose.3d.csv"), True)[1]
            }
            ],
        "vp8":[
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp8/run1d_2018-05-23-14-54-38.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp8/run1d_2018-05-23-14-54-38.ids_1.openpose.3d.csv"), True)[1]
            }, 
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp8/run2_2018-05-23-15-30-27.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp8/run2_2018-05-23-15-30-27.ids_1.openpose.3d.csv"), True)[1]
            }
            ],
        "vp9":[
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp9/run1b_2018-05-23-16-19-17.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp9/run1b_2018-05-23-16-19-17.ids_1.openpose.3d.csv"), True)[1]
            }
            ],
        "vp10":[
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp10/run1_2018-05-24-13-14-41.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp10/run1_2018-05-24-13-14-41.ids_1.openpose.3d.csv"), True)[1]
            }, 
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp10/run2_2018-05-24-14-08-46.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp10/run2_2018-05-24-14-08-46.ids_1.openpose.3d.csv"), True)[1]
            }
            ],
        "vp11":[
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp11/run1_2018-05-24-13-44-01.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp11/run1_2018-05-24-13-44-01.ids_1.openpose.3d.csv"), True)[1]
            }, 
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp11/run2_2018-05-24-14-35-56.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp11/run2_2018-05-24-14-35-56.ids_1.openpose.3d.csv"), True)[1]
            }
            ],
        "vp12":[
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp12/run1_2018-05-24-15-44-28.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp12/run1_2018-05-24-15-44-28.ids_1.openpose.3d.csv"), True)[1]
            }, 
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp12/run2_2018-05-24-16-21-35.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp12/run2_2018-05-24-16-21-35.ids_1.openpose.3d.csv"), True)[1]
            }
            ],
        "vp13":[
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp13/run1_2018-05-29-15-21-10.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp13/run1_2018-05-29-15-21-10.ids_1.openpose.3d.csv"), True)[1]
            }, 
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp13/run2_2018-05-30-11-34-54.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp13/run2_2018-05-30-11-34-54.ids_1.openpose.3d.csv"), True)[1]
            }
            ],
        "vp14":[
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp14/run1_2018-05-30-10-11-09.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp14/run1_2018-05-30-10-11-09.ids_1.openpose.3d.csv"), True)[1]
            }, 
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp14/run2_2018-05-30-10-42-33.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp14/run2_2018-05-30-10-42-33.ids_1.openpose.3d.csv"), True)[1]
            }
            ],
        "vp15":[
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp15/run1_2018-05-30-13-05-35.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp15/run1_2018-05-30-13-05-35.ids_1.openpose.3d.csv"), True)[1]
            }, 
            {
                'orientation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp15/run2_2018-05-30-13-34-33.ids_1.openpose.3d.csv"), True)[0], 
                'translation': dad_data._load_calibration_data(os.path.join(datapath, "inner_mirror/vp15/run2_2018-05-30-13-34-33.ids_1.openpose.3d.csv"), True)[1]
            }
            ],   
    }

INTRINSIC_PARAMS = [
    {
        'id': '',
        'center': [640., 512.],
        'focal_length': [567., 567.],
        'radial_distortion': [-0.2661,  0.0549, 0],
        'tangential_distortion': [0, 0],
        'res_w': 1280,
        'res_h': 1024,
        'azimuth': 70, # Only used for visualization
    },
]
    
    
SKELETON = Skeleton(parents=[-1, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14], 
                    joints_left=[1, 3, 5, 7, 9, 11, 13, 15], 
                    joints_right=[2, 4, 6, 8, 10, 12, 14, 16])
    
    
    
class DadHuman36MDataset(H36mTypeDataset):
    
    def __init__(self, datapath, remove_static_joints=False) -> None:
        super().__init__(fps=30, skeleton=SKELETON)
        self._cameras = copy.deepcopy(EXTRINSIC_PARAMS)

        for cameras in self._cameras.values():
            for i, cam in enumerate(cameras):
                cam.update(INTRINSIC_PARAMS[0])
                cam['id'] = i
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')
                
                # Normalize camera frame
                cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype('float32')
                cam['focal_length'] = cam['focal_length']/cam['res_w']*2
                if 'translation' in cam:
                    cam['translation'] = cam['translation']
                
                # Add intrinsic parameters vector
                cam['intrinsic'] = np.concatenate((cam['focal_length'],
                                                   cam['center'],
                                                   cam['radial_distortion'],
                                                   cam['tangential_distortion']))
                    
        
        # Load serialized dataset
        data = np.load(datapath, allow_pickle=True)['positions_3d'].item()
        
        self._data = {}
        for subject, records in data.items():
            self._data[subject] = {}
            for i, (video_name, positions) in enumerate(records.items()):
                self._data[subject][video_name] = {
                    'positions': [positions],
                    'cameras': [self._cameras[subject][i]],
                }
                
            
            
            
    def supports_semi_supervised(self):
        return True
    



if __name__ == '__main__':
    print()
    data = DadHuman36MDataset(datapath="/datasets_local/DriveAndAct/data_3d_dad_test.npz")