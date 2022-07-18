from dataclasses import replace
import json
import numpy as np
from .dataset import COCOTypeDataset
import os
import pandas as pd
import cv2
from copy import deepcopy
import sys


class DadDataset(COCOTypeDataset):

    def __init__(self, datapath, point_of_view) -> None:
        super().__init__(datapath)
        self.counter = 0
        self.image_counter = 0
        self.people_counter = 0
        self.point_of_view = os.path.join(self.root_dir, point_of_view)
        self.anns_path = os.path.join(self.root_dir, 'annots', 'openpose_3d')
        self.train_ids = ('vp1', 'vp2', 'vp3', 'vp4', 'vp5', 'vp6', 'vp7', 'vp8', 'vp9', 'vp10', 'vp11', 'vp12')
        self.val_ids = ('vp13', 'vp14', 'vp15')

    def _load_calibration_data(self, video_path, only_extrinsic=False):
        calidration_data = json.load(open(deepcopy(video_path).replace(self.anns_path, self.point_of_view).replace('.openpose.3d.csv', '.calibration.json'), 'r'))
        intrinsic_data = calidration_data['intrinsics']
        fx = intrinsic_data["focallength"]["fx"]
        fy = intrinsic_data["focallength"]["fy"]
        cx = intrinsic_data["principal_point"]["cx"]
        cy = intrinsic_data["principal_point"]["cy"]
        k1 = intrinsic_data["distortion"]["k1"]
        k2 = intrinsic_data["distortion"]["k2"]
        k3 = intrinsic_data["distortion"]["k3"]
        p1 = intrinsic_data["distortion"]["p1"]
        p2 = intrinsic_data["distortion"]["p2"]

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist_coeffs = np.array([k1, k2, p1, p2, k3])

        extrinsic_data = calidration_data['extrinsics'] # No rotation neither tranlation
        # Rotation
        theta = extrinsic_data['rotation']["w"]
        ux = extrinsic_data['rotation']["x"]
        uy = extrinsic_data['rotation']["y"]
        uz = extrinsic_data['rotation']["z"]
        u = np.array([ux, uy, uz])
        Rot = np.cos(theta) * np.eye(3)  + np.sin(theta)*np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]]) + (1-np.cos(theta)) * np.dot(np.expand_dims(u, axis=1), np.expand_dims(u, axis=0))
        # Translation
        tx = extrinsic_data['translation']["x"]
        ty = extrinsic_data['translation']["y"]
        tz = extrinsic_data['translation']["z"]        
        R = np.zeros((4, 4))
        t = np.array([tx, ty, tz])
        R[:3, :3] = Rot
        R[:3, 3] = -np.dot(Rot, t)
        R[3, 3] = 1

        width = intrinsic_data["img_size"]["width"]
        height = intrinsic_data["img_size"]["height"]
        
        if only_extrinsic:
            R = [theta, ux, uy, uz]
            t = [tx, ty, tz]
            return R, t
        
        return R, K, dist_coeffs, width, height
    
    def _load_keypoints_data(self, video_csv_path):
        new_path = os.path.join(self.anns_path, "openpose_3d")
        data = pd.read_csv(video_csv_path)
        return data
    
    def _compute_2d_keypoints(self, data, R, K, dist_coeffs):
        kpt = data[[
        'nose_x', 'nose_y', 'nose_z',
        'lEye_x', 'lEye_y', 'lEye_z', 
        'rEye_x', 'rEye_y', 'rEye_z', 
        'lEar_x', 'lEar_y', 'lEar_z',
        'rEar_x', 'rEar_y', 'rEar_z', 
        'lShoulder_x', 'lShoulder_y', 'lShoulder_z',  
        'rShoulder_x', 'rShoulder_y', 'rShoulder_z', 
        'lElbow_x', 'lElbow_y', 'lElbow_z', 
        'rElbow_x', 'rElbow_y', 'rElbow_z',
        'lWrist_x', 'lWrist_y', 'lWrist_z', 
        'rWrist_x', 'rWrist_y', 'rWrist_z', 
        'lHip_x', 'lHip_y', 'lHip_z', 
        'rHip_x', 'rHip_y', 'rHip_z', 
        'lKnee_x', 'lKnee_y', 'lKnee_z',  
        'rKnee_x', 'rKnee_y', 'rKnee_z',  
        'lAnkle_x', 'lAnkle_y', 'lAnkle_z',  
        'rAnkle_x', 'rAnkle_y', 'rAnkle_z'   
        ]].to_numpy()

        visibility = data[['nose_p', 
            'lEye_p', 'rEye_p', 
            'lEar_p', 'rEar_p', 
            'lShoulder_p', 'rShoulder_p', 
            'lElbow_p', 'rElbow_p', 
            'lWrist_p', 'rWrist_p', 
            'lHip_p', 'rHip_p', 
            'lKnee_p', 'rKnee_p', 
            'lAnkle_p','rAnkle_p']].to_numpy()

        kpt_3d = np.reshape(kpt, (len(kpt), self.num_joints, 3))
        kpt_4d = np.concatenate((kpt_3d, np.ones((len(kpt_3d), self.num_joints, 1))), axis=2) # Shape N x nb_joints x 4

        # Project into camera coordinates space
        if R is not None:
            kpt_3d_cam = np.dot(R, kpt_3d.transpose(0, 2, 1)).transpose(1, 2, 0)[:, :, :3] # Shape N x nb_joints x 3
        else:
            kpt_3d_cam = kpt_3d
        # Project into 2d
        kpt_2d = np.dot(K, kpt_3d_cam.transpose(0, 2, 1)).transpose(1, 2, 0)
        
        kpt_2d = np.divide(kpt_2d, np.expand_dims(kpt_3d_cam[:, :, 2], axis=2), out=np.zeros_like(kpt_2d), where=np.expand_dims(kpt_3d_cam[:, :, 2], axis=2)!=0)
        kpt_2d = kpt_2d[:, :, :2]

        return kpt_3d_cam, kpt_2d, visibility


    def _compute_bbox(self, kpt_2d, height, width):
      
        if np.count_nonzero(kpt_2d) > 0:
            maxs=np.max(kpt_2d[kpt_2d[:, 1]!=0], axis=0)
            mins=np.min(kpt_2d[kpt_2d[:, 1]!=0], axis=0)
        
            x_min = max(mins[0]-0.1*(maxs[0] - mins[0]), 0)
            y_min = max(mins[1]-(maxs[1] - mins[1])*0.2, 0)

            return [x_min, 
                    y_min, 
                    min((maxs[0] - mins[0])*1.2, (width-1) - x_min) , 
                    min((maxs[1] - mins[1])*1.2, (height-1) - y_min)] # x, y, w, h 
        else:
            return [-1, -1, -1, -1]


    def build_json_id(self, id):
        coco_data = {"images": [], "annotations": []}

        for fold in os.listdir(os.path.join(self.anns_path, id)):
            sys.stdout.write('\r Processing folder {} video {}....'.format(id, fold))
            
            video_data_path =  os.path.join(self.anns_path, id, fold)
            video_data = self._load_keypoints_data(video_data_path)
            R, K, dist_coeffs, width, height = self._load_calibration_data(video_data_path)
            #print(R)
            kpts_3d_cam, kpts_2d, visibility = self._compute_2d_keypoints(video_data, R=None, K=K, dist_coeffs=dist_coeffs)
            
            num_frames = len(kpts_3d_cam)
            
            for frame in range(len(kpts_3d_cam)):
                if np.count_nonzero(kpts_3d_cam[frame]) <= 1:
                    continue
                coco_img = {"height": height, "width": width, "id": self.counter}
                coco_img["file_name"] = os.path.join(os.path.basename(self.point_of_view), id, os.path.basename(video_data_path).replace(".openpose.3d.csv", "") + "_frame_" + str(video_data["frame_id"][frame]) + ".jpg")
                coco_data["images"].append(coco_img)

                coco_anns = {
                    "id": self.counter, 
                    "image_id": self.image_counter, # can have several people on the same frame
                    "segmentation": [], 
                    "keypoints": [], 
                    "keypoints_3d": [], 
                    "num_keypoints": 0, 
                    "category_id": 1,
                    "area": 0, 
                    "iscrowd": 0, 
                    "bbox": []}
                self.counter += 1
                self.image_counter += 1
                coco_anns["bbox"] = self._compute_bbox(kpts_2d[frame], coco_img["height"], coco_img["width"])
                coco_anns["area"] = coco_anns["bbox"][2] * coco_anns["bbox"][3]
                
                coco_anns["keypoints"] = list(np.reshape(np.concatenate((kpts_2d[frame], np.reshape(visibility[frame], (self.num_joints, 1))), axis=1), (self.num_joints*3,)))
                coco_anns["keypoints_3d"] = list(np.reshape(kpts_3d_cam[frame], (self.num_joints*3,)))
                coco_anns["num_keypoints"] = np.count_nonzero(visibility[frame])

                coco_data["annotations"].append(coco_anns)
        
        with open(os.path.join(self.root_dir, "{}_coco_format.json".format(id)), "w") as json_file:
            json.dump(coco_data, json_file, indent=1)

    def build_coco_val_json(self):
        coco_data = {"images": [], "annotations": [], "categories": []}
        val_jsonfile = {"images":[], "annotations": []}
        
        val_jsonfile["categories"] = [{
            "supercategory": "person",
            "id": 1, 
            "name": "person",
            "keypoints": ["nose", "left_eye", "right_eye", 
                            "left_ear", "right_ear", 
                            "left_shoulder", "right_shoulder", 
                            "left_elbow", "right_elbow",
                            "left_wrist", "right_wrist",
                            "left_hip", "right_hip",
                            "left_knee", "right_knee",
                            "left_ankle", "right_ankle"
            ],
            "skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],
                        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]}
            ]

        for id in self.val_ids:
            json_id_filename = json.load(open(os.path.join(self.root_dir, '{}_coco_format.json'.format(id)), 'r'))
            val_jsonfile["images"] += json_id_filename["images"]
            val_jsonfile["annotations"] += json_id_filename["annotations"]

        with open(os.path.join(self.root_dir, 'person_keypoints_val_coco_format.json'), 'w') as json_f:
            json.dump(val_jsonfile, json_f)
        


    def build_coco_train_json(self):
        coco_data = {"images": [], "annotations": [], "categories": []}
        train_jsonfile = {"images":[], "annotations": []}
        train_jsonfile["categories"] = [{
            "supercategory": "person",
            "id": 1, 
            "name": "person",
            "keypoints": ["nose", "left_eye", "right_eye", 
                            "left_ear", "right_ear", 
                            "left_shoulder", "right_shoulder", 
                            "left_elbow", "right_elbow",
                            "left_wrist", "right_wrist",
                            "left_hip", "right_hip",
                            "left_knee", "right_knee",
                            "left_ankle", "right_ankle"
            ],
            "skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],
                        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]}
            ]
 
        for id in self.train_ids:
            json_id_filename = json.load(open(os.path.join(self.root_dir, '{}_coco_format.json'.format(id)), 'r'))
            train_jsonfile["images"] += json_id_filename["images"]
            train_jsonfile["annotations"] += json_id_filename["annotations"]
           
        with open(os.path.join(self.root_dir, 'person_keypoints_train_coco_format.json'), 'w') as json_f:
            json.dump(train_jsonfile, json_f)  

    def build_coco_small_train_json(self, samples_per_fold=500):
        coco_data = {"images": [], "annotations": [], "categories": []}
        train_jsonfile = {"images":[], "annotations": []}
        train_jsonfile["categories"] = [{
            "supercategory": "person",
            "id": 1, 
            "name": "person",
            "keypoints": ["nose", "left_eye", "right_eye", 
                            "left_ear", "right_ear", 
                            "left_shoulder", "right_shoulder", 
                            "left_elbow", "right_elbow",
                            "left_wrist", "right_wrist",
                            "left_hip", "right_hip",
                            "left_knee", "right_knee",
                            "left_ankle", "right_ankle"
            ],
            "skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],
                        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]}
            ]
        new_img_id = 0
        for id in self.train_ids:
            json_id_filename = json.load(open(os.path.join(self.root_dir, '{}_coco_format.json'.format(id)), 'r'))
            print("len images", len(json_id_filename["images"]))
            idxs = np.arange(len(json_id_filename["images"]))
            np.random.shuffle(idxs)
            idxs = idxs[:samples_per_fold]
            for frame_idx in idxs:
                image = json_id_filename["images"][frame_idx]
                image["id"] = new_img_id
                ann = json_id_filename["annotations"][frame_idx]
                ann["image_id"] = new_img_id
                train_jsonfile["images"].append(image)
                train_jsonfile["annotations"].append(ann)
                new_img_id += 1
           
        with open(os.path.join(self.root_dir, 'person_keypoints_small_train_{}_coco_format.json'.format(samples_per_fold)), 'w') as json_f:
            json.dump(train_jsonfile, json_f)
            
    def select_data(self, image_set, index_list=[]):
        if len(index_list)==0:
            index_list = [489, 650, 2041, 2213, 2323, 2336, 2522, 2958, 3235, 3835, 6321, 
                          6627,6846,10084,12166,12202,12634,12897,13115,13621,13856,13946,14185,15087,15337,
                          15420,15442,15587,15681,15689,16185,16317,16605,16895,17041,17230,17664,17879]
        data = json.load(open(os.path.join(self.root_dir, 'person_keypoints_{}_coco_format.json'.format(image_set)), 'r'))
        print('Length of images: ', len(data["images"]))
        
        new_data = {"images":[], "annotations": []}
        new_data["categories"] = data["categories"]
        new_img_id = 0
        for frame_idx in range(len(data["images"])):
            if frame_idx not in index_list:
                image = data["images"][frame_idx]
                image["id"] = new_img_id
                ann = data["annotations"][frame_idx]
                ann["image_id"] = new_img_id
                new_data["images"].append(image)
                new_data["annotations"].append(ann)
                new_img_id += 1
        with open(os.path.join(self.root_dir, 'person_keypoints_filt_{}_coco_format.json'.format(image_set)), 'w') as file:
            json.dump(new_data, file)
        
    def build_coco_small_val_json(self, samples_per_fold=500):
        coco_data = {"images": [], "annotations": [], "categories": []}
        val_jsonfile = {"images":[], "annotations": []}
        
        val_jsonfile["categories"] = [{
            "supercategory": "person",
            "id": 1, 
            "name": "person",
            "keypoints": ["nose", "left_eye", "right_eye", 
                            "left_ear", "right_ear", 
                            "left_shoulder", "right_shoulder", 
                            "left_elbow", "right_elbow",
                            "left_wrist", "right_wrist",
                            "left_hip", "right_hip",
                            "left_knee", "right_knee",
                            "left_ankle", "right_ankle"
            ],
            "skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],
                        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]}
            ]
        new_img_id = 0
        for id in self.val_ids:
            json_id_filename = json.load(open(os.path.join(self.root_dir, '{}_coco_format.json'.format(id)), 'r'))
            idxs = np.arange(len(json_id_filename["images"]))
            np.random.shuffle(idxs)
            idxs = idxs[:samples_per_fold]
            for frame_idx in idxs:
                image = json_id_filename["images"][frame_idx]
                image["id"] = new_img_id
                ann = json_id_filename["annotations"][frame_idx]
                ann["image_id"] = new_img_id
                val_jsonfile["images"].append(image)
                val_jsonfile["annotations"].append(ann)
                new_img_id += 1

        with open(os.path.join(self.root_dir, 'person_keypoints_small_val_{}_coco_format.json'.format(samples_per_fold)), 'w') as json_f:
            json.dump(val_jsonfile, json_f)    

    def build_coco_json(self):
        coco_data = {"images": [], "annotations": [], "categories": []}
        train_jsonfile = {"images":[], "annotations": []}
        val_jsonfile = {"images":[], "annotations": []}
        train_jsonfile["categories"] = [{
            "supercategory": "person",
            "id": 1, 
            "name": "person",
            "keypoints": ["nose", "left_eye", "right_eye", 
                            "left_ear", "right_ear", 
                            "left_shoulder", "right_shoulder", 
                            "left_elbow", "right_elbow",
                            "left_wrist", "right_wrist",
                            "left_hip", "right_hip",
                            "left_knee", "right_knee",
                            "left_ankle", "right_ankle"
            ],
            "skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],
                        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]}
            ]
        val_jsonfile["categories"] = [{
            "supercategory": "person",
            "id": 1, 
            "name": "person",
            "keypoints": ["nose", "left_eye", "right_eye", 
                            "left_ear", "right_ear", 
                            "left_shoulder", "right_shoulder", 
                            "left_elbow", "right_elbow",
                            "left_wrist", "right_wrist",
                            "left_hip", "right_hip",
                            "left_knee", "right_knee",
                            "left_ankle", "right_ankle"
            ],
            "skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],
                        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]}
            ]
        for id in self.train_ids:
            json_id_filename = json.load(open(osp.join(self.root_dir, '{}_coco_format.json'.format(id)), 'r'))
            train_jsonfile["images"] += json_id_filename["images"]
            train_jsonfile["annotations"] += json_id_filename["annotations"]
   
        for id in self.valid_ids:
            json_id_filename = json.load(open(osp.join(self.root_dir, '{}_coco_format.json'.format(id)), 'r'))
            val_jsonfile["images"] += json_id_filename["images"]
            val_jsonfile["annotations"] += json_id_filename["annotations"]
     
        
        with open(osp.join(self.root_dir, 'person_keypoints_train_coco_format.json'), 'w') as json_f:
            json.dump(train_jsonfile, json_f)  
        with open(osp.join(self.root_dir, 'person_keypoints_val_coco_format.json'), 'w') as json_f:
            json.dump(val_jsonfile, json_f)
        