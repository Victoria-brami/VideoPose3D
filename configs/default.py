#----------------------------------------------------------------------------
# Created By  : Victoria BRAMI   
# Created Date: 2022/07/time ..etc
# version ='1.0'
# ---------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

_C = CN()
_C.DEBUG = False


# DATASET
_C.DATASET = CN()
_C.DATASET.DATASET = 'dad'
_C.DATASET.KEYPOINTS = 'gt_train'
_C.DATASET.SUBJECTS_TRAIN = 'vp1,vp2,vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10'
_C.DATASET.SUBJECTS_TEST = 'vp11,vp12'
_C.DATASET.SUBJECTS_UNLABELED = ''
_C.DATASET.ACTIONS = ''


# MODEL
_C.MODEL = CN()
_C.MODEL.STRIDE = 1
_C.MODEL.LR = 0.001
_C.MODEL.LR_DECAY = 0.95
_C.MODEL.DROPOUT = 0.25
_C.MODEL.ARCHITECTURE = None
_C.MODEL.CAUSAL = True
_C.MODEL.AUGMENTATION = True
_C.MODEL.TEST_TIME_AUGMENTATION = True
_C.MODEL.CHANNELS = 1024
_C.MODEL.EPOCHS = 80
_C.MODEL.BATCH_SIZE = 1024

  

# VIS
_C.VIS = CN()
_C.VIS.RENDER = False
_C.VIS.SUBJECT = ''
_C.VIS.ACTION = ''
_C.VIS.CAMERA = 0
_C.VIS.VIDEO = ''
_C.VIS.SKIP = 0
_C.VIS.OUTPUT = ''
_C.VIS.EXPORT = ''
_C.VIS.BITRATE = ''
_C.VIS.NO_GT = False
_C.VIS.FRAME_LIMIT = 200
_C.VIS.SIZE = 6
_C.VIS.DOWNSAMPLE = 4

# TRAIN
_C.TRAIN = CN()
_C.TRAIN.IS_TRAIN = True
_C.TRAIN.RESUME = ''
_C.TRAIN.EVALUATE  = ''
_C.TRAIN.BY_SUBJECT = False

# EXPS
_C.EXPS = CN()
_C.EXPS.BONE_SYM = False
_C.EXPS.ILLEGAL_ANGLE = False
_C.EXPS.LAMBDA_SYM = 0.
_C.EXPS.LAMBDA_ANGLE = 0.
_C.EXPS.NO_PROJ = False
_C.EXPS.SUBSET = 1.
_C.EXPS.DOWNSAMPLE = 1
_C.EXPS.WARMUP = 10
_C.EXPS.LINEAR_PROJECTION = False
_C.EXPS.BONE_LENGTH = True
_C.EXPS.DISABLE_OPTIMIZATIONS = False
_C.EXPS.EVAL = True
_C.EXPS.DENSE = False

# LOGS 
_C.LOGS = CN()
_C.LOGS.TENSORBOARD = ''
_C.LOGS.SAVE_CHECKPOINT = True
_C.LOGS.CHECKPOINT = ''
_C.LOGS.CHECKPOINT_FREQUENCY = 80
_C.LOGS.SEQ_START = 2000
_C.LOGS.SEQ_LENGTH = 10000
_C.LOGS.PAD = 2000
_C.LOGS.EXPORT_TRAINING_CURVES = True



def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    # cfg.merge_from_list(args.opts)
    
    # CREATE CHECKPOINT FILE
    
    if cfg.LOGS.SAVE_CHECKPOINT:
        
        checkpoint_path = os.path.join("checkpoint")
        tb_path = os.path.join("logs")
        
        if cfg.DEBUG:
            cfg.LOGS.TENSORBOARD = os.path.join(tb_path, "debug")
            cfg.LOGS.CHECKPOINT = os.path.join(checkpoint_path, "debug")
        else:
            if cfg.MODEL.CAUSAL:
                checkpoint_path = os.path.join(checkpoint_path, "causal")
                tb_path = os.path.join(tb_path, "causal")
            
            if len(cfg.DATASET.SUBJECTS_UNLABELED) > 0:
                checkpoint_path = os.path.join(checkpoint_path, "semi_supervised")
                tb_path = os.path.join(tb_path, "semi_supervised")
            else:
                checkpoint_path = os.path.join(checkpoint_path, "fully_supervised")
                tb_path = os.path.join(tb_path, "fully_supervised")
            
            arc = 'x'.join([str(filt) for filt in cfg.MODEL.ARCHITECTURE])
            checkpoint_path = os.path.join(checkpoint_path, arc)
            tb_path = os.path.join(tb_path, arc)
            
            if cfg.EXPS.BONE_SYM and not cfg.EXPS.ILLEGAL_ANGLE:
                checkpoint_path += '_sym_{}'.format(cfg.EXPS.LAMBDA_SYM)
                tb_path += '_sym_{}'.format(cfg.EXPS.LAMBDA_SYM)
            elif cfg.EXPS.BONE_SYM and cfg.EXPS.ILLEGAL_ANGLE:
                checkpoint_path += '_sym_{}_angle_{}'.format(cfg.EXPS.LAMBDA_SYM, cfg.EXPS.LAMBDA_ANGLE)
                tb_path += '_sym_{}_angle_{}'.format(cfg.EXPS.LAMBDA_SYM, cfg.EXPS.LAMBDA_ANGLE)
            elif not cfg.EXPS.BONE_SYM and cfg.EXPS.ILLEGAL_ANGLE:
                checkpoint_path += '_sym_{}'.format(cfg.EXPS.LAMBDA_ANGLE)
                tb_path += '_sym_{}'.format(cfg.EXPS.LAMBDA_ANGLE)
            cfg.LOGS.TENSORBOARD = tb_path
            cfg.LOGS.CHECKPOINT = checkpoint_path
    cfg.freeze()