# cloneright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    
    visible = target.clone() 
    visible[visible != 0] = 1

    return torch.mean(torch.norm(predicted*visible - target, dim=len(target.shape)-1))
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    visible = target.clone() 
    visible[visible != 0] = 1
    return torch.mean(w * torch.norm(predicted*visible - target, dim=len(target.shape)-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    visible = target.clone() 
    visible[visible != 0] = 1
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted*visible, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted*visible - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned*visible - target, axis=len(target.shape)-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    visible = target.clone() 
    visible[visible != 0] = 1
    
    norm_predicted = torch.mean(torch.sum((predicted*visible)**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted*visible, target)

def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    visible = target.clone() 
    visible[visible != 0] = 1
    
    velocity_predicted = np.diff(predicted*visible, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))

def angle_error(predicted, debug=True):
    """_summary_

    Args:
        predicted (tensor):  (B, S, 17, 3) tensor with the 3d pose
    """
    # Extract specific joints positions
    nose = predicted[:, :, 0]
    leye, reye = predicted[:, :, 1], predicted[:, :, 2]
    lear, rear = predicted[:, :, 3], predicted[:, :, 4]
    lshoulder, rshoulder = predicted[:, :, 5], predicted[:, :, 6]
    lelbow, relbow = predicted[:, :, 7], predicted[:, :, 8]
    lwrist, rwrist = predicted[:, :, 9], predicted[:, :, 10]
    lhip, rhip = predicted[:, :, 11], predicted[:, :, 12]
    lknee, rknee = predicted[:, :, 13], predicted[:, :, 14]
    lfoot, rfoot = predicted[:, :, 15], predicted[:, :, 16]

    v_rs_ls = lshoulder - rshoulder
    v_rh_lh = rhip - lhip
    
    # Compute the angle between left lower arm and body
    v_le_ls = lelbow - lshoulder
    v_lw_le = lwrist - lelbow
    n_ls = torch.cross(v_rs_ls, v_le_ls)
    up_left_error = torch.sum(torch.mul(n_ls / torch.norm(n_ls), v_lw_le  / torch.norm(v_lw_le)), dim=-1)
    up_left_error = torch.max(up_left_error, torch.zeros(up_left_error.shape).cuda())
    
    # Compute the angle between right lower arm and body
    v_re_rs = relbow - rshoulder
    v_rw_re = rwrist - relbow
    n_rs = torch.cross(v_rs_ls, v_re_rs)
    up_right_error = torch.sum(torch.mul(n_rs / torch.norm(n_rs), v_rw_re  / torch.norm(v_rw_re)), dim=-1)
    up_right_error = torch.max(up_right_error, torch.zeros(up_right_error.shape).cuda())
    
    # Compute the angle between left lower leg and body
    v_lk_lh = lknee - lhip
    v_lf_lk = lfoot - lknee
    n_lh = torch.cross(v_rh_lh, v_lk_lh)
    low_left_error = torch.sum(torch.mul(n_lh / torch.norm(n_lh), v_lf_lk  / torch.norm(v_lf_lk)), dim=-1)
    low_left_error = torch.max(low_left_error, torch.zeros(low_left_error.shape).cuda())
    
    # Compute the angle between right lower leg and body
    v_rk_rh = rknee - rhip
    v_rf_rk = rfoot - rknee
    n_rh = torch.cross(v_rh_lh, v_rk_rh)
    low_right_error = torch.sum(torch.mul(n_rh / torch.norm(n_rh), v_rf_rk  / torch.norm(v_rf_rk)), dim=-1)
    low_right_error = torch.max(low_right_error, torch.zeros(low_right_error.shape).cuda())
    
    # Compute the angle between left left face and right face
    v_le_le = lear - leye
    v_le_n = nose - leye
    v_re_re = reye - rear
    v_n_re = nose - reye
    n_le = -torch.cross(v_le_le, v_le_n)
    n_re = torch.cross(v_n_re, v_re_re)
    head_error = torch.sum(torch.mul(n_le / torch.norm(n_le), n_re / torch.norm(n_re)), dim=-1)
    head_error = torch.max(head_error, torch.zeros(head_error.shape).cuda())
   
    if debug:
        print("/////"*5, " ILLEGAL ANGLE ", "/////"*5)
        print()
        print("   Head: ", head_error.shape, torch.mean(head_error).item(),torch.mean(head_error*torch.exp(head_error)).item())
        print("   Up Left: ",  torch.mean(up_left_error).item(), torch.mean(up_left_error*torch.exp(up_left_error)).item())
        print("   Up Right: ",  torch.mean(up_right_error).item(), torch.mean(up_right_error*torch.exp(up_right_error)).item())
        print("   Low Left: ",  torch.mean(low_left_error).item(), torch.mean(low_left_error*torch.exp(low_left_error)).item())
        print("   Low Right: ",  torch.mean(low_right_error).item(), torch.mean(low_right_error*torch.exp(low_right_error)).item())
        print("   MEAN ERROR: ", torch.mean(up_left_error + up_right_error + low_left_error + low_right_error + head_error).item(), torch.mean((up_left_error + up_right_error + low_left_error + low_right_error + head_error)*torch.exp(up_left_error + up_right_error + low_left_error + low_right_error + head_error)).item())
        print()
        print("/////"*15)
    return up_left_error, up_right_error, low_left_error, low_right_error, head_error
    
    