# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import matplotlib
import torch
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess as sp
from .loss import *

SUB_FIG_SIZE = 1

matplotlib.rcParams['xtick.labelsize']  = 13
matplotlib.rcParams['ytick.labelsize']  = 13
matplotlib.rcParams['legend.fontsize'] = 14

colors_list = ['red', 'darkviolet', 'darkblue', 'deeppink', 'darkred']

def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
            return int(w), int(h)
            
def get_fps(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            a, b = line.decode().strip().split('/')
            return int(a) / int(b)

def read_video(filename, skip=0, limit=-1):
    w, h = get_resolution(filename)
    
    command = ['ffmpeg',
            '-i', filename,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vsync', '0',
            '-vcodec', 'rawvideo', '-']
    
    i = 0
    with sp.Popen(command, stdout = sp.PIPE, bufsize=-1) as pipe:
        while True:
            data = pipe.stdout.read(w*h*3)
            if not data:
                break
            i += 1
            if i > limit and limit != -1:
                continue
            if i > skip:
                yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))
            
                
                
    
def downsample_tensor(X, factor):
    length = X.shape[0]//factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)


def plot_2d_skeleton(frame_id, keypoints, all_frames, parents, ax):
    image = ax.imshow(all_frames[frame_id], aspect='equal')
    h, w, _ = all_frames[frame_id].shape
            
    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue 
        if (keypoints[frame_id, j, 0] > 0 and keypoints[frame_id, j, 1] > 0) \
            and (keypoints[frame_id, j_parent, 0] > 0 and keypoints[frame_id, j_parent, 1] > 0) \
            and (keypoints[frame_id, j, 0] < h and keypoints[frame_id, j, 1] < w) \
            and (keypoints[frame_id, j_parent, 0] < h and keypoints[frame_id, j_parent, 1] < w):
            # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
            ax.plot([keypoints[frame_id, j, 0], keypoints[frame_id, j_parent, 0]],
                    [keypoints[frame_id, j, 1], keypoints[frame_id, j_parent, 1]], color='pink')
            
        if (keypoints[frame_id, 11, 0] != 0 and keypoints[frame_id, 12, 0] != 0) \
            and (keypoints[frame_id, 11, 1] != 0 and keypoints[frame_id, 12, 1] != 0) :   
            ax.plot([keypoints[frame_id, 11, 0], keypoints[frame_id, 12, 0]],
                    [keypoints[frame_id, 11, 1], keypoints[frame_id, 12, 1]], color="cornflowerblue")
        if (keypoints[frame_id, 5, 0] != 0 and keypoints[frame_id, 6, 0] != 0) \
            and (keypoints[frame_id, 5, 1] != 0 and keypoints[frame_id, 6, 1] != 0) :   
            ax.plot([keypoints[frame_id, 5, 0], keypoints[frame_id, 6, 0]],
                    [keypoints[frame_id, 5, 1], keypoints[frame_id, 6, 1]], color="cornflowerblue")
            
    plt.savefig('/root/no_backup/tmp2d'+ str(frame_id) + '.jpg')

def plot_3d_skeleton(frame_id, keypoints, poses, parents, skeleton, ax): 
    pos = poses[frame_id]
    
    for j, j_parent in enumerate(parents): 
        col = 'red' if j in skeleton.joints_right() else 'black' 
        if j_parent == -1:
            continue 
        if (keypoints[frame_id, j, 0] > 0 and keypoints[frame_id, j, 1] > 0) \
        and (keypoints[frame_id, j_parent, 0] > 0 and keypoints[frame_id, j_parent, 1] > 0):
            ax.plot(pos[[j, j_parent], 0], -pos[[j, j_parent], 1],
                            pos[[j, j_parent], 2], c=col, zdir='y')  
        if (keypoints[frame_id, 11, 0] != 0 and keypoints[frame_id, 12, 0] != 0) \
            and (keypoints[frame_id, 11, 1] != 0 and keypoints[frame_id, 12, 1] != 0) :   
            ax.plot(pos[[11, 12], 0], -pos[[11, 12], 1],
                            pos[[11, 12], 2], c='blue', zdir='y')  
        if (keypoints[frame_id, 5, 0] != 0 and keypoints[frame_id, 6, 0] != 0) \
            and (keypoints[frame_id, 5, 1] != 0 and keypoints[frame_id, 6, 1] != 0) :   
            ax.plot(pos[[5, 6], 0], -pos[[5, 6], 1],
                            pos[[5, 6], 2], c='blue', zdir='y')        
        
             
    plt.savefig('/root/no_backup/tmp3d'+ str(frame_id) + '.jpg')     
    

def render_opencv_animation(keypoints, keypoints_metadata, poses, skeleton, start_frame, end_frame, azim, output, viewport, fps, 
                            downsample=1, size=6, input_video_path=None, elev=10.):
    import cv2
    import os
    
    whole_poses = list(poses.values())
    
    poses_pred = whole_poses[0]
    poses_gt = whole_poses[1]
    
    poses_pred = poses_pred[:end_frame-start_frame]
    poses_gt = poses_gt[:end_frame-start_frame]
    m2mm = 1000
    new_poses_pred = np.expand_dims(poses_pred, axis=0)
    new_poses_gt = np.expand_dims(poses_gt, axis=0)
    
    # Compute errors between predictions and Ground Truth
    with torch.no_grad():
        mpjpe_out = mpjpe_eval(poses_pred, poses_gt) * m2mm
        n_mpjpe_out = n_mpjpe_eval(poses_pred, poses_gt) * m2mm
        p_mpjpe_out = p_mpjpe(poses_pred, poses_gt, mode='visu') * m2mm
        
        veloc_out =  mean_velocity_error(poses_pred, poses_gt, mode='visu') * m2mm
    
    print("Different Metric shapes: MPJPE {} P-MPJPE {} N MPJPE {} Velocity {}".format(mpjpe_out.shape, p_mpjpe_out.shape, n_mpjpe_out.shape, veloc_out))
    
    # Compute 2D reprojection error
    
    # Get the video frames
    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, skip=start_frame, limit=end_frame):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]
        
    if fps is None:
        fps = get_fps(input_video_path)
        
    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
        fps /= downsample
    
    # Define the figures
    anim_output = poses
    
    videoWriter = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
        (size * 1 * len(anim_output) * SUB_FIG_SIZE,
         size * 3 * SUB_FIG_SIZE)) # 100 is the width of a subfigure
    
    radius = 1.7
    parents = skeleton.parents()
    joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
    colors_2d = np.full(keypoints.shape[1], 'black')
    colors_2d[joints_right_2d] = 'red'
    
    for frame_id in range(len(all_frames)):
        fig = plt.figure(figsize=(SUB_FIG_SIZE * 1 * len(anim_output),
                                    SUB_FIG_SIZE * 3))
        plt.subplots_adjust(left=None,
                            bottom=None,
                            right=None,
                            top=None,
                            wspace=None,
                            hspace=0.5)
        ax_in = fig.add_subplot(3, 1 + len(poses), 1)
        ax_in.get_xaxis().set_visible(False)
        ax_in.get_yaxis().set_visible(False)
        ax_in.set_axis_off()
        ax_in.set_title('Input')
        
        plot_2d_skeleton(frame_id, keypoints, all_frames, parents, ax_in)
        points = ax_in.scatter(*keypoints[frame_id].T, 10, color=colors_2d, edgecolors='white', zorder=10)
        
        
        for index, (title, data) in enumerate(poses.items()):
            
            ax = fig.add_subplot(3, 1 + len(poses), index+2, projection='3d')
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim3d([-radius/2, radius/2])
            ax.set_ylim3d([0, radius])
            ax.set_ylim3d([radius/2, radius/2*3])
            ax.set_zlim3d([-radius/2, radius/2]) # exchange limits for y and z
            try:
                ax.set_aspect('equal')
            except NotImplementedError:
                ax.set_aspect('auto')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.dist = 7.5
            ax.set_title(title) #, pad=35
            
            plot_3d_skeleton(frame_id, keypoints, data, parents, skeleton, ax)
            
            ax_vel = fig.add_subplot(3, 1, 2)
            ax_vel.set_title('Velocity Error Visualize', fontsize=4*SUB_FIG_SIZE)
            ax_vel.plot(veloc_out[:frame_id],
                    color=(202 / 255, 0 / 255, 32 / 255),
                    label='velocity Error')
            ax_vel.legend()
            ax_vel.grid(True)
            ax_vel.set_xlabel('Frame', fontsize=3*SUB_FIG_SIZE)
            ax_vel.set_ylabel('Mean Velocity Error (mm/s)', fontsize=3*SUB_FIG_SIZE)
            ax_vel.set_xlim((0, len(veloc_out)))
            ax_vel.set_ylim((0, np.max((np.max(veloc_out), np.max(veloc_out)))))
            
            ax_mpjpe = fig.add_subplot(3, 1, 3)
            ax_mpjpe.set_title('MPJPE Visualize', fontsize=4*SUB_FIG_SIZE)
            ax_mpjpe.plot(mpjpe_out[:frame_id],
                      color=(202 / 255, 0 / 255, 32 / 255),
                      label='MPJPE (Protocol #1)')
            ax_mpjpe.plot(p_mpjpe_out[:frame_id],
                    color=(117/255,112/255,179/255),
                    label='P-MPJPE (Protocol #2)')
            ax_mpjpe.plot(n_mpjpe_out[:frame_id],
                      color='c',
                      label='Normalized MPJPE (Protocol #3)')
            
            ax_mpjpe.legend()
            ax_mpjpe.grid(True)
            ax_mpjpe.set_xlabel('Frame', fontsize=3*SUB_FIG_SIZE)
            ax_mpjpe.set_ylabel('Mean Position Error (mm)', fontsize=3*SUB_FIG_SIZE)
            ax_mpjpe.set_xlim((0, len(mpjpe_out)))
            ax_mpjpe.set_ylim((0, np.max((np.max(n_mpjpe_out), np.max(mpjpe_out)))))
            ax_mpjpe.tick_params(axis="x", labelsize=3*SUB_FIG_SIZE)
            ax_mpjpe.tick_params(axis="y", labelsize=3*SUB_FIG_SIZE)
            ax_mpjpe.legend(fontsize=3*SUB_FIG_SIZE)
            
            plt.savefig("/root/no_backup/tmp" + str(frame_id) + ".jpg")
            plt.show()
            
            canvas = FigureCanvasAgg(plt.gcf())
            canvas.draw()
            final_img = np.array(canvas.renderer.buffer_rgba())[:, :, [2, 1, 0]]
            print("final shape: ", final_img.shape)
            

            videoWriter.write(final_img)
            plt.close()

    videoWriter.release()
    print(f"Finish! The video is stored in "+ output)

            
            
            
def render_other_animation(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0, elev=10.):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    
    
    whole_poses = list(poses.values())
    
    poses_pred = whole_poses[0]
    poses_gt = whole_poses[1]
    
    print("\n /////////////// Estimated Keypoints in 3D: ///////// \n")
    print(poses_pred[10, :17, :], poses_gt[10, :17, :])
    print("\n HEAD \n")
    print(poses_pred[10, 17:, :], "\n GT", poses_gt[10, 17:, :])
    
    
    poses_pred = poses_pred[:limit-input_video_skip]
    poses_gt = poses_gt[:limit-input_video_skip]
    m2mm = 1000
    new_poses_pred = np.expand_dims(poses_pred, axis=0)
    new_poses_gt = np.expand_dims(poses_gt, axis=0)
    
    # Compute errors between predictions and Ground Truth
    with torch.no_grad():
        mpjpe_out = mpjpe_eval(poses_pred, poses_gt) * m2mm
        n_mpjpe_out = n_mpjpe_eval(poses_pred, poses_gt) * m2mm
        p_mpjpe_out = p_mpjpe(poses_pred, poses_gt, mode='visu') * m2mm
        
        veloc_out =  mean_velocity_error(poses_pred, poses_gt, mode='visu') * m2mm
    
    print("Different Metric shapes: MPJPE {} P-MPJPE {} N MPJPE {} Velocity {}".format(mpjpe_out.shape, p_mpjpe_out.shape, n_mpjpe_out.shape, veloc_out))
    
    plt.ioff()
    fig = plt.figure(figsize=(size*(1 + len(poses)), 3*size))
    ax_in = fig.add_subplot(3, 1 + len(poses), 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')
    
    SUB_FIG_SIZE = size
    
    ax_3d = []
    lines_3d = []
    trajectories = []
    lines_veloc = []
    lines_mpjpe = []
    
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(3, 1 + len(poses), index+2, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim3d([-radius/2, radius/2])
        ax.set_ylim3d([0, radius])
        ax.set_ylim3d([radius/2, radius/2*3])
        ax.set_zlim3d([-radius/2, radius/2]) # exchange limits for y and z
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title) #, pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    
    ax_vel = fig.add_subplot(3, 1, 2)
    ax_vel.set_title('Velocity Error Visualize', fontsize=3*SUB_FIG_SIZE)

    ax_vel.legend()
    ax_vel.grid(True)
    ax_vel.set_xlabel('Frame', fontsize=3*SUB_FIG_SIZE)
    ax_vel.set_ylabel('Mean Velocity Error (mm/s)', fontsize=3*SUB_FIG_SIZE)
    ax_vel.set_xlim((0, len(veloc_out)))
    ax_vel.set_ylim((0, 200))
    
    ax_mpjpe = fig.add_subplot(3, 1, 3)
    ax_mpjpe.set_title('MPJPE Visualize', fontsize=3*SUB_FIG_SIZE)
    
    ax_mpjpe.legend()
    ax_mpjpe.grid(True)
    ax_mpjpe.set_xlabel('Frame', fontsize=3*SUB_FIG_SIZE)
    ax_mpjpe.set_ylabel('Mean Position Error (mm)', fontsize=3*SUB_FIG_SIZE)
    ax_mpjpe.set_xlim((0, len(mpjpe_out)))
    ax_mpjpe.set_ylim((0, 200))
    ax_mpjpe.tick_params(axis="x", labelsize=3*SUB_FIG_SIZE)
    ax_mpjpe.tick_params(axis="y", labelsize=3*SUB_FIG_SIZE)
    ax_mpjpe.legend(fontsize=4*SUB_FIG_SIZE)
    

    
    poses = list(poses.values())

    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip, limit=limit):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]
        
        """
        keypoints = keypoints[input_video_skip:] # todo remove
        for idx in range(len(poses)):
            poses[idx] = poses[idx][input_video_skip:]
        """
        if fps is None:
            fps = get_fps(input_video_path)
    
    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
            trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None
    
    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))
    
    debug_data = []

    parents = skeleton.parents()
    def update_video(i):
        nonlocal initialized, image, lines, points
        print(" Frame {}: GT Neck {} Det Neck {}".format(i, poses[1][i][0], poses[0][i][0]))
        #for n, ax in enumerate(ax_3d):
        #    ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
        #    ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])

        # Update 2D poses
        joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
        colors_2d = np.full(keypoints.shape[1], 'black')
        colors_2d[joints_right_2d] = 'red'
        # print("Metrics shapes: ", mpjpe_out.shape, p_mpjpe_out.shape, n_mpjpe_out.shape, veloc_out.shape)
        if not initialized:
            image = ax_in.imshow(all_frames[i + input_video_skip], aspect='equal')
            h, w, _ = all_frames[i + input_video_skip].shape
            
            lines_mpjpe.append(ax_mpjpe.plot(np.arange(i+1), mpjpe_out[:i+1], color=(202 / 255, 0 / 255, 32 / 255),
                label='MPJPE (Protocol #1)'))
                
            lines_mpjpe.append(ax_mpjpe.plot(np.arange(i+1), np.mean(p_mpjpe_out[:i+1], axis=1),
            color=(117/255,112/255,179/255),
            label='P-MPJPE (Protocol #2)'))
            lines_mpjpe.append(ax_mpjpe.plot(np.arange(i+1), n_mpjpe_out[:i+1],
                        color='c',
                        label='Normalized MPJPE (Protocol #3)'))
            lines_veloc.append(ax_vel.plot(np.arange(i+1), np.mean(veloc_out[:i+1], axis=1),
            color=(202 / 255, 0 / 255, 32 / 255),
            label='velocity Error'))
            
            
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue 
                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    if (keypoints[i, j, 0] > 0 and keypoints[i, j, 1] > 0) and (keypoints[i, j_parent, 0] > 0 and keypoints[i, j_parent, 1] > 0) \
                    and (keypoints[i, j, 0] < h and keypoints[i, j, 1] < w) and (keypoints[i, j_parent, 0] < h and keypoints[i, j_parent, 1] < w):
                        # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                        lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                                [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))
                    else:
                        lines.append(ax_in.plot([0, 0],
                                                [0, 0], color='white'))
                col = 'red' if j in skeleton.joints_right() else 'black'
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    if (keypoints[i, j, 0] != 0 and keypoints[i, j, 1] != 0) and (keypoints[i, j_parent, 0] != 0 and keypoints[i, j_parent, 1] != 0):
                        lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                                [-pos[j, 1], -pos[j_parent, 1]],
                                                [pos[j, 2], pos[j_parent, 2]], zdir='y', c=col))
                        
                    else:
                         lines_3d[n].append(ax.plot([0, 0],
                                                [0, 0],
                                                [0, 0], zdir='y', color="white"))

            
            if keypoints[i, 11, 0] != 0 and keypoints[i, 12, 0] != 0 and keypoints[i, 11, 1] != 0 and keypoints[i, 12, 1] != 0 :   
                col = "blue"
                extra_parent = 12
                lines.append(ax_in.plot([keypoints[i, 11, 0], keypoints[i, 12, 0]],
                                                [keypoints[i, 11, 1], keypoints[i, 12, 1]], color=col))
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[11, 0], pos[extra_parent, 0]],
                                                [-pos[11, 1], -pos[extra_parent, 1]],
                                                [pos[11, 2], pos[extra_parent, 2]], zdir='y', color=col))
            else:
                col = "white"
                lines.append(ax_in.plot([0, 0], [0, 0], color=col))
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([0, 0],
                                            [0, 0],
                                            [0, 0], zdir='y', color=col))
            
            
            if keypoints[i, 5, 0] != 0 and keypoints[i, 6, 0] != 0 and keypoints[i, 5, 1] != 0 and keypoints[i, 6, 1] != 0 :
                col = "blue"
                lines.append(ax_in.plot([keypoints[i, 5, 0], keypoints[i, 6, 0]],
                                            [keypoints[i, 5, 1], keypoints[i, 6, 1]], color=col))
                extra_parent = 6
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[5, 0], pos[extra_parent, 0]],
                                            [-pos[5, 1], -pos[extra_parent, 1]],
                                            [pos[5, 2], pos[extra_parent, 2]], zdir='y', color=col))
            else:
                col = "white"
                lines.append(ax_in.plot([0, 0], [0, 0], color=col))
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([0, 0],
                                            [0, 0],
                                            [0, 0], zdir='y', color=col))
            
            
            points = ax_in.scatter(*keypoints[i].T, 10, color=colors_2d, edgecolors='white', zorder=10)
            initialized = True
            
        else:
            if mpjpe_out[i] > 200:
                fig = plt.figure(figsize=(16, 8))
                ax = fig.add_subplot(121)
                ax.imshow(all_frames[i])
                plt.title("MPJPE: {} mm".format(mpjpe_out[i]))
            
                ax2  = fig.add_subplot(122, projection="3d")
                ax2.view_init(elev=elev, azim=azim)
                ax2.set_xlim3d([-radius/2, radius/2])
                ax2.set_ylim3d([0, radius])
                ax2.set_ylim3d([radius/2, radius/2*3])
                ax2.set_zlim3d([-radius/2, radius/2]) # exchange limits for y and z
                for j, j_parent in enumerate(parents):
                    if j_parent == -1:
                        continue
                    col = "cornflowerblue"
                    pos = poses[0][i]
                    pos_gt = poses[1][i]
                    col_gt = "deeppink"
                    if (keypoints[i, j, 0] != 0 and keypoints[i, j, 1] != 0) and (keypoints[i, j_parent, 0] != 0 and keypoints[i, j_parent, 1] != 0):
                        ax.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]], c='deeppink')
                        ax2.plot([pos[j, 0], pos[j_parent, 0]],
                                                [-pos[j, 1], -pos[j_parent, 1]],
                                                [pos[j, 2], pos[j_parent, 2]], zdir='y', c=col)
                        ax2.plot([pos_gt[j, 0], pos_gt[j_parent, 0]],
                                                [-pos_gt[j, 1], -pos_gt[j_parent, 1]],
                                                [pos_gt[j, 2], pos_gt[j_parent, 2]], zdir='y', c=col_gt)
                debug_data.append({"id": int(i+198), "keypoints_gt": keypoints[i, :, :].reshape(-1).tolist(), "keypoints_3d_gt": pos_gt.reshape(-1).tolist(), "keypoints_3d": pos.reshape(-1).tolist()})
                plt.savefig("/root/no_backup/Debug_figure_{}.png".format(i))
                    
                
            image.set_data(all_frames[i])
            
            lines_mpjpe[0][0].set_color((202 / 255, 0 / 255, 32 / 255))
            lines_mpjpe[0][0].set_data(np.arange(i+1), mpjpe_out[:i+1])
            lines_mpjpe[0][0].set_label('MPJPE (Protocol #1)')
                
            lines_mpjpe[1][0].set_color((117/255,112/255,179/255))
            lines_mpjpe[1][0].set_data(np.arange(i+1), np.mean(p_mpjpe_out[:i+1], axis=1))
            lines_mpjpe[1][0].set_label('P-MPJPE (Protocol #2)')
            lines_mpjpe[2][0].set_color('cyan')
            lines_mpjpe[2][0].set_data(np.arange(i+1), n_mpjpe_out[:i+1])
            lines_mpjpe[2][0].set_label('N-MPJPE (Protocol #3)')
                        
            lines_veloc[0][0].set_color((202 / 255, 0 / 255, 32 / 255))
            lines_veloc[0][0].set_data(np.arange(i), np.mean(veloc_out[:i], axis=1))
 

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                
                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    if (keypoints[i, j, 0] != 0 and keypoints[i, j, 1] != 0) and (keypoints[i, j_parent, 0] != 0 and keypoints[i, j_parent, 1] != 0):
                        lines[j-1][0].set_color('pink')
                        lines[j-1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]])
                            
                    else:
                        lines[j-1][0].set_color('white')
                        lines[j-1][0].set_data([0, 0],
                                            [0, 0])

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    if (keypoints[i, j, 0] != 0 and keypoints[i, j, 1] != 0) and (keypoints[i, j_parent, 0] != 0 and keypoints[i, j_parent, 1] != 0):
                        lines_3d[n][j-1][0].set_color(colors_2d[j-1])
                        lines_3d[n][j-1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                        lines_3d[n][j-1][0].set_ydata(-np.array([pos[j, 1], pos[j_parent, 1]]))
                        lines_3d[n][j-1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='y')
                                      
                    else:
                        lines_3d[n][j-1][0].set_color("white")
                        lines_3d[n][j-1][0].set_xdata(np.array([0, 0]))
                        lines_3d[n][j-1][0].set_ydata(np.array([0, 0]))
                        lines_3d[n][j-1][0].set_3d_properties(np.array([0, 0]), zdir='y')

                # ADDED BONES
            if keypoints[i, 11, 0] != 0 and keypoints[i, 12, 0] != 0 and keypoints[i, 11, 1] != 0 and keypoints[i, 12, 1] != 0:
                lines[len(lines)-2][0].set_color('cornflowerblue')
                extra_parent = 12
                lines[len(lines)-2][0].set_data([keypoints[i, 11, 0], keypoints[i, extra_parent, 0]],
                                            [keypoints[i, 11, 1], keypoints[i, extra_parent, 1]])
            if  keypoints[i, 5, 0] != 0 and keypoints[i, 6, 0] != 0 and keypoints[i, 5, 1] != 0 and keypoints[i, 6, 1] != 0:
                lines[len(lines)-1][0].set_color('cornflowerblue')
                extra_parent = 6
                lines[len(lines)-1][0].set_data([keypoints[i, 5, 0], keypoints[i, extra_parent, 0]],
                                            [keypoints[i, 5, 1], keypoints[i, extra_parent, 1]])
            
            for n, ax in enumerate(ax_3d): 
                pos = poses[n][i]         
                if keypoints[i, 11, 0] != 0 and keypoints[i, 12, 0] != 0 and keypoints[i, 11, 1] != 0 and keypoints[i, 12, 1] != 0:
                    col = "blue"
                else:
                    col = "white"
                extra_parent = 12
                lines_3d[n][len(lines_3d[n])-2][0].set_color(col)
                lines_3d[n][len(lines_3d[n])-2][0].set_xdata(np.array([pos[11, 0], pos[extra_parent, 0]]))
                lines_3d[n][len(lines_3d[n])-2][0].set_ydata(-np.array([pos[11, 1], pos[extra_parent, 1]]))
                lines_3d[n][len(lines_3d[n])-2][0].set_3d_properties(np.array([pos[11, 2], pos[extra_parent, 2]]), zdir='y')
                            
                if  keypoints[i, 5, 0] != 0 and keypoints[i, 6, 0] != 0 and keypoints[i, 5, 1] != 0 and keypoints[i, 6, 1] != 0:
                    col = "blue"
                else:
                    col = "white"
                extra_parent = 6
                lines_3d[n][len(lines_3d[n])-1][0].set_color(col)
                lines_3d[n][len(lines_3d[n])-1][0].set_xdata(np.array([pos[5, 0], pos[extra_parent, 0]]))
                lines_3d[n][len(lines_3d[n])-1][0].set_ydata(-np.array([pos[5, 1], pos[extra_parent, 1]]))
                lines_3d[n][len(lines_3d[n])-1][0].set_3d_properties(np.array([pos[5, 2], pos[extra_parent, 2]]), zdir='y')
            points.set_offsets(keypoints[i])
            plt.legend()
        
        print('{}/{}      '.format(i, limit), end='\r')
        
    
    fig.tight_layout()
    
    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000/fps, repeat=False)
    if output.endswith('.mp4'):
        anim.save(output.replace('.mp4', '_ffmpeg.mp4'), writer='ffmpeg', fps=30)
        from IPython.display import HTML
        HTML(anim.to_html5_video())  
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
        #print("\n DEBUG Data contents: ", debug_data[0])
        with open("/root/no_backup/debug_images.json", "w") as jfile:
            json.dump(debug_data,  jfile, indent=2)
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()    
    



def render_animation(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0, elev=10.):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    plt.ioff()
    fig = plt.figure(figsize=(size*(1 + len(poses)), size))
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index+2, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim3d([-radius/2, radius/2])
        ax.set_ylim3d([0, radius])
        ax.set_ylim3d([radius/2, radius/2*3])
        ax.set_zlim3d([-radius/2, radius/2]) # exchange limits for y and z
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title) #, pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    poses = list(poses.values())

    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip, limit=limit):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]
        
        """
        keypoints = keypoints[input_video_skip:] # todo remove
        for idx in range(len(poses)):
            poses[idx] = poses[idx][input_video_skip:]
        """
        if fps is None:
            fps = get_fps(input_video_path)
    
    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
            trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None
    
    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    parents = skeleton.parents()
    def update_video(i):
        nonlocal initialized, image, lines, points
        print(" Frame {}: GT Neck {} Det Neck {}".format(i, poses[1][i][0], poses[0][i][0]))
        #for n, ax in enumerate(ax_3d):
        #    ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
        #    ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])

        # Update 2D poses
        joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
        colors_2d = np.full(keypoints.shape[1], 'black')
        colors_2d[joints_right_2d] = 'red'
        if not initialized:
            image = ax_in.imshow(all_frames[i+ input_video_skip], aspect='equal')
            h, w, _ = all_frames[i].shape
            
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue 
                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    if (keypoints[i, j, 0] > 0 and keypoints[i, j, 1] > 0) and (keypoints[i, j_parent, 0] > 0 and keypoints[i, j_parent, 1] > 0) \
                    and (keypoints[i, j, 0] < h and keypoints[i, j, 1] < w) and (keypoints[i, j_parent, 0] < h and keypoints[i, j_parent, 1] < w):
                        # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                        lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                                [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))
                    else:
                        lines.append(ax_in.plot([0, 0],
                                                [0, 0], color='white'))
                col = 'red' if j in skeleton.joints_right() else 'black'
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    if (keypoints[i, j, 0] != 0 and keypoints[i, j, 1] != 0) and (keypoints[i, j_parent, 0] != 0 and keypoints[i, j_parent, 1] != 0):
                        lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                                [-pos[j, 1], -pos[j_parent, 1]],
                                                [pos[j, 2], pos[j_parent, 2]], zdir='y', c=col))
                        
                    else:
                         lines_3d[n].append(ax.plot([0, 0],
                                                [0, 0],
                                                [0, 0], zdir='y', color="white"))

            
            if keypoints[i, 11, 0] != 0 and keypoints[i, 12, 0] != 0 and keypoints[i, 11, 1] != 0 and keypoints[i, 12, 1] != 0 :   
                col = "blue"
                extra_parent = 12
                lines.append(ax_in.plot([keypoints[i, 11, 0], keypoints[i, 12, 0]],
                                                [keypoints[i, 11, 1], keypoints[i, 12, 1]], color=col))
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[11, 0], pos[extra_parent, 0]],
                                                [-pos[11, 1], -pos[extra_parent, 1]],
                                                [pos[11, 2], pos[extra_parent, 2]], zdir='y', color=col))
            else:
                col = "white"
                lines.append(ax_in.plot([0, 0], [0, 0], color=col))
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([0, 0],
                                            [0, 0],
                                            [0, 0], zdir='y', color=col))
            
            
            if keypoints[i, 5, 0] != 0 and keypoints[i, 6, 0] != 0 and keypoints[i, 5, 1] != 0 and keypoints[i, 6, 1] != 0 :
                col = "blue"
                lines.append(ax_in.plot([keypoints[i, 5, 0], keypoints[i, 6, 0]],
                                            [keypoints[i, 5, 1], keypoints[i, 6, 1]], color=col))
                extra_parent = 6
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[5, 0], pos[extra_parent, 0]],
                                            [-pos[5, 1], -pos[extra_parent, 1]],
                                            [pos[5, 2], pos[extra_parent, 2]], zdir='y', color=col))
            else:
                col = "white"
                lines.append(ax_in.plot([0, 0], [0, 0], color=col))
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([0, 0],
                                            [0, 0],
                                            [0, 0], zdir='y', color=col))
            
            
                
            # vis_keypoints = keypoints[i][keypoints[i] != 0]
            # vis_keypoints = vis_keypoints.reshape(len(vis_keypoints) // 2, 2)
            # points = ax_in.scatter(*vis_keypoints.T, 10, color=colors_2d[keypoints[i] != 0], edgecolors='white', zorder=10)
            points = ax_in.scatter(*keypoints[i].T, 10, color=colors_2d, edgecolors='white', zorder=10)
            
            
            initialized = True
        else:
            image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                
                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    if (keypoints[i, j, 0] != 0 and keypoints[i, j, 1] != 0) and (keypoints[i, j_parent, 0] != 0 and keypoints[i, j_parent, 1] != 0):
                        lines[j-1][0].set_color('pink')
                        lines[j-1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]])
                            
                    else:
                        lines[j-1][0].set_color('white')
                        lines[j-1][0].set_data([0, 0],
                                            [0, 0])

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    if (keypoints[i, j, 0] != 0 and keypoints[i, j, 1] != 0) and (keypoints[i, j_parent, 0] != 0 and keypoints[i, j_parent, 1] != 0):
                        lines_3d[n][j-1][0].set_color(colors_2d[j-1])
                        lines_3d[n][j-1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                        lines_3d[n][j-1][0].set_ydata(-np.array([pos[j, 1], pos[j_parent, 1]]))
                        lines_3d[n][j-1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='y')
                                      
                    else:
                        lines_3d[n][j-1][0].set_color("white")
                        lines_3d[n][j-1][0].set_xdata(np.array([0, 0]))
                        lines_3d[n][j-1][0].set_ydata(np.array([0, 0]))
                        lines_3d[n][j-1][0].set_3d_properties(np.array([0, 0]), zdir='y')

                # ADDED BONES
            if keypoints[i, 11, 0] != 0 and keypoints[i, 12, 0] != 0 and keypoints[i, 11, 1] != 0 and keypoints[i, 12, 1] != 0:
                lines[len(lines)-2][0].set_color('cornflowerblue')
                extra_parent = 12
                lines[len(lines)-2][0].set_data([keypoints[i, 11, 0], keypoints[i, extra_parent, 0]],
                                            [keypoints[i, 11, 1], keypoints[i, extra_parent, 1]])
            if  keypoints[i, 5, 0] != 0 and keypoints[i, 6, 0] != 0 and keypoints[i, 5, 1] != 0 and keypoints[i, 6, 1] != 0:
                lines[len(lines)-1][0].set_color('cornflowerblue')
                extra_parent = 6
                lines[len(lines)-1][0].set_data([keypoints[i, 5, 0], keypoints[i, extra_parent, 0]],
                                            [keypoints[i, 5, 1], keypoints[i, extra_parent, 1]])
            
            for n, ax in enumerate(ax_3d): 
                pos = poses[n][i]         
                if keypoints[i, 11, 0] != 0 and keypoints[i, 12, 0] != 0 and keypoints[i, 11, 1] != 0 and keypoints[i, 12, 1] != 0:
                    col = "blue"
                else:
                    col = "white"
                extra_parent = 12
                lines_3d[n][len(lines_3d[n])-2][0].set_color(col)
                lines_3d[n][len(lines_3d[n])-2][0].set_xdata(np.array([pos[11, 0], pos[extra_parent, 0]]))
                lines_3d[n][len(lines_3d[n])-2][0].set_ydata(-np.array([pos[11, 1], pos[extra_parent, 1]]))
                lines_3d[n][len(lines_3d[n])-2][0].set_3d_properties(np.array([pos[11, 2], pos[extra_parent, 2]]), zdir='y')
                            
                if  keypoints[i, 5, 0] != 0 and keypoints[i, 6, 0] != 0 and keypoints[i, 5, 1] != 0 and keypoints[i, 6, 1] != 0:
                    col = "blue"
                else:
                    col = "white"
                extra_parent = 6
                lines_3d[n][len(lines_3d[n])-1][0].set_color(col)
                lines_3d[n][len(lines_3d[n])-1][0].set_xdata(np.array([pos[5, 0], pos[extra_parent, 0]]))
                lines_3d[n][len(lines_3d[n])-1][0].set_ydata(-np.array([pos[5, 1], pos[extra_parent, 1]]))
                lines_3d[n][len(lines_3d[n])-1][0].set_3d_properties(np.array([pos[5, 2], pos[extra_parent, 2]]), zdir='y')
            
            points.set_offsets(keypoints[i])
        
        print('{}/{}      '.format(i, limit), end='\r')
        

    fig.tight_layout()
    
    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000/fps, repeat=False)
    if output.endswith('.mp4'):
        from IPython.display import HTML
        HTML(anim.to_html5_video())  
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()





def render_several_animations(checkpoints, keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0, elev=10., watch_video=True, video_start=0, video_end=0):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    list_mpjpe_out = []
    list_n_mpjpe_out = []
    list_p_mpjpe_out = []
    list_veloc_out = []
    
    
    whole_poses = np.array(poses["Reconstruction"])
    poses_gt = np.array(poses["Ground truth"])

    if  "gt" in checkpoints[0]:
        whole_poses = np.concatenate((np.expand_dims(poses_gt, axis=0), whole_poses), axis=0)
    
    for pose in whole_poses:    
        poses_pred = pose
    
        #poses_pred = poses_pred[:limit-input_video_skip]
        #poses_gt = poses_gt[:limit-input_video_skip]
        m2mm = 1000
        new_poses_pred = np.expand_dims(poses_pred, axis=0)
        new_poses_gt = np.expand_dims(poses_gt, axis=0)
    
        # Compute errors between predictions and Ground Truth
        with torch.no_grad():
            mpjpe_out = mpjpe_eval(poses_pred, poses_gt) * m2mm
            n_mpjpe_out = n_mpjpe_eval(poses_pred, poses_gt) * m2mm
            p_mpjpe_out = p_mpjpe(poses_pred, poses_gt, mode='visu') * m2mm
            
            veloc_out =  mean_velocity_error(poses_pred, poses_gt, mode='visu') * m2mm

            list_mpjpe_out.append(mpjpe_out.tolist())
            list_n_mpjpe_out.append(n_mpjpe_out.tolist())
            list_p_mpjpe_out.append(p_mpjpe_out.tolist())
            list_veloc_out.append(veloc_out)

    scores = dict(names=list(checkpoints),  
            mpjpe=list_mpjpe_out, 
            mpjve=[list(p_m) for p_m in list_veloc_out], 
            n_mpjpe=list_n_mpjpe_out, 
            p_mpjpe=[list(p_m) for p_m in list_p_mpjpe_out])
    preds_and_gt = dict(preds=[pose.tolist() for pose in poses['Reconstruction']], gt=[pose.tolist() for pose in poses['Ground truth']],names=list(checkpoints))
    for k in scores.keys():
        print(k, type(scores[k][0][0]))
    import json
    with open("/root/no_backup/scores_3x3x3_wholebody_occlusions_with_conf.json", "w") as json_file:
        json.dump(scores, json_file, indent=2)
    with open("/root/no_backup/preds_and_gt_3x3x3_wholebody_occlusions_with_conf.json", "w") as json_file2:
        json.dump(preds_and_gt, json_file2)
      
    plt.ioff()
    fig = plt.figure(figsize=(size*(1 + len(whole_poses)), 2*size))
    ax_in = fig.add_subplot(2, 1 + len(whole_poses), 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')
    
    SUB_FIG_SIZE = size
    
    ax_3d = []
    lines_3d = []
    trajectories = []
    lines_mpjpe = []
    
    radius = 1.7
    new_poses = {}
    for i, ch in enumerate(checkpoints):
        if "gt" in ch:
            new_poses["Ground Truth"] = whole_poses[i]
        else:
            new_poses[ch.split('/')[-2]] = whole_poses[i]
    for index, (title, data) in enumerate(new_poses.items()):
        ax = fig.add_subplot(2, 1 + len(whole_poses), index+2, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim3d([-radius/2, radius/2])
        ax.set_ylim3d([0, radius])
        ax.set_ylim3d([radius/2, radius/2*3])
        ax.set_zlim3d([-radius/2, radius/2]) # exchange limits for y and z
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title) #, pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])

    
    ax_mpjpe = fig.add_subplot(2, 1, 2)
    ax_mpjpe.set_title('MPJPE Visualize', fontsize=3*SUB_FIG_SIZE)
    
    ax_mpjpe.legend()
    ax_mpjpe.grid(True)
    ax_mpjpe.set_xlabel('Frame', fontsize=3*SUB_FIG_SIZE)
    ax_mpjpe.set_ylabel('Mean Position Error (mm)', fontsize=3*SUB_FIG_SIZE)
    ax_mpjpe.set_xlim((0, limit))
    ax_mpjpe.set_ylim((0, 150))
    ax_mpjpe.tick_params(axis="x", labelsize=3*SUB_FIG_SIZE)
    ax_mpjpe.tick_params(axis="y", labelsize=3*SUB_FIG_SIZE)
    ax_mpjpe.legend(fontsize='large') # fontsize=6*SUB_FIG_SIZE, prop={"size":14}
    

    
    poses = list(poses.values())
    all_frames =  np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    """
    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip, limit=limit):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]
        

        if fps is None:
            fps = get_fps(input_video_path)
    """

    if not input_video_path.endswith('.mp4'):
        import os
        all_frames = np.array([plt.imread(input_video_path + "_frame_" + str(frame_id) + "_resized.jpg") for frame_id in range(video_start, video_end)])
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
            trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None
    
    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))
    
    debug_data = []

    parents = skeleton.parents()
    def update_video(i):
        nonlocal initialized, image, lines, points
        
        # Update 2D poses
        joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
        colors_2d = np.full(keypoints.shape[1], 'black')
        colors_2d[joints_right_2d] = 'red'
        # print("Metrics shapes: ", mpjpe_out.shape, p_mpjpe_out.shape, n_mpjpe_out.shape, veloc_out.shape)
        if not initialized:
            if watch_video:
                image = ax_in.imshow(all_frames[i], aspect='equal')
                h, w, _ = all_frames[i].shape
            else:
                h, w, _ = 256, 320, 0 #all_frames[i + input_video_skip].shape
            
            for id in range(len(list_mpjpe_out)):
                lines_mpjpe.append(ax_mpjpe.plot(np.arange(i+1), list_mpjpe_out[id][:i+1], linewidth=2, color=colors_list[id],
                    label=list(new_poses.keys())[id]))

            
            
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue 
                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    if (keypoints[i, j, 0] > 0 and keypoints[i, j, 1] > 0) and (keypoints[i, j_parent, 0] > 0 and keypoints[i, j_parent, 1] > 0) \
                    and (keypoints[i, j, 0] < h and keypoints[i, j, 1] < w) and (keypoints[i, j_parent, 0] < h and keypoints[i, j_parent, 1] < w):
                        # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                        lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                                [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))
                    else:
                        lines.append(ax_in.plot([0, 0],
                                                [0, 0], color='white'))
                col = 'red' if j in skeleton.joints_right() else 'black'
                for n, ax in enumerate(ax_3d):
                    pos = whole_poses[n][i]
                    if (keypoints[i, j, 0] != 0 and keypoints[i, j, 1] != 0) and (keypoints[i, j_parent, 0] != 0 and keypoints[i, j_parent, 1] != 0):
                        lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                                [-pos[j, 1], -pos[j_parent, 1]],
                                                [pos[j, 2], pos[j_parent, 2]], zdir='y',  c=(202 / 255, 0 / 255, 32 / 255)))
                        
                    else:
                        lines_3d[n].append(ax.plot([0, 0],
                                                [0, 0],
                                                [0, 0], zdir='y', color="white"))

            
            if keypoints[i, 11, 0] != 0 and keypoints[i, 12, 0] != 0 and keypoints[i, 11, 1] != 0 and keypoints[i, 12, 1] != 0 :   
                col = "blue"
                extra_parent = 12
                lines.append(ax_in.plot([keypoints[i, 11, 0], keypoints[i, 12, 0]],
                                                [keypoints[i, 11, 1], keypoints[i, 12, 1]], color=col))
                for n, ax in enumerate(ax_3d):
                    pos = whole_poses[n][i]
                    lines_3d[n].append(ax.plot([pos[11, 0], pos[extra_parent, 0]],
                                                [-pos[11, 1], -pos[extra_parent, 1]],
                                                [pos[11, 2], pos[extra_parent, 2]], zdir='y', color=(202 / 255, 0 / 255, 32 / 255)))
        
            else:
                col = "white"
                lines.append(ax_in.plot([0, 0], [0, 0], color=col))
                for n, ax in enumerate(ax_3d):
                    pos = whole_poses[n][i]
                    lines_3d[n].append(ax.plot([0, 0],
                                            [0, 0],
                                            [0, 0], zdir='y', color=col))
            
            
            if keypoints[i, 5, 0] != 0 and keypoints[i, 6, 0] != 0 and keypoints[i, 5, 1] != 0 and keypoints[i, 6, 1] != 0 :
                col = "blue"
                lines.append(ax_in.plot([keypoints[i, 5, 0], keypoints[i, 6, 0]],
                                            [keypoints[i, 5, 1], keypoints[i, 6, 1]], color=col))
                extra_parent = 6
                for n, ax in enumerate(ax_3d):
                    pos = whole_poses[n][i]
                    lines_3d[n].append(ax.plot([pos[5, 0], pos[extra_parent, 0]],
                                            [-pos[5, 1], -pos[extra_parent, 1]],
                                            [pos[5, 2], pos[extra_parent, 2]], zdir='y', color=(202 / 255, 0 / 255, 32 / 255)))
        
            else:
                col = "white"
                lines.append(ax_in.plot([0, 0], [0, 0], color=col))
                for n, ax in enumerate(ax_3d):
                    pos = whole_poses[n][i]
                    lines_3d[n].append(ax.plot([0, 0],
                                            [0, 0],
                                            [0, 0], zdir='y', color=col))
            
            
            points = ax_in.scatter(*keypoints[i].T, 20, color=colors_2d, edgecolors='white', zorder=20)
            plt.savefig("/root/no_backup/several_debug_figure_{}.png".format(i))
            # print(len(lines), len(lines[0]),len(lines_3d), len(lines_3d[0]), len(lines_3d[1]), len(lines_3d[2]) )
            initialized = True
            
        else:

            image.set_data(all_frames[i])
            
            """
            lines_mpjpe[0][0].set_color((202 / 255, 0 / 255, 32 / 255))
            lines_mpjpe[0][0].set_data(np.arange(i+1), mpjpe_out[:i+1])
            lines_mpjpe[0][0].set_label('MPJPE (Protocol #1)')
                
            lines_mpjpe[1][0].set_color((117/255,112/255,179/255))
            lines_mpjpe[1][0].set_data(np.arange(i+1), np.mean(p_mpjpe_out[:i+1], axis=1))
            lines_mpjpe[1][0].set_label('P-MPJPE (Protocol #2)')
            lines_mpjpe[2][0].set_color('cyan')
            lines_mpjpe[2][0].set_data(np.arange(i+1), n_mpjpe_out[:i+1])
            lines_mpjpe[2][0].set_label('N-MPJPE (Protocol #3)')
                        
            lines_veloc[0][0].set_color((202 / 255, 0 / 255, 32 / 255))
            lines_veloc[0][0].set_data(np.arange(i), np.mean(veloc_out[:i], axis=1))
            """

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                
                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    if (keypoints[i, j, 0] != 0 and keypoints[i, j, 1] != 0) and (keypoints[i, j_parent, 0] != 0 and keypoints[i, j_parent, 1] != 0):
                        lines[j-1][0].set_color('pink')
                        lines[j-1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]])
                            
                    else:
                        lines[j-1][0].set_color('white')
                        lines[j-1][0].set_data([0, 0],
                                            [0, 0])

                for n, ax in enumerate(ax_3d):
                    pos = whole_poses[n][i]
                    lines_mpjpe[n][0].set_color(colors_list[n])
                    lines_mpjpe[n][0].set_data(np.arange(i+1), list_mpjpe_out[n][:i+1])
                    lines_mpjpe[n][0].set_label(checkpoints[n].split('_with_conf')[0])
                    lines_mpjpe[n][0].set_linewidth(2.0)
                    """
                    lines_veloc[n][0].set_color(((202-n*45) / 255, 0+n*45 / 255, (32+n**2*15) / 255))
                    lines_veloc[n][0].set_data(np.arange(i+1), list_veloc_out[n][:i+1])
                    lines_veloc[n][0].set_label(checkpoints[n].split('_with_conf')[0])
                    lines_veloc[n][0].set_linewidth(2.0)
                    """
                    if (keypoints[i, j, 0] != 0 and keypoints[i, j, 1] != 0) and (keypoints[i, j_parent, 0] != 0 and keypoints[i, j_parent, 1] != 0):
                        lines_3d[n][j-1][0].set_color(colors_2d[j-1])
                        lines_3d[n][j-1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                        lines_3d[n][j-1][0].set_ydata(-np.array([pos[j, 1], pos[j_parent, 1]]))
                        lines_3d[n][j-1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='y')
                                      
                    else:
                        lines_3d[n][j-1][0].set_color("white")
                        lines_3d[n][j-1][0].set_xdata(np.array([0, 0]))
                        lines_3d[n][j-1][0].set_ydata(np.array([0, 0]))
                        lines_3d[n][j-1][0].set_3d_properties(np.array([0, 0]), zdir='y')

                # ADDED BONES
            if keypoints[i, 11, 0] != 0 and keypoints[i, 12, 0] != 0 and keypoints[i, 11, 1] != 0 and keypoints[i, 12, 1] != 0:
                lines[len(lines)-2][0].set_color('cornflowerblue')
                extra_parent = 12
                lines[len(lines)-2][0].set_data([keypoints[i, 11, 0], keypoints[i, extra_parent, 0]],
                                            [keypoints[i, 11, 1], keypoints[i, extra_parent, 1]])
            if  keypoints[i, 5, 0] != 0 and keypoints[i, 6, 0] != 0 and keypoints[i, 5, 1] != 0 and keypoints[i, 6, 1] != 0:
                lines[len(lines)-1][0].set_color('cornflowerblue')
                extra_parent = 6
                lines[len(lines)-1][0].set_data([keypoints[i, 5, 0], keypoints[i, extra_parent, 0]],
                                            [keypoints[i, 5, 1], keypoints[i, extra_parent, 1]])
            
            for n, ax in enumerate(ax_3d): 
                pos = whole_poses[n][i]         
                if keypoints[i, 11, 0] != 0 and keypoints[i, 12, 0] != 0 and keypoints[i, 11, 1] != 0 and keypoints[i, 12, 1] != 0:
                    col = colors_list[n]
                else:
                    col = "white"
                extra_parent = 12
                lines_3d[n][len(lines_3d[n])-2][0].set_color(col)
                lines_3d[n][len(lines_3d[n])-2][0].set_xdata(np.array([pos[11, 0], pos[extra_parent, 0]]))
                lines_3d[n][len(lines_3d[n])-2][0].set_ydata(-np.array([pos[11, 1], pos[extra_parent, 1]]))
                lines_3d[n][len(lines_3d[n])-2][0].set_3d_properties(np.array([pos[11, 2], pos[extra_parent, 2]]), zdir='y')
                            
                if  keypoints[i, 5, 0] != 0 and keypoints[i, 6, 0] != 0 and keypoints[i, 5, 1] != 0 and keypoints[i, 6, 1] != 0:
                    col = col = colors_list[n]
                else:
                    col = "white"
                extra_parent = 6
                lines_3d[n][len(lines_3d[n])-1][0].set_color(col)
                lines_3d[n][len(lines_3d[n])-1][0].set_xdata(np.array([pos[5, 0], pos[extra_parent, 0]]))
                lines_3d[n][len(lines_3d[n])-1][0].set_ydata(-np.array([pos[5, 1], pos[extra_parent, 1]]))
                lines_3d[n][len(lines_3d[n])-1][0].set_3d_properties(np.array([pos[5, 2], pos[extra_parent, 2]]), zdir='y')
            points.set_offsets(keypoints[i])
            plt.legend()
        
        print('{}/{}      '.format(i, limit), end='\r')
        
    
    fig.tight_layout()
    
    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000/fps, repeat=False)
    if output.endswith('.mp4'):
        anim.save(output.replace('.mp4', '_ffmpeg.mp4'), writer='ffmpeg', fps=30)
        from IPython.display import HTML
        HTML(anim.to_html5_video())  
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()    
    


def render_modified_animations(checkpoints, keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0, elev=10., watch_video=True, video_start=0, video_end=0):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    list_mpjpe_out = []
    list_n_mpjpe_out = []
    list_p_mpjpe_out = []
    list_veloc_out = []
    
    
    whole_poses = np.array(poses["Reconstruction"])
    poses_gt = np.array(poses["Ground truth"])
    
    for pose in whole_poses:    
        poses_pred = pose
    
        #poses_pred = poses_pred[:limit-input_video_skip]
        #poses_gt = poses_gt[:limit-input_video_skip]
        m2mm = 1000
        new_poses_pred = np.expand_dims(poses_pred, axis=0)
        new_poses_gt = np.expand_dims(poses_gt, axis=0)
    
        # Compute errors between predictions and Ground Truth
        with torch.no_grad():
            mpjpe_out = mpjpe_eval(poses_pred, poses_gt) * m2mm
            n_mpjpe_out = n_mpjpe_eval(poses_pred, poses_gt) * m2mm
            p_mpjpe_out = p_mpjpe(poses_pred, poses_gt, mode='visu') * m2mm
            
            veloc_out =  mean_velocity_error(poses_pred, poses_gt, mode='visu') * m2mm

            list_mpjpe_out.append(mpjpe_out.tolist())
            list_n_mpjpe_out.append(n_mpjpe_out.tolist())
            list_p_mpjpe_out.append(p_mpjpe_out.tolist())
            list_veloc_out.append(veloc_out)

    scores = dict(names=list(checkpoints),  
            mpjpe=list_mpjpe_out, 
            mpjve=[list(p_m) for p_m in list_veloc_out], 
            n_mpjpe=list_n_mpjpe_out, 
            p_mpjpe=[list(p_m) for p_m in list_p_mpjpe_out])
    preds_and_gt = dict(preds=[pose.tolist() for pose in poses['Reconstruction']], gt=[pose.tolist() for pose in poses['Ground truth']],names=list(checkpoints))
    for k in scores.keys():
        print(k, type(scores[k][0][0]))
    import json
    with open("/root/no_backup/scores_3x3x3_wholebody_occlusions_with_conf.json", "w") as json_file:
        json.dump(scores, json_file, indent=2)
    with open("/root/no_backup/preds_and_gt_3x3x3_wholebody_occlusions_with_conf.json", "w") as json_file2:
        json.dump(preds_and_gt, json_file2)
      
    plt.ioff()
    fig = plt.figure(figsize=(size*(1 + len(whole_poses)), 3*size))
    ax_in = fig.add_subplot(3, 1 + len(whole_poses), 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')
    
    SUB_FIG_SIZE = size
    
    ax_3d = []
    lines_3d = []
    trajectories = []
    lines_veloc = []
    lines_mpjpe = []
    
    radius = 1.7
    new_poses = {}
    for i, ch in enumerate(checkpoints):
        new_poses[ch.split('/')[-2]] = whole_poses[i]
    for index, (title, data) in enumerate(new_poses.items()):
        ax = fig.add_subplot(3, 1 + len(whole_poses), index+2, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim3d([-radius/2, radius/2])
        ax.set_ylim3d([0, radius])
        ax.set_ylim3d([radius/2, radius/2*3])
        ax.set_zlim3d([-radius/2, radius/2]) # exchange limits for y and z
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title) #, pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    
    ax_vel = fig.add_subplot(3, 1, 3)
    ax_vel.set_title('Velocity Error Visualize', fontsize=3*SUB_FIG_SIZE)

    ax_vel.legend()
    ax_vel.grid(True)
    ax_vel.set_xlabel('Frame', fontsize=3*SUB_FIG_SIZE)
    ax_vel.set_ylabel('Mean Velocity Error (mm/s)', fontsize=3*SUB_FIG_SIZE)
    ax_vel.set_xlim((0, limit))
    ax_vel.set_ylim((0, 150))
    
    ax_mpjpe = fig.add_subplot(3, 1, 2)
    ax_mpjpe.set_title('MPJPE Visualize', fontsize=3*SUB_FIG_SIZE)
    
    ax_mpjpe.legend()
    ax_mpjpe.grid(True)
    ax_mpjpe.set_xlabel('Frame', fontsize=3*SUB_FIG_SIZE)
    ax_mpjpe.set_ylabel('Mean Position Error (mm)', fontsize=3*SUB_FIG_SIZE)
    ax_mpjpe.set_xlim((0, limit))
    ax_mpjpe.set_ylim((0, 150))
    ax_mpjpe.tick_params(axis="x", labelsize=3*SUB_FIG_SIZE)
    ax_mpjpe.tick_params(axis="y", labelsize=3*SUB_FIG_SIZE)
    ax_mpjpe.legend(fontsize='large') # fontsize=6*SUB_FIG_SIZE, prop={"size":14}
    

    
    poses = list(poses.values())
    all_frames =  np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    """
    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip, limit=limit):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]
        

        if fps is None:
            fps = get_fps(input_video_path)
    """

    if not input_video_path.endswith('.mp4'):
        import os
        all_frames = np.array([plt.imread(input_video_path + "_frame_" + str(frame_id) + "_resized.jpg") for frame_id in range(video_start, video_end)])
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
            trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None
    
    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))
    
    debug_data = []

    parents = skeleton.parents()
    def update_video(i):
        nonlocal initialized, image, lines, points
        
        # Update 2D poses
        joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
        colors_2d = np.full(keypoints.shape[1], 'black')
        colors_2d[joints_right_2d] = 'red'
        # print("Metrics shapes: ", mpjpe_out.shape, p_mpjpe_out.shape, n_mpjpe_out.shape, veloc_out.shape)
        if not initialized:
            if watch_video:
                image = ax_in.imshow(all_frames[i], aspect='equal')
                h, w, _ = all_frames[i].shape
            else:
                h, w, _ = 256, 320, 0 #all_frames[i + input_video_skip].shape
            
            for id in range(len(list_mpjpe_out)):
                lines_mpjpe.append(ax_mpjpe.plot(np.arange(i+1), list_mpjpe_out[id][:i+1], linewidth=2, color=((202 - id*45) / 255, (0+id*45) / 255, (32+id**2*15) / 255),
                    label=list(new_poses.keys())[id]))
                """
                lines_mpjpe.append(ax_mpjpe.plot(np.arange(i+1), np.mean(p_mpjpe_out[:i+1], axis=1),
                color=(117/255,112/255,179/255),
                label='P-MPJPE (Protocol #2)'))
                lines_mpjpe.append(ax_mpjpe.plot(np.arange(i+1), n_mpjpe_out[:i+1],
                        color='c',
                        label='Normalized MPJPE (Protocol #3)'))
                """
                lines_veloc.append(ax_vel.plot(np.arange(i+1), np.mean(list_veloc_out[id][:i+1], axis=1),
                 color=((202 - id*45) / 255, (0+id*45) / 255, (32+id**2*15) / 255) , linewidth=2,
                label=list(new_poses.keys())[id]))
            
            
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue 
                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    if (keypoints[i, j, 0] > 0 and keypoints[i, j, 1] > 0) and (keypoints[i, j_parent, 0] > 0 and keypoints[i, j_parent, 1] > 0) \
                    and (keypoints[i, j, 0] < h and keypoints[i, j, 1] < w) and (keypoints[i, j_parent, 0] < h and keypoints[i, j_parent, 1] < w):
                        # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                        lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                                [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))
                    else:
                        lines.append(ax_in.plot([0, 0],
                                                [0, 0], color='white'))
                col = 'red' if j in skeleton.joints_right() else 'black'
                for n, ax in enumerate(ax_3d):
                    pos = whole_poses[n][i]
                    if (keypoints[i, j, 0] != 0 and keypoints[i, j, 1] != 0) and (keypoints[i, j_parent, 0] != 0 and keypoints[i, j_parent, 1] != 0):
                        lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                                [-pos[j, 1], -pos[j_parent, 1]],
                                                [pos[j, 2], pos[j_parent, 2]], zdir='y',  c=((202-n*45) / 255, 0+n*45 / 255, (32+n**2*15) / 255)))
                        
                    else:
                        lines_3d[n].append(ax.plot([0, 0],
                                                [0, 0],
                                                [0, 0], zdir='y', color="white"))

            
            if keypoints[i, 11, 0] != 0 and keypoints[i, 12, 0] != 0 and keypoints[i, 11, 1] != 0 and keypoints[i, 12, 1] != 0 :   
                col = "blue"
                extra_parent = 12
                lines.append(ax_in.plot([keypoints[i, 11, 0], keypoints[i, 12, 0]],
                                                [keypoints[i, 11, 1], keypoints[i, 12, 1]], color=col))
                for n, ax in enumerate(ax_3d):
                    pos = whole_poses[n][i]
                    lines_3d[n].append(ax.plot([pos[11, 0], pos[extra_parent, 0]],
                                                [-pos[11, 1], -pos[extra_parent, 1]],
                                                [pos[11, 2], pos[extra_parent, 2]], zdir='y', color=((202-n*45) / 255, 0+n*45 / 255, (32+n**2*15) / 255)))
        
            else:
                col = "white"
                lines.append(ax_in.plot([0, 0], [0, 0], color=col))
                for n, ax in enumerate(ax_3d):
                    pos = whole_poses[n][i]
                    lines_3d[n].append(ax.plot([0, 0],
                                            [0, 0],
                                            [0, 0], zdir='y', color=((202-n*45) / 255, 0+n*45 / 255, (32+n**2*15) / 255)))
            
            
            if keypoints[i, 5, 0] != 0 and keypoints[i, 6, 0] != 0 and keypoints[i, 5, 1] != 0 and keypoints[i, 6, 1] != 0 :
                col = "blue"
                lines.append(ax_in.plot([keypoints[i, 5, 0], keypoints[i, 6, 0]],
                                            [keypoints[i, 5, 1], keypoints[i, 6, 1]], color=col))
                extra_parent = 6
                for n, ax in enumerate(ax_3d):
                    pos = whole_poses[n][i]
                    lines_3d[n].append(ax.plot([pos[5, 0], pos[extra_parent, 0]],
                                            [-pos[5, 1], -pos[extra_parent, 1]],
                                            [pos[5, 2], pos[extra_parent, 2]], zdir='y', color=((202-n*45) / 255, 0+n*45 / 255, (32+n**2*15) / 255)))
        
            else:
                col = "white"
                lines.append(ax_in.plot([0, 0], [0, 0], color=col))
                for n, ax in enumerate(ax_3d):
                    pos = whole_poses[n][i]
                    lines_3d[n].append(ax.plot([0, 0],
                                            [0, 0],
                                            [0, 0], zdir='y', color=((202-n*45) / 255, 0+n*45 / 255, (32+n**2*15) / 255)))
            
            
            points = ax_in.scatter(*keypoints[i].T, 20, color=colors_2d, edgecolors='white', zorder=20)
            plt.savefig("/root/no_backup/several_debug_figure_{}.png".format(i))
            print(len(lines), len(lines[0]),len(lines_3d), len(lines_3d[0]), len(lines_3d[1]), len(lines_3d[2]) )
            initialized = True
            
        else:

            image.set_data(all_frames[i])
            
            """
            lines_mpjpe[0][0].set_color((202 / 255, 0 / 255, 32 / 255))
            lines_mpjpe[0][0].set_data(np.arange(i+1), mpjpe_out[:i+1])
            lines_mpjpe[0][0].set_label('MPJPE (Protocol #1)')
                
            lines_mpjpe[1][0].set_color((117/255,112/255,179/255))
            lines_mpjpe[1][0].set_data(np.arange(i+1), np.mean(p_mpjpe_out[:i+1], axis=1))
            lines_mpjpe[1][0].set_label('P-MPJPE (Protocol #2)')
            lines_mpjpe[2][0].set_color('cyan')
            lines_mpjpe[2][0].set_data(np.arange(i+1), n_mpjpe_out[:i+1])
            lines_mpjpe[2][0].set_label('N-MPJPE (Protocol #3)')
                        
            lines_veloc[0][0].set_color((202 / 255, 0 / 255, 32 / 255))
            lines_veloc[0][0].set_data(np.arange(i), np.mean(veloc_out[:i], axis=1))
            """

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                
                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    if (keypoints[i, j, 0] != 0 and keypoints[i, j, 1] != 0) and (keypoints[i, j_parent, 0] != 0 and keypoints[i, j_parent, 1] != 0):
                        lines[j-1][0].set_color('pink')
                        lines[j-1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]])
                            
                    else:
                        lines[j-1][0].set_color('white')
                        lines[j-1][0].set_data([0, 0],
                                            [0, 0])

                for n, ax in enumerate(ax_3d):
                    pos = whole_poses[n][i]
                    lines_mpjpe[n][0].set_color(((202-n*45) / 255, 0+n*45 / 255, (32+n**2*15) / 255))
                    lines_mpjpe[n][0].set_data(np.arange(i+1), list_mpjpe_out[n][:i+1])
                    lines_mpjpe[n][0].set_label(checkpoints[n].split('_with_conf')[0])
                    lines_mpjpe[n][0].set_linewidth(2.0)
                    """
                    lines_veloc[n][0].set_color(((202-n*45) / 255, 0+n*45 / 255, (32+n**2*15) / 255))
                    lines_veloc[n][0].set_data(np.arange(i+1), list_veloc_out[n][:i+1])
                    lines_veloc[n][0].set_label(checkpoints[n].split('_with_conf')[0])
                    lines_veloc[n][0].set_linewidth(2.0)
                    """
                    if (keypoints[i, j, 0] != 0 and keypoints[i, j, 1] != 0) and (keypoints[i, j_parent, 0] != 0 and keypoints[i, j_parent, 1] != 0):
                        lines_3d[n][j-1][0].set_color(colors_2d[j-1])
                        lines_3d[n][j-1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                        lines_3d[n][j-1][0].set_ydata(-np.array([pos[j, 1], pos[j_parent, 1]]))
                        lines_3d[n][j-1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='y')
                                      
                    else:
                        lines_3d[n][j-1][0].set_color("white")
                        lines_3d[n][j-1][0].set_xdata(np.array([0, 0]))
                        lines_3d[n][j-1][0].set_ydata(np.array([0, 0]))
                        lines_3d[n][j-1][0].set_3d_properties(np.array([0, 0]), zdir='y')

                # ADDED BONES
            if keypoints[i, 11, 0] != 0 and keypoints[i, 12, 0] != 0 and keypoints[i, 11, 1] != 0 and keypoints[i, 12, 1] != 0:
                lines[len(lines)-2][0].set_color('cornflowerblue')
                extra_parent = 12
                lines[len(lines)-2][0].set_data([keypoints[i, 11, 0], keypoints[i, extra_parent, 0]],
                                            [keypoints[i, 11, 1], keypoints[i, extra_parent, 1]])
            if  keypoints[i, 5, 0] != 0 and keypoints[i, 6, 0] != 0 and keypoints[i, 5, 1] != 0 and keypoints[i, 6, 1] != 0:
                lines[len(lines)-1][0].set_color('cornflowerblue')
                extra_parent = 6
                lines[len(lines)-1][0].set_data([keypoints[i, 5, 0], keypoints[i, extra_parent, 0]],
                                            [keypoints[i, 5, 1], keypoints[i, extra_parent, 1]])
            
            for n, ax in enumerate(ax_3d): 
                pos = whole_poses[n][i]         
                if keypoints[i, 11, 0] != 0 and keypoints[i, 12, 0] != 0 and keypoints[i, 11, 1] != 0 and keypoints[i, 12, 1] != 0:
                    col = ((202-n*45) / 255, 0+n*45 / 255, (32+n**2*15) / 255)
                else:
                    col = "white"
                extra_parent = 12
                lines_3d[n][len(lines_3d[n])-2][0].set_color(col)
                lines_3d[n][len(lines_3d[n])-2][0].set_xdata(np.array([pos[11, 0], pos[extra_parent, 0]]))
                lines_3d[n][len(lines_3d[n])-2][0].set_ydata(-np.array([pos[11, 1], pos[extra_parent, 1]]))
                lines_3d[n][len(lines_3d[n])-2][0].set_3d_properties(np.array([pos[11, 2], pos[extra_parent, 2]]), zdir='y')
                            
                if  keypoints[i, 5, 0] != 0 and keypoints[i, 6, 0] != 0 and keypoints[i, 5, 1] != 0 and keypoints[i, 6, 1] != 0:
                    col = ((202-n*45) / 255, 0+n*45 / 255, (32+n**2*15) / 255)
                else:
                    col = "white"
                extra_parent = 6
                lines_3d[n][len(lines_3d[n])-1][0].set_color(col)
                lines_3d[n][len(lines_3d[n])-1][0].set_xdata(np.array([pos[5, 0], pos[extra_parent, 0]]))
                lines_3d[n][len(lines_3d[n])-1][0].set_ydata(-np.array([pos[5, 1], pos[extra_parent, 1]]))
                lines_3d[n][len(lines_3d[n])-1][0].set_3d_properties(np.array([pos[5, 2], pos[extra_parent, 2]]), zdir='y')
            points.set_offsets(keypoints[i])
            plt.legend()
        
        print('{}/{}      '.format(i, limit), end='\r')
        
    
    fig.tight_layout()
    
    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000/fps, repeat=False)
    if output.endswith('.mp4'):
        anim.save(output.replace('.mp4', '_ffmpeg.mp4'), writer='ffmpeg', fps=30)
        from IPython.display import HTML
        HTML(anim.to_html5_video())  
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()    
    
