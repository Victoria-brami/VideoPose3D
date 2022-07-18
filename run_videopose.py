import subprocess

# JOB 21059
call_ = 'python run.py --epochs 40 --keypoints gt_train -arc 3,3,3,3 --tb_logs logs/fully_supervised_arc_81_train_10_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-train vp1,vp2,vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10 --subjects-test vp11,vp12'

# JOB 21115
call_ = 'python run.py --epochs 40 --checkpoint checkpoint/fully_supervised_27 --keypoints gt_train -arc 3,3,3 --tb_logs logs/fully_supervised_arc_27_train_10_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-train vp1,vp2,vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10 --subjects-test vp11,vp12'
   
# JOB 21
call_ = 'python run.py --epochs 40 --checkpoint checkpoint/fully_supervised_243 --keypoints gt_train -arc 3,3,3,3,3 --tb_logs logs/fully_supervised_arc_243_train_10_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-train vp1,vp2,vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10 --subjects-test vp11,vp12'
       
subprocess.run(call_.split(' '))