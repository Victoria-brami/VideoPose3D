import subprocess

# JOB 21115
call_ = 'python run.py --epochs 80 --checkpoint checkpoint/fully_supervised_27 --keypoints gt_train -arc 3,3,3 --tb_logs logs/fully_supervised_arc_27_train_10_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-train vp1,vp2,vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10 --subjects-test vp11,vp12'
   
# JOB 21
# call_ = 'python run.py --epochs 80 --checkpoint checkpoint/fully_supervised_243 --keypoints gt_train -arc 3,3,3,3,3 --tb_logs logs/fully_supervised_arc_243_train_10_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-train vp1,vp2,vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10 --subjects-test vp11,vp12'

# JOB 21059
# call_ = 'python run.py --epochs 80 --keypoints gt_train -arc 3,3,3,3 --tb_logs logs/fully_supervised_arc_81_train_10_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-train vp1,vp2,vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10 --subjects-test vp11,vp12'
       
       
# JOB 21115
call_ = 'python run.py --causal --epochs 80 --checkpoint checkpoint/fully_supervised/fully_supervised_3x5x3/ --keypoints gt_train -arc 3,5,3 --tb_logs logs/causal/fully_supervised_arc_3x5x3_train_8_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-train vp1,vp2,vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10 --subjects-test vp11,vp12'
   
# JOB 21
# call_ = 'python run.py --causal --epochs 80 --checkpoint checkpoint/fully_supervised/fully_supervised_243/ --keypoints gt_train -arc 3,3,3,3,3 --tb_logs logs/causal/fully_supervised_arc_243_train_8_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-train vp1,vp2,vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10 --subjects-test vp11,vp12'

# JOB 21059
call_ = 'python run.py --causal --epochs 80 --checkpoint checkpoint/fully_supervised/fully_supervised_3x7x5x3/ --keypoints gt_train -arc 3,7,5,3 --tb_logs logs/causal/fully_supervised_arc_3x7x5x3_train_8_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-train vp1,vp2,vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10 --subjects-test vp11,vp12'
call_ = 'python run.py --causal --epochs 80 --checkpoint checkpoint/fully_supervised/fully_supervised_3x3x3/ --keypoints gt_train -arc 3,3,3 --tb_logs logs/causal/tuned_loss_fully_supervised_arc_3x3x3_train_8_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-train vp1,vp2,vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10 --subjects-test vp11,vp12'

# call_ = 'python run.py --causal --epochs 80 --checkpoint checkpoint/fully_supervised/fully_supervised_3x3x5/ --keypoints gt_train -arc 3,3,5 --tb_logs logs/causal/fully_supervised_arc_3x3x5_train_8_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-train vp1,vp2,vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10 --subjects-test vp11,vp12'
call_ = 'python run.py --causal --lambda_angle 1. --lambda_sym 0.1 --epochs 80 --checkpoint checkpoint/fully_supervised/fully_supervised_3x3x3/ --keypoints gt_train -arc 3,3,3 --tb_logs logs/causal/tuned_loss_sym_01_fully_supervised_arc_3x3x3_train_8_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-train vp1,vp2,vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10 --subjects-test vp11,vp12'
# call_ = 'python run.py --causal --epochs 80 --checkpoint checkpoint/debug --keypoints gt_train -arc 3,3,3 --tb_logs logs/causal/debug --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-train vp1,vp2,vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10 --subjects-test vp11,vp12'
call_ = 'python run.py --causal --lambda_angle 10. --lambda_sym 0.1 --epochs 80 --checkpoint checkpoint/fully_supervised/fully_supervised_3x3x3_sym_01_angle_10/ --keypoints gt_train -arc 3,3,3 --tb_logs logs/causal/loss_sym_01_angle_10_fully_supervised_arc_3x3x3_train_8_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-train vp1,vp2,vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10 --subjects-test vp11,vp12'
call_ = 'python run.py --causal --no-illegal-angle --lambda_sym 0.001 --epochs 80 --checkpoint checkpoint/fully_supervised/fully_supervised_3x3x3_sym_0001/ --keypoints gt_train -arc 3,3,3 --tb_logs logs/causal/tuned_loss_sym_0001_fully_supervised_arc_3x3x3_train_8_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-train vp1,vp2,vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10 --subjects-test vp11,vp12 --checkpoint-frequency 80'


      
subprocess.run(call_.split(' '))