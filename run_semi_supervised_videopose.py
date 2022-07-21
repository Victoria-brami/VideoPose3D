import subprocess

# JOB 81
call_ = 'python run.py --causal --batch-size 64 --epochs 200 -lrd 0.98 --warmup 40 --subjects-train vp1,vp2,vp3,vp4 --subset 0.5 --subjects-unlabeled vp5,vp6,vp7,vp8,vp9,vp10 --checkpoint checkpoint/semi_supervised_81/warm_40_subset_05_lab_vp1_vp2 --keypoints gt_train -arc 3,3,3,3 --tb_logs logs/causal/semi_supervised_arc_81_subset_05_warm_40_lab_train_2_unlab_8_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-test vp11,vp12'

# JOB 27
call_ = 'python run.py --causal --batch-size 64 --epochs 200 -lrd 0.98 --warmup 10 --subjects-train vp1,vp2,vp3,vp4 --subset 0.5 --subjects-unlabeled vp5,vp6,vp7,vp8,vp9,vp10 --checkpoint checkpoint/semi_supervised_27/warm_10_subset_05_vp1234 --keypoints gt_train -arc 3,3,3 --tb_logs logs/causal/semi_supervised_arc_27_subset_05_warm_10_lab_train_4_unlab_6_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-test vp11,vp12'
   
# JOB 243
# call_ = 'python run.py --causal --batch-size 64 --epochs 200 -lrd 0.98 --warmup 40 --subjects-train vp1,vp2,vp3,vp4 --subset 0.5 --subjects-unlabeled vp5,vp6,vp7,vp8,vp9,vp10 --checkpoint checkpoint/semi_supervised_243/warm_40_subset_05_lab_vp1_vp2 --keypoints gt_train -arc 3,3,3,3,3 --tb_logs logs/causal/semi_supervised_arc_243_subset_05_warm_40_lab_train_2_unlab_8_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-test vp11,vp12'
       
subprocess.run(call_.split(' '))