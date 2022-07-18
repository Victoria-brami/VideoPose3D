import subprocess

# JOB 21059
#call_ = 'python run.py --epochs 200 -lrd 0.98 --warmup 5 --subjects-train vp1,vp2 --subset 0.5 --subjects-unlabeled vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10 --checkpoint checkpoint/semi_supervised_81/subset_05_labvp1_vp2 --keypoints gt_train -arc 3,3,3,3 --tb_logs logs/semi_supervised_arc_81_subset_05_lab_train_2_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-test vp11,vp12'

# JOB 21115
call_ = 'python run.py --epochs 200 -lrd 0.98 --warmup 5 --subjects-train vp1,vp2 --subset 0.5 --subjects-unlabeled vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10 --checkpoint checkpoint/semi_supervised_27/subset_05_labvp1_vp2 --keypoints gt_train -arc 3,3,3 --tb_logs logs/semi_supervised_arc_27_subset_05_lab_train_2_unlab_tr8_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-test vp11,vp12'
   
# JOB 21
# call_ = 'python run.py --epochs 200 -lrd 0.98 --warmup 5 --subjects-train vp1,vp2 --subset 0.5 --subjects-unlabeled vp3,vp4,vp5,vp6,vp7,vp8,vp9,vp10 --checkpoint checkpoint/semi_supervised_243/subset_05_labvp1_vp2 --keypoints gt_train -arc 3,3,3,3,3 --tb_logs logs/semi_supervised_arc_243_subset_05_lab_train_2_unlab_tr8_val_2_length_10000_start_2000_pad_2000 --dataset dad --seq_start 2000 --seq_length 10000 --pad 2000 --export-training-curves --subjects-test vp11,vp12'
       
subprocess.run(call_.split(' '))