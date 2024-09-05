python -m torch.distributed.launch --nproc_per_node=8 --master_port 47769 --use_env main.py --model fdvit_s --opt adamp --batch-size 256 --data-path /group/modelzoo/test_dataset/imagenet_torchvision/ --output_dir ./output/fdvit_s/ --epochs 300 --warmup-epochs 20 --ratio 0.03 --mask_thre 0.2 --resume ./output/fdvit_s/checkpoint.pth


nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port 47769 --use_env main.py --model pit_l --setting 15.3 --opt adamp --batch-size 64 --data-path /group/modelzoo/test_dataset/imagenet_torchvision/ --output_dir /group/dphi_algo_scratch_14/yixingx/code/pit/output/pit_l_15_3/ --epochs 300 --warmup-epochs 10 --ratio 0.03 --mask_thre 0.2 > output/pit_l_15_3/myout.txt 2>&1 &
