python train.py --save_dir vox2dev_80mel_ResNet34StatsPool-34-128_ArcFace-32-0.2 \
	  --data_name train_vox2_dev_speed --dur_range 2 4 \
		--val_data_name vox_test --val_dur_range 8 8 \
	  --batch_size 256 --workers 16 \
	  --mels 80 --fft 512 \
		--model ResNet34StatsPool --in_planes 34 --embd_dim 128 \
		--classifier ArcFace --angular_m 0.2 --angular_s 32 --dropout 0 \
		--gpu 4,5,6 --epochs 160 --start_epoch 101  --lr 0.001 &
