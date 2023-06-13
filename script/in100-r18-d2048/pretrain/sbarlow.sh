python solo-learn/main_pretrain.py --dataset imagenet100 --backbone resnet18 --data_dir ./data --train_dir imagenet100/train --val_dir imagenet100/val --max_epochs 400 --devices 0,1,2,3,4,5,6,7 --accelerator gpu --strategy ddp --sync_batchnorm --num_workers 32 --precision 32 --optimizer lars --grad_clip_lars --eta_lars 0.02 --exclude_bias_n_norm --scheduler warmup_cosine --lr 0.3 --weight_decay 0.0001 --batch_size 32 --dali --brightness 0.4 --contrast 0.4 --saturation 0.2 --hue 0.1 --gaussian_prob 1 0.1 --solarization_prob 0 0.2 --num_crops_per_aug 1 1 --save_checkpoint --method sbarlow_twins --proj_hidden_dim 2048 --proj_output_dim 2048 --group_size 2048 --rand_type batch --exponent 2 --lamb 0.0009765625 --scale_loss 0.125 \
  --wandb --name sbarlow --project $WANDB_PROJECT --entity $WANDB_ENTITY
