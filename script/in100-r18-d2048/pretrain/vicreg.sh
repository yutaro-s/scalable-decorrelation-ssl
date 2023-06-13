python solo-learn/main_pretrain.py --dataset imagenet100 --backbone resnet18 --data_dir ./data --train_dir imagenet100/train --val_dir imagenet100/val --max_epochs 400 --devices 0,1,2,3,4,5,6,7 --accelerator gpu --strategy ddp --sync_batchnorm --precision 32 --optimizer lars --grad_clip_lars --eta_lars 0.02 --exclude_bias_n_norm --scheduler warmup_cosine --lr 0.3 --weight_decay 1e-4 --batch_size 32 --num_workers 32 --dali --min_scale 0.2 --brightness 0.4 --contrast 0.4 --saturation 0.2 --hue 0.1 --solarization_prob 0.1 --num_crops_per_aug 2 --save_checkpoint --method vicreg --proj_hidden_dim 2048 --proj_output_dim 2048 --sim_loss_weight 25.0 --var_loss_weight 25.0 --cov_loss_weight 1.0 \
  --wandb --name vicreg --project $WANDB_PROJECT --entity $WANDB_ENTITY
