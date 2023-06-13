python solo-learn/main_pretrain.py --dataset imagenet --backbone resnet50 --data_dir ./data --train_dir imagenet/train --max_epochs 1000 --devices 0,1,2,3,4,5,6,7 --accelerator gpu --strategy ddp --sync_batchnorm --precision 32 --optimizer lars --eta_lars 0.001 --exclude_bias_n_norm --scheduler warmup_cosine --lr 0.25 --weight_decay 1e-6 --batch_size 128 --num_workers 4 --dali --brightness 0.4 --contrast 0.4 --saturation 0.2 --hue 0.1 --gaussian_prob 1 0.1 --solarization_prob 0 0.2 --num_crops_per_aug 1 1 --save_checkpoint --method svicreg --proj_hidden_dim 8192 --proj_output_dim 8192 --sim_loss_weight 2.5 --var_loss_weight 2.5 --cov_loss_weight 0.1 --group_size 8192 --rand_type batch --exponent 1 \
  --wandb --name svicreg --project $WANDB_PROJECT --entity $WANDB_ENTITY
