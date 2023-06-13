python solo-learn/main_linear.py --dataset imagenet100 --backbone resnet18 --data_dir ./data --train_dir imagenet100/train --val_dir imagenet100/val --max_epochs 100 --devices 0 --accelerator gpu --precision 32 --optimizer sgd --scheduler step --lr 0.1 --lr_decay_steps 60 80 --weight_decay 0 --batch_size 256 --num_workers 4 --dali --save_checkpoint \
  --pretrained_feature_extractor $1 \
  --wandb --name barlow --project $WANDB_PROJECT --entity $WANDB_ENTITY
