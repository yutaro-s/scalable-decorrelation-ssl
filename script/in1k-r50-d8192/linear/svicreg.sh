python solo-learn/main_linear.py --dataset imagenet --backbone resnet50 --data_dir ./data --train_dir imagenet/train --val_dir imagenet/val --max_epochs 100 --devices 0,1,2,3,4,5,6,7 --accelerator gpu --strategy ddp --precision 32 --optimizer sgd --scheduler warmup_cosine --warmup_epochs 0 --lr 0.125 --weight_decay 1e-6 --batch_size 32 --num_workers 8 --dali --save_checkpoint \
  --pretrained_feature_extractor $1 \
  --wandb --name svicreg --project $WANDB_PROJECT --entity $WANDB_ENTITY
