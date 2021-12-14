# PetFinder Prediction Pawpularity\

You can Download data from https://www.kaggle.com/c/petfinder-pawpularity-score

and You Have to Resize image before training.

You can Use Various model by just adding like this.

python  main.py \
        --model vit_b_16 \
        --num_workers 12 \
        --batch_size 128 \
        --weight_decay 1e-5 \
        --num_epochs 100 \
        --lr 1e-4 \
        --dir_data ./petfinder-pawpularity-score \
        --dir_ckpt ./ckpt \
       
You Can use model such as viT_b_16, viT_b_32, Resnet(ResNet50), EfficientNetb-0, EfficientNetHybridviT Version 1, EfficientNetHybridviT Version 2, EfficientNetHybridSwinT, ensemble model(EfficientNet and viT), Ensemble model using meta data (EfficitentNet and viT)

This is model Choices that you can choose
'vit_b_16', 'vit_b_32', 'vit_L_16', 'resnet', 'swin', 'effinet', 'effinet_b5', 'use_meta', 'ensemble', 'vit_scheduler', 'hybrid', 'hybrid_2', 'hybrid_swin'
