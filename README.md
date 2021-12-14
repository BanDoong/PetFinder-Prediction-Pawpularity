# paw_predict

You can Use Various model by just adding like this.

python  main.py \
        --model vit_b_16 \
        --num_workers 12 \
        --batch_size 128 \
        --weight_decay 1e-5
        --num_epochs 100 \
        --lr 1e-4 \
        --dir_data ./petfinder-pawpularity-score \
        --dir_ckpt ./ckpt \
       
You Can use model such as viT_b_16, viT_b_32, Resnet(ResNet50), EfficientNetb-0, EfficientNetHybridviT Version 1, EfficientNetHybridviT Version 2, EfficientNetHybridSwinT, ensemble model(EfficientNet and viT), Ensemble model using meta data (EfficitentNet and viT)
