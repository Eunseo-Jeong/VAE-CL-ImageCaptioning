#/bin/bash

latent_size=64
dd=0.7
dc=0.2
di=0.1
cuda=0
contrastive=true
cls_latent_vector=false

if [ $contrastive = true ] && [ $cls_latent_vector = true ];then
    python -u main.py \
        --result_path "dl_cl_cls_latent_$latent_size"\
        --latent_size $latent_size \
        --dd $dd \
        --dc $dc \
        --di $di \
        --cuda $cuda \
        --contrastive \
        --cls_latent_vector
fi

if [ $contrastive = false ] && [ $cls_latent_vector = true ] ; then
    python -u main.py \
        --result_path "dl_cls_latent_$latent_size"\
        --latent_size $latent_size \
        --dd $dd \
        --dc $dc \
        --di $di \
        --cuda $cuda \
        --cls_latent_vector
fi

if [ $contrastive = true ] && [ $cls_latent_vector = false ] ; then
    python -u main.py \
        --result_path "dl_cl$latent_size"\
        --latent_size $latent_size \
        --dd $dd \
        --dc $dc \
        --di $di \
        --cuda $cuda \
        --contrastive
fi

if [ $contrastive = false ] && [ $cls_latent_vector = false ] ; then
    python -u main.py \
        --result_path "dl_$latent_size"\
        --latent_size $latent_size \
        --dd $dd \
        --dc $dc \
        --di $di \
        --cuda $cuda
fi