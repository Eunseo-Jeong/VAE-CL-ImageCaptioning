#/bin/bash

LATENT_SIZE=(64 128 768)
dd=0.9
dc=0.0
di=0.1
cuda=0
contrastive=false
CLS_LATENT_VECTOR=(false true)
learning_rate=(1e-04)

for latent_size in ${LATENT_SIZE[@]}; do
    for cls_latent_vector in ${CLS_LATENT_VECTOR[@]}; do
        if [ $contrastive = false ] && [ $cls_latent_vector = true ];then
            python -u main.py \
                --result_path "baseline_$latent_size$cls_latent_vector$learning_rate"\
                --latent_size $latent_size \
                --learning_rate $learning_rate \
                --dd $dd \
                --dc $dc \
                --di $di \
                --cuda $cuda \
                # --contrastive \
                --cls_latent_vector
        fi

        if [ $contrastive = false ] && [ $cls_latent_vector = false ] ; then
            python -u main.py \
                --result_path "baseline_$latent_size$cls_latent_vector$learning_rate"\
                --latent_size $latent_size \
                --learning_rate $learning_rate \
                --dd $dd \
                --dc $dc \
                --di $di \
                --cuda $cuda 
                # --contrastive $contrastive 
        fi
    done
done