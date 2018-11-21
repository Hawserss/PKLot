#!/usr/bin/env sh
# Este script realiza o treinamento da Rede definida em caffe_models.

for LOT in UFPR04 UFPR05 PUC
do
    if [ -f "caffe_models/$LOT/snapshot_iter_40000.caffemodel" ]; then
        echo "Caffemodel existe"
    else
        if [! -d "input/$LOT" ]; then
            mkdir input/$LOT
            mkdir input/$LOT/train_lmdb
            mkdir input/$LOT/validation_lmdb
        fi
        echo "Criando LMDB"
        python3 code/create_lmdb.py $LOT 7
        echo "LMDB criado"

        echo "Gerando a imagem média dos dados de treinamento"        
        compute_image_mean -backend=lmdb input/$LOT/train_lmdb input/$LOT/mean.binaryproto
        echo "Média gerada"

        echo "Executando o treinamento"
        caffe train --solver caffe_models/$LOT/solver_1.prototxt 2>&1 | tee caffe_models/$LOT/model_1_train.log
        echo "Feito"
    fi
done