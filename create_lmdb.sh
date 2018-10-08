#!/usr/bin/env sh
# Este script realiza o treinamento da Rede definida em caffe_models.

for LOT in UFPR04 UFPR05 PUC
do
    echo "Criando LMDB"
    cd code
    python3 create_lmdb.py $LOT 7
    cd ..
    echo "LMDB criado"

    echo "Gerando a imagem média dos dados de treinamento"
    if [! -d "input/$LOT" ]
        mkdir input/$LOT
    fi
    compute_image_mean -backend=lmdb input/$LOT/train_lmdb input/$LOT/mean.binaryproto
    echo "Média gerada"

    echo "Executando o treinamento"
    if [! -d "caffe_models/$LOT" ]
        mkdir caffe_models/$LOT
    fi
    caffe train --solver caffe_models/$LOT/solver_1.prototxt 2>&1 | tee caffe_models/$LOT/model_1_train.log
    echo "Feito"

    echo "Testando a rede"
    cd code
    for i in UFPR04 UFPR05 PUC
    do
        python3 make_preditions $LOT $i
    done
    cd ..
    echo "Teste finalizado"
done