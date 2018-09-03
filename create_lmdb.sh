#!/usr/bin/env sh
# Este script realiza o treinamento da Rede definida em caffe_models.
echo "Criando LMDB"
cd code
python3 create_lmdb.py 2012 12 7 10
cd ..
echo "LMDB criado"

echo "Gerando a imagem média dos dados de treinamento"
compute_image_mean -backend=lmdb input/train_lmdb input/mean.binaryproto
echo "Média gerada"

echo "Executando o treinamento"
caffe train --solver caffe_models/caffe_model_1/solver_1.prototxt 2>&1 | tee caffe_models/caffe_model_1/model_1_train.log
echo "Feito"