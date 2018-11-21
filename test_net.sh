#!/usr/bin/env sh
# Este script realiza o teste de cada Rede definida em caffe_models 
# contra cada um dos repositorios de imagens.

cd code
for LOT in UFPR04 UFPR05 PUC
do
    echo "Testando a rede"
    python3 make_predictions.py $LOT
    echo "Teste finalizado"
done
cd ..
