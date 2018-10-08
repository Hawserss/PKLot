#!/usr/bin/env sh
# Este script baixa e descompacta o PKLot data set para uso.
SOURCES=sources
INPUT=input

if [ ! -d "$SOURCES" ]; then 
    echo "Diretório não encontrado, criando diretório /${SOURCES}"
    mkdir ${SOURCES}
fi

if [ ! -d "$INPUT" ]; then 
    echo "Diretório não encontrado, criando diretório /${INPUT}"
    mkdir ${INPUT}
fi

if [ ! -f "$SOURCES/PKLot.tar.gz" ]; then 
    echo "Baixando PKLot..."
    wget http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz -O ${SOURCES}/PKLot.tar.gz
fi

if [ ! -d "PKLot" ]; then 
    echo "Descompactando..."
    tar xvzf ${SOURCES}/PKLot.tar.gz
fi

echo "Feito."