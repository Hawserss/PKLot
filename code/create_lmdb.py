import os
import sys
import glob
import random
import numpy as np

import datetime
from datetime import timedelta

import image

import caffe
from caffe.proto import caffe_pb2
import lmdb

def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=image.IMAGE_WIDTH,
        height=image.IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

def create_lmdb(lmdb_file, data):
    in_db = lmdb.open(lmdb_file, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, img_path in enumerate(data):
            if in_idx %  6 == 0:
                continue
            img = image.load_transform_img(img_path, img_width=image.IMAGE_WIDTH, img_height=image.IMAGE_HEIGHT)
            if 'Empty' in img_path:
                label = 0
            else:
                label = 1
            datum = make_datum(img, label)
            in_txn.put('{:0>5d}'.format(in_idx).encode(), datum.SerializeToString())
            print('{:0>5d}'.format(in_idx) + ':' + img_path)
    in_db.close()

def toDate(img_path):
    vDate = img_path.split('/')[-1][:-4].split('_')[0].split('-')
    year = int(vDate[0])
    month = int(vDate[1])
    day = int(vDate[2])
    return datetime.date(year, month, day);    

def load_data(lot, window):    
    raw_data = [img for img in glob.glob("../PKLot/PKLotSegmented/"+lot+"/**/*jpg", recursive=True)]
    # if toDate(img) >= vDate and toDate(img) <= vDateF]

    '''
    Aplicando o protocolo:
    '''
    # Obtenho as datas das imagens
    date_list = list(set([toDate(img).strftime('%Y-%m-%d') for img in raw_data]))
    date_list.sort()
    date_list = date_list[0:int(window)]
    
    # Randomizo a lista de datas para pegar as datas de maneira aleatória.
    random.shuffle(date_list)
    
    # Defino o tamanho da amostragem de treinamento
    train_range = int(len(date_list) * 0.8)
    # Filtro as datas em dias utilizados para treinamento
    train_list = date_list[0:train_range]
    # Filtro as datas em dias utilizados para validação
    test_list = date_list[train_range:]

    # Aplico o filtro das imagens para utilizar as imagens da data de alvo para treinamento.
    train_data = [img for img in raw_data if toDate(img).strftime('%Y-%m-%d') in train_list]
    # Aplico o filtro das imagens para utilizar as imagens da data de alvo para validação.
    test_data = [img for img in raw_data if toDate(img).strftime('%Y-%m-%d') in test_list]   

    # Randomizamos os dados de treinamento
    random.shuffle(train_data)

    # Randomizamos os dados de validação
    random.shuffle(test_data)

    train_lmdb = '../input/'+ lot + '/train_lmdb'
    validation_lmdb = '../input/'+ lot + '/validation_lmdb'

    os.system('rm -rf  ' + train_lmdb)
    os.system('rm -rf  ' + validation_lmdb)

    #Shuffle train_data
    random.shuffle(train_data)

    print('Creating train_lmdb')
    create_lmdb(train_lmdb, train_data)
    print('\nCreating validation_lmdb')
    create_lmdb(validation_lmdb, test_data)
    print('\nFinished processing all images')


arglot = sys.argv[1]
argwindow = int(sys.argv[2])

load_data(arglot, argwindow)
