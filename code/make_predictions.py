import os
import sys
import glob

os.environ['GLOG_minloglevel'] = '3'

import caffe
import image
import lmdb
import datetime
import operator

import numpy as np
import matplotlib.pyplot as plt

from caffe.proto import caffe_pb2
from result import Result

def toDate(img_path):
    vDate = img_path.split('/')[-1][:-4].split('_')[0].split('-')
    year = int(vDate[0])
    month = int(vDate[1])
    day = int(vDate[2])
    return datetime.date(year, month, day)

def first(iterable, default=None):
    for item in iterable:
        return item
    return default
'''
Making predicitions
'''
def makePredictions(modelDir, sourceDir):
    caffe.set_mode_gpu() 

    '''
    Reading mean image, caffe model and its weights 
    '''
    #Read mean image
    mean_blob = caffe_pb2.BlobProto()
    with open('../input/' + modelDir + '/mean.binaryproto', 'rb') as f:
        mean_blob.ParseFromString(f.read())
    mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
        (mean_blob.channels, mean_blob.height, mean_blob.width))
    
    for i in range(5000, 45000, 5000):
        #Read model architecture and trained model's weights
        net = caffe.Net('../caffe_models/' + modelDir + '/caffenet_deploy_1.prototxt',
                        '../caffe_models/' + modelDir + '/snapshot_iter_'+ str(i) +'.caffemodel',
                        caffe.TEST)

        #Define image transformers
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_mean('data', mean_array)
        transformer.set_transpose('data', (2,0,1))

        #Reading image paths
        test_img_paths = [img_path for img_path in glob.glob("../PKLot/PKLotSegmented/" + sourceDir + "/**/*.jpg", recursive=True)]
        date_list = list(set([toDate(img) for img in test_img_paths]))
        date_list.sort()

        #Making predictions
        results = [Result(date) for date in date_list]
        results.sort(key=operator.attrgetter('date'))

        P = 0
        N = 0

        print("Testando rede com modelo treinado " + str(i) + " vezes\n")
        for index, img_path in enumerate(test_img_paths):
            sys.stdout.write('\r')
            img = image.load_transform_img(img_path, img_width=image.IMAGE_WIDTH, img_height=image.IMAGE_HEIGHT)
            date = toDate(img_path)

            rAux = first(x for x in results if x.date == date)
            
            net.blobs['data'].data[...] = transformer.preprocess('data', img)
            out = net.forward()
            pred_probas = out['prob']

            result = pred_probas.argmax()
            
            rAux.setValue(result, (0, 1)['Occupied' in img_path])
            
            if result == 1:
                P+=1
            else:
                N+=1
            
            j = (index + 1)/len(test_img_paths)
            sys.stdout.write("[%-50s] %d%%" % ('='*int(50*j), 100*j))
            sys.stdout.flush()
        
        print('\n')
            
        trueNegatives = 0
        truePositives = 0
        falseNegatives = 0
        falsePositives = 0
        
        for res in results:
            trueNegatives+=res.trueNegative
            truePositives+=res.truePositive
            falseNegatives+=res.falseNegative
            falsePositives+=res.falsePositive
        
        with open("../caffe_models/" + modelDir + "/results_matrix_" + str(i) + "_" + sourceDir + ".result", "w") as f:
            f.write('Empty:    |' + str(trueNegatives) + ' | ' + str(falseNegatives) + '\n')
            f.write('Occupied: |' + str(truePositives) + ' | ' + str(falsePositives) + '\n')
            f.write('TA:       |' + str((truePositives+trueNegatives)/(P+N)) + '\n')
            f.write('TPR:      |' + str(truePositives/P)+ '\n')
            f.write('TNR:      |' + str(trueNegatives/N)+ '\n')
        f.close()

        x = [index for index, res in enumerate(results)]
        y = [res.getTA() for res in results]
        dpi = 80
        plt.title('Modelo '+ str(i) + ' de: ' + results[0].date.strftime('%Y-%m-%d') + ' até ' + results[-1].date.strftime('%Y-%m-%d'))
        plt.figure(figsize=(1000/dpi,500/dpi))
        plt.plot(x, y)
        figurePath = "../caffe_models/" + modelDir + "/results_graph_" + str(i) + "_" + sourceDir + ".png"
        plt.savefig(figurePath, dpi=dpi, bbox_inches='tight')

makePredictions(sys.argv[1], sys.argv[2])
