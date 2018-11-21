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
import matplotlib.dates as mdates

from caffe.proto import caffe_pb2
from result import Result

def toDate(img_path):
    vfDate = img_path.split('/')[-1][:-4].split('#')[0]
    vDate = vfDate.replace('_', '-').split('-')
    year = int(vDate[0])
    month = int(vDate[1])
    day = int(vDate[2])
    hour = int(vDate[3])
    minute = int(vDate[4])
    second = int(vDate[5])
    return datetime.datetime(year, month, day, hour, minute, second)

def first(iterable, default=None):
    for item in iterable:
        return item
    return default

'''
Making predicitions
'''
def makePredictions(modelDir):
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

    results = []

    for ind, sourceDir in enumerate(['UFPR04', 'UFPR05', 'PUC']):
        #Read model architecture and trained model's weights
        net = caffe.Net('../caffe_models/' + modelDir + '/caffenet_deploy_1.prototxt',
                        '../caffe_models/' + modelDir + '/snapshot_iter_40000.caffemodel',
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

        print("Testando rede com modelo treinado 40000 vezes\n")
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
        
        with open("../caffe_models/" + modelDir + "/results_matrix_" + sourceDir + ".result", "w") as f:
            f.write('Empty:    |' + str(trueNegatives) + ' | ' + str(falseNegatives) + '\n')
            f.write('Occupied: |' + str(truePositives) + ' | ' + str(falsePositives) + '\n')
            f.write('TA:       |' + str((truePositives+trueNegatives)/(P+N)) + '\n')
            f.write('TPR:      |' + str(truePositives/P)+ '\n')
            f.write('TNR:      |' + str(trueNegatives/N)+ '\n')
        f.close()

        results[ind] = [[res for index, res in enumerate(results)], [res.getTA() for res in results]]
    
    fig, ax = plt.subplots(1,1)
    for ind in results:
        label = 'UFPR04'
        if ind == 1:
            label = 'UFPR05'
        else:
            if ind == 2:
                label = 'PUC'

        ax.plot(ind[0], ind[1], label=label)

    months = mdates.YearLocator()   # every year
    days = mdates.MonthLocator()    # every month
    yearsFmt = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(days)

    ax.format_xdata = yearsFmt
    ax.grid(True)

    fig.autofmt_xdate()

    plt.show()
    figurePath = "../caffe_models/" + modelDir + "/results_graph_" + sourceDir + ".png"
    plt.savefig(figurePath, bbox_inches='tight')

makePredictions(sys.argv[1])
