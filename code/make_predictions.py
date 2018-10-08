import os
import sys
import glob
import caffe
import image
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

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

    #Making predictions
    truePositives = 0
    trueNegatives = 0
    falsePositives = 0
    falseNegatives = 0

    for img_path in test_img_paths:
        img = image.load_transform_img(img_path, img_width=image.IMAGE_WIDTH, img_height=image.IMAGE_HEIGHT)

        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        out = net.forward()
        pred_probas = out['prob']

        result = pred_probas.argmax()
        label = 0
        if 'Empty' in img_path:
            label = 0
            if result == label:
                trueNegatives += 1
            else:
                falseNegatives += 1
        else:
            label = 1
            if result == label:
                truePositives += 1
            else:
                falsePositives += 1
        
    P = truePositives+falsePositives
    N = trueNegatives+falseNegatives
    with open("../caffe_models/" + modelDir + "/results_matrix_" + sourceDir + ".txt", "w") as f:
        f.write('Empty:    |' + str(trueNegatives) + ' | ' + str(falseNegatives) + '\n')
        f.write('Occupied: |' + str(truePositives) + ' | ' + str(falsePositives) + '\n')
        f.write('TA:       |' + str((truePositives+trueNegatives)/(P+N)) + '\n')
        f.write('TPR:      |' + str(truePositives/P)+ '\n')
        f.write('TNR:      |' + str(trueNegatives/N)+ '\n')

    f.close()

makePredictions(sys.argv[1], sys.argv[2])
