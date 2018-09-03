import os
import glob
import caffe
import image
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

'''
Making predicitions
'''
def makePredictions(sourceDir):
    caffe.set_mode_gpu() 

    '''
    Reading mean image, caffe model and its weights 
    '''
    #Read mean image
    mean_blob = caffe_pb2.BlobProto()
    with open('../input/mean.binaryproto', 'rb') as f:
        mean_blob.ParseFromString(f.read())
    mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
        (mean_blob.channels, mean_blob.height, mean_blob.width))


    #Read model architecture and trained model's weights
    net = caffe.Net('../caffe_models/caffe_model_1/caffenet_deploy_1.prototxt',
                    '../caffe_models/caffe_model_1/snapshot_iter_30000.caffemodel',
                    caffe.TEST)

    #Define image transformers
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', mean_array)
    transformer.set_transpose('data', (2,0,1))

    #Reading image paths
    test_img_paths = [img_path for img_path in glob.glob("../PKLot/PKLotSegmented/" + sourceDir + "/**/*.jpg", recursive=True)]

    #Making predictions
    test_ids = []
    labels = []
    preds = []
    hits = []
    for img_path in test_img_paths:
        img = image.load_transform_img(img_path, img_width=image.IMAGE_WIDTH, img_height=image.IMAGE_HEIGHT)

        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        out = net.forward()
        pred_probas = out['prob']

        label = 0
        if 'Empty' in img_path:
            label = 0
        else:
            label = 1
        labels = labels + [label]
        test_ids = test_ids + [img_path]
        preds = preds + [pred_probas.argmax()]
        if label == pred_probas.argmax():
            hits = hits + [True] 
        else:
            hits = hits + [False]
    
    with open("../caffe_models/caffe_model_1/results_" + sourceDir + ".csv","w") as f:
        true_positives = sum(1 for item in hits if item)
        f.write('Acurácia: ' + str(true_positives/len(hits))+ '\n')
        f.write("id, label, predicted_label\n")
        for i in range(len(test_ids)):
            f.write(str(test_ids[i])+", "+str(labels[i])+", "+str(preds[i])+"\n")
        
    f.close()