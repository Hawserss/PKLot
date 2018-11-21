import sys
import os

os.environ['GLOG_minloglevel'] = '3'

import caffe

def train(folder):
    caffe.set_mode_gpu() 

    solver = caffe.get_solver('caffe_models/' + folder + '/solver_1.prototxt')
    
    print("Treinando rede para " + folder + "\n")
    iterations = range(17)
    for index in iterations:
        sys.stdout.write('\r')
        solver.step(1)
        j = (index + 1)/len(iterations)
        sys.stdout.write("[%-50s] %d%%" % ('='*int(50*j), 100*j))
        sys.stdout.flush()

train(sys.argv[1])