#coding: utf-8

import sys,os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

def get_data():
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test,t_test

def init_network():
    with open("sample_weight.pkl",'rb') as f:
        nerwork = pickle.load(f)
    return nerwork

def predict(network, x):
    w1,w2.w3 = network['w1'],network['w2'],network['w3']
