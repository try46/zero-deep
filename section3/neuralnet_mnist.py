#coding: utf-8

import sys,os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common,funcitons import sigmoid, softmax