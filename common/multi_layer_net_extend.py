#coding : utf-8

import sys,os
sys.path.append(os.pardir)
import numpy as np
from collection import OrderdDict
from common.layers import *
from common.gradient import numerical_gradient

class MultiLayerNetExtend:
    def __init__(self,input_size,hidden_size_list,output_size,
                 activation='relu',weight_init_std='relu',weight_decay_lambda=0,
                 use_dropout= False,dropout_ration=0.5,use_batchnorm=False):
        self.input_size = input_size
        self.output_size = output_size


