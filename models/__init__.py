from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import misc.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

from .ShowTellModel import ShowTellModel
from .FCModel import FCModel
from .CaptionModel import ShowAttendTellModel, AllImgModel
from .Att2inModel import Att2inModel
from .AttModel import *

from .VSEFCModel import VSEFCModel
from .VSEAttModel import VSEAttModel

__all__ = ['setup', 'load', 'JointModel']

def setup(opt, model_name, caption = True):
    if caption:
        if model_name == 'show_tell':
            model = ShowTellModel(opt)
        elif model_name == 'show_attend_tell':
            model = ShowAttendTellModel(opt)
        # img is concatenated with word embedding at every time step as the input of lstm
        elif model_name == 'all_img':
            model = AllImgModel(opt)
        # FC model in self-critical
        elif model_name == 'fc':
            model = FCModel(opt)
        elif model_name == 'fc2':
            model = FC2Model(opt)
        # Att2in model in self-critical
        elif model_name == 'att2in':
            model = Att2inModel(opt)
        # Att2in model with two-layer MLP img embedding and word embedding
        elif model_name == 'att2in2':
            model = Att2in2Model(opt)
        # Adaptive Attention model from Knowing when to look
        elif model_name == 'adaatt':
            model = AdaAttModel(opt)
        # Adaptive Attention with maxout lstm
        elif model_name == 'adaattmo':
            model = AdaAttMOModel(opt)
        # Top-down attention model
        elif model_name == 'topdown':
            model = TopDownModel(opt)
        else:
            raise Exception("Caption model not supported: {}".format(model_name))
    else:
        if model_name == 'fc':
            model = VSEFCModel(opt)
        elif model_name == 'dual_att':
            model = VSEAttModel(opt)
        else:
            raise Exception("VSE model not supported: {}".format(model_name))

    return model

def load(model, opt):
    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        utils.load_state_dict(model, torch.load(os.path.join(opt.start_from, 'model.pth')))


from .JointModel import *