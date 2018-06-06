from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import collections
import torch
import torch.nn as nn
from torch.autograd import Variable

import skimage
import skimage.io
from scipy.misc import imresize
import skimage.transform

import torchvision

import numpy as np
import json
from PIL import Image, ImageFont, ImageDraw

def if_use_att(opt):
    # Decide if load attention feature according to caption model
    if opt.caption_model in ['show_tell', 'all_img', 'fc'] and opt.vse_model in ['fc', 'fc2']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def load_image(file_name, size = None):
    img = skimage.io.imread('/mnt/ilcompf8d0/user/rluo/datasets/coco/' + file_name)

    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
        img = np.concatenate((img, img, img), axis=2)

    if size:
        img = imresize(img, size)

    img = img.astype('float32') / 255.0
    img = torch.from_numpy(img.transpose([2,0,1]))

    return img

def make_html(id, iteration):
    output = {}
    output['main_id'] = logger.to_htmls.keys()
    
    output['img_urls'] = []
    output['img_urls2'] = []
    output['captions'] = []
    for i in output['main_id']:
        output['img_urls'].append(['../../datasets/coco/'+_[0] for _ in logger.to_htmls[i]])
        output['img_urls2'].append(['../../datasets/coco/'+_[1] for _ in logger.to_htmls[i]])
        output['captions'].append([_[2] + '\n' + _[3] for _ in logger.to_htmls[i]])

    if not os.path.isdir('htmls_'+id):
        os.mkdir('htmls_'+id)
        os.system('cp htmls/index.html htmls_'+id+'/')
    json.dump(output, open('htmls_'+id+'/result'+str(iteration)+'.json', 'w'))

def var_wrapper(x, cuda=True, volatile=False):
    if type(x) is dict:
        return {k: var_wrapper(v, cuda, volatile) for k,v in x.items()}
    if type(x) is list or type(x) is tuple:
        return [var_wrapper(v, cuda, volatile) for v in x]
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if cuda:
        x = x.cuda()
    else:
        x = x.cpu()
    if torch.is_tensor(x):
        x = Variable(x, volatile=volatile)
    if isinstance(x, Variable) and volatile!=x.volatile:
        x = Variable(x.data, volatile=volatile)
    return x

def load_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    keys = set(model_state_dict.keys() + state_dict.keys())
    for k in keys:
        if k not in state_dict:
            print('key %s in model.state_dict() not in loaded state_dict' %(k))
        elif k not in model_state_dict:
            print('key %s in loaded state_dict not in model.state_dict()' %(k))
        else:
            if state_dict[k].size() != model_state_dict[k].size():
                print('key %s size not match in model.state_dict() and loaded state_dict. Try to flatten and copy the values in common parts' %(k))
            model_state_dict[k].view(-1)[:min(model_state_dict[k].numel(), state_dict[k].numel())]\
                .copy_(state_dict[k].view(-1)[:min(model_state_dict[k].numel(), state_dict[k].numel())])

    model.load_state_dict(model_state_dict)