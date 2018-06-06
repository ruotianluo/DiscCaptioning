"""
Evaluate on a collection
large seq_per_img 
batch size is 1
closest.pkl contrains non-distractor
distractor_proportion is 1.
Print the vote from distractor_mlp
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import argparse
import misc.utils as utils
import torch
import torch.nn as nn
import eval_utils

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
# Basic options
parser.add_argument('--batch_size', type=int, default=0,
                help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=0,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=1,
                help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--beam_size', type=int, default=2,
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--temperature', type=float, default=1.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='', 
                help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='', 
                help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_fc_dir', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_att_dir', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_label_h5', type=str, default='',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='', 
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='test', 
                help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='', 
                help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
# misc
parser.add_argument('--id', type=str, default='', 
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')

parser.add_argument('--seq_per_img', type=int, default=5, 
                help='')
parser.add_argument('--closest_num', type=int, default=5, 
                help='')
parser.add_argument('--closest_file', type=str, default='data/collections.pkl',
                help='Closest_file')

parser.add_argument('--decoding_constraint', type=int, default=0,
                help='1 if not allowing decoding two same words in a row')

parser.add_argument('--initialize_retrieval', type=str, default=None,
                help="""xxxx.pth""")

# vse
parser.add_argument('--vse_model', type=str, default=None,
                help='fc, None')
parser.add_argument('--vse_rnn_type', type=str, default=None,
                help='rnn, gru, or lstm')
parser.add_argument('--vse_margin', default=None, type=float,
                help='Rank loss margin; when margin is -1, it means use binary cross entropy (usually works with MLP).')
parser.add_argument('--vse_embed_size', default=None, type=int,
                help='Dimensionality of the joint embedding.')
parser.add_argument('--vse_num_layers', default=None, type=int,
                help='Number of GRU layers.')
parser.add_argument('--vse_max_violation', default=None, type=int,
                help='Use max instead of sum in the rank loss.')
parser.add_argument('--vse_measure', default='cosine',
                help='Similarity measure used (cosine|order|MLP)')
parser.add_argument('--vse_use_abs', default=None, type=int,
                help='Take the absolute value of embedding vectors.')
parser.add_argument('--vse_no_imgnorm', default=None, type=int,
                help='Do not normalize the image embeddings.')
parser.add_argument('--vse_loss_type', default=None, type=str,
                help='contrastive or pair')
parser.add_argument('--vse_pool_type', default=None, type=str,
                help='last, mean, max')

parser.add_argument('--fold5', default=0, type=int,
                help='fold5')

opt = parser.parse_args()

np.random.seed(123)

# Load infos
with open(opt.infos_path) as f:
    infos = cPickle.load(f)

# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_dir = infos['opt'].input_fc_dir
    opt.input_att_dir = infos['opt'].input_att_dir
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
    opt.id = infos['opt'].id
ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval",
          "input_fc_dir", "input_att_dir", "input_label_h5", 'seq_per_img', 'closest_num', 'closest_file']
for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt) and getattr(opt, k) is not None:
            assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

assert opt.seq_per_img == 5

opt.vse_loss_weight = vars(opt).get('vse_loss_weight', 1)
opt.caption_loss_weight = vars(opt).get('caption_loss_weight', 1)

# Setup the model
model = models.JointModel(opt)
utils.load_state_dict(model, torch.load(opt.model))
if opt.initialize_retrieval is not None:
    print("Make sure the vse opt are the same !!!!!\n"*100)
    utils.load_state_dict(model, {k:v for k,v in torch.load(opt.initialize_retrieval).items() if 'vse' in k})
model.cuda()
model.eval()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
  loader = DataLoader(opt)
else:
  loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
  loader.ix_to_word = infos['vocab']

opt.id = opt.id + '_retrieval'

result = eval_utils.evalrank(model, loader, vars(opt))

json.dump(result, open('eval_results/'+opt.id+'_retreival_results.json', 'w'))


