from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.norm(X, dim=1, keepdim=True) + 1e-7
    X = torch.div(X, norm)
    return X

class EncoderImage(nn.Module):

    def __init__(self, opt):
        super(EncoderImage, self).__init__()
        self.embed_size = opt.vse_embed_size
        self.no_imgnorm = opt.vse_no_imgnorm
        self.use_abs = opt.vse_use_abs
        self.fc_feat_size = opt.fc_feat_size

        self.fc = nn.Linear(self.fc_feat_size, self.embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, opt):
        super(EncoderText, self).__init__()
        self.use_abs = opt.vse_use_abs
        self.input_encoding_size = opt.input_encoding_size
        self.embed_size = opt.vse_embed_size
        self.num_layers = opt.vse_num_layers
        self.rnn_type = opt.vse_rnn_type
        self.vocab_size = opt.vocab_size
        self.use_abs = opt.vse_use_abs
        self.pool_type = getattr(opt, 'vse_pool_type', '')
        # word embedding
        self.embed = nn.Embedding(self.vocab_size + 2, self.input_encoding_size)

        # caption embedding
        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size, self.embed_size, self.num_layers, batch_first=True)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def pad_sentences(self, seqs, masks):
        len_sents = (masks > 0).long().sum(1)
        len_sents, len_ix = len_sents.sort(0, descending=True)

        inv_ix = len_ix.clone()
        inv_ix.data[len_ix.data] = torch.arange(0, len(len_ix)).type_as(inv_ix.data)

        new_seqs = seqs[len_ix].contiguous()

        return new_seqs, len_sents, len_ix, inv_ix

    def forward(self, seqs, masks):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        padded_seqs, sorted_lens, len_ix, inv_ix = self.pad_sentences(seqs, masks)

        if seqs.dim() > 2:
            seqs_embed = torch.matmul(padded_seqs, self.embed.weight) # one hot input
        else:
            seqs_embed = self.embed(padded_seqs)

        seqs_pack = pack_padded_sequence(seqs_embed, list(sorted_lens.data), batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(seqs_pack)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = sorted_lens.view(-1,1,1).expand(seqs.size(0), 1, self.embed_size) - 1
        if self.pool_type == 'mean':
            out = padded[0]
            _masks = masks[len_ix].float()
            out = (out * _masks[:,:out.size(1)].unsqueeze(-1)).sum(1) / _masks.sum(1, keepdim=True)
        elif self.pool_type == 'max':
            out = padded[0]
            _masks = masks[len_ix][:,:out.size(1)].float()
            out = (out * _masks.unsqueeze(-1) + (_masks == 0).unsqueeze(-1).float() * -1e10).max(1)[0]
        else:
            out = padded[0].gather(1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        out = out[inv_ix].contiguous()

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).squeeze(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt):
        super(ContrastiveLoss, self).__init__()
        self.margin = opt.vse_margin
        self.measure = opt.vse_measure
        if self.measure == 'cosine':
            self.sim = cosine_sim
        elif self.measure == 'order':
            self.sim = order_sim
        else:
            print("Warning: Similarity measure not supported: {}".format(self.measure))
            self.sim = None

        self.max_violation = opt.vse_max_violation

    def forward(self, im, s, whole_batch=False, only_one_retrieval='off'):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        else:
            cost_s = cost_s.mean(1)
            cost_im = cost_im.mean(0)

        if whole_batch:
            fn = lambda x: x
        else:
            fn = lambda x: x.sum()

        if only_one_retrieval == 'image':
            return fn(cost_im)
        elif only_one_retrieval == 'caption':
            return fn(cost_s)
        else:
            return fn(cost_s) + fn(cost_im)


class PairLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt):
        super(PairLoss, self).__init__()
        self.margin = opt.vse_margin
        self.measure = opt.vse_measure
        if self.measure == 'cosine':
            self.sim = self.cosine_sim
        elif self.measure == 'order':
            self.sim = self.order_sim
        else:
            self.sim_net = nn.Sequential(
                nn.Linear(opt.vse_embed_size * 2, opt.vse_embed_size),
                nn.Dropout(0.5),
                nn.Linear(opt.vse_embed_size, 1))
            self.sim = lambda x, y: self.sim_net(torch.cat([x,y], 1))
        self.max_violation = opt.vse_max_violation


    def cosine_sim(self, im, s):
        return torch.bmm(im.unsqueeze(1), s.unsqueeze(2)).squeeze()

    def order_sim(self, im, s):
        YmX = s - im
        score = -YmX.clamp(min=0).pow(2).sum(2).squeeze(2).sqrt().t()
        return score
    

    def forward(self, im, im_d, s, whole_batch=False, only_one_retrieval='off'):
        scores = self.sim(im, s)
        im_d = im_d.view(im.size(0), -1, im.size(1))
        scores_d = self.sim(im_d.view(-1, im_d.size(2)), s.unsqueeze(1).expand_as(im_d).contiguous().view(-1, im_d.size(2)))
        # compare every diagonal score to scores in its column
        # caption retrieval
        if self.max_violation:
            cost = (self.margin + torch.max(scores_d.view(scores.size(0), -1), 1)[0] - scores).clamp(min=0)
        else:
            cost = (self.margin + scores_d.view(scores.size(0), -1) - scores.view(-1, 1)).clamp(min=0).mean()
        if whole_batch:
            return cost
        else:
            return cost.sum() # ???? why not mean


class VSEFCModel(nn.Module):
    """
    rkiros/uvs model
    """

    def __init__(self, opt):
        super(VSEFCModel, self).__init__()
        # tutorials/09 - Image Captioning
        # Build Models
        self.loss_type = opt.vse_loss_type

        self.img_enc = EncoderImage(opt)
        self.txt_enc = EncoderText(opt)
        # Loss and Optimizer
        self.contrastive_loss = ContrastiveLoss(opt)
        self.pair_loss = PairLoss(opt)

        self.margin = opt.vse_margin
        self.embed_size = opt.vse_embed_size

        self._loss = {}

    def forward(self, fc_feats, att_feats, seq, masks, whole_batch=False, only_one_retrieval='off'):

        img_emb = self.img_enc(fc_feats)
        cap_emb = self.txt_enc(seq, masks)

        if self.loss_type == 'contrastive':
            loss = self.contrastive_loss(img_emb, cap_emb, whole_batch, only_one_retrieval)
            if not whole_batch:
                self._loss['contrastive'] = loss.data[0]
        elif self.loss_type == 'pair':
            img_emb_d = self.img_enc(fc_feats_d)
            loss = self.pair_loss(img_emb, img_emb_d, cap_emb, whole_batch, only_one_retrieval)
            if not whole_batch:
                self._loss['pair'] = loss.data[0]
        else:
            img_emb_d = self.img_enc(fc_feats_d)
            loss_con = self.contrastive_loss(img_emb, cap_emb, whole_batch, only_one_retrieval) 
            loss_pair = self.pair_loss(img_emb, img_emb_d, cap_emb, whole_batch, only_one_retrieval)
            loss = (loss_con + loss_pair) / 2
            if not whole_batch:
                self._loss['contrastive'] = loss_con.data[0]
                self._loss['pair'] = loss_pair.data[0]

        return loss

