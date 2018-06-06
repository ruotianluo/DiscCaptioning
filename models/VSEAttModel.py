from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm

import torch.nn.functional as F

import numpy as np
from collections import OrderedDict


from .VSEFCModel import cosine_sim, order_sim, ContrastiveLoss, PairLoss


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.norm(X, dim=1, keepdim=True)
    X = torch.div(X, norm)
    return X

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.embed_size = opt.vse_embed_size
        self.att_hid_size = opt.att_hid_size
        self.fc_ctx = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.embed_size, self.att_hid_size), nn.Tanh())
        self.fc_att = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.embed_size, self.att_hid_size), nn.Tanh())
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.proj = nn.Sequential(nn.Linear(self.embed_size, self.embed_size), nn.Tanh())

    def forward(self, ctx, att, mask=None):
        h = self.fc_ctx(ctx).unsqueeze(1) * self.fc_att(att)
        alpha = F.softmax(self.alpha_net(h).squeeze(2))
        if mask is not None:
            alpha = alpha * mask[:, :alpha.size(1)].float()
            alpha = alpha / alpha.sum(1, keepdim=True) # normalize to 1

        return self.proj(torch.bmm(alpha.unsqueeze(1), att).squeeze(1))

class EncoderImage(nn.Module):

    def __init__(self, opt):
        super(EncoderImage, self).__init__()
        self.embed_size = opt.vse_embed_size
        self.no_imgnorm = opt.vse_no_imgnorm
        self.use_abs = opt.vse_use_abs
        self.fc_feat_size = opt.fc_feat_size

        self.fc = nn.Sequential(nn.Linear(self.fc_feat_size, self.embed_size), nn.Tanh())

        self.att1 = Attention(opt)
        self.att2 = Attention(opt)

    #     self.init_weights()

    # def init_weights(self):
    #     """Xavier initialization for the fully connected layer
    #     """
    #     r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
    #                               self.fc.out_features)
    #     self.fc.weight.data.uniform_(-r, r)
    #     self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        features = features.view(features.size(0), -1, features.size(3))

        v_0 = features.mean(1)
        v_1 = self.att1(v_0, features)
        v_2 = self.att2(v_1, features)

        features = torch.cat([v_0, v_1, v_2], 1)


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
        # word embedding
        self.embed = nn.Embedding(self.vocab_size + 2, self.input_encoding_size)

        # caption embedding
        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size, self.embed_size, self.num_layers, batch_first=True)

        self.att1 = Attention(opt)
        self.att2 = Attention(opt)   

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def pad_sentences(self, seqs, masks):
        len_sents = masks.long().sum(1)
        len_sents, len_ix = len_sents.sort(0, descending=True)

        inv_ix = len_ix.clone()
        inv_ix.data[len_ix.data] = torch.arange(0, len(len_ix)).type_as(inv_ix.data)

        new_seqs = seqs[len_ix].contiguous()

        return new_seqs, len_sents, inv_ix

    def forward(self, seqs, masks):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        padded_seqs, sorted_lens, inv_ix = self.pad_sentences(seqs, masks)

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
        out = padded[0]
        # out = padded[0].gather(1, I).squeeze(1)

        masks = masks.float()

        u_0 = out.sum(1) / masks.sum(1, keepdim=True)

        u_1 = self.att1(u_0, out, masks)
        u_2 = self.att2(u_1, out, masks)

        out = torch.cat([u_0, u_1, u_2], 1)

        # normalization in the joint embedding space
        out = l2norm(out)

        out = out[inv_ix].contiguous()

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out


class VSEAttModel(nn.Module):
    """
    rkiros/uvs model
    """

    def __init__(self, opt):
        super(VSEAttModel, self).__init__()
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

    def forward(self, fc_feats, att_feats, seq, masks, whole_batch=False):

        img_emb = self.img_enc(att_feats)
        cap_emb = self.txt_enc(seq, masks)

        if self.loss_type == 'contrastive':
            loss = self.contrastive_loss(img_emb, cap_emb, whole_batch)
            if not whole_batch:
                self._loss['contrastive'] = loss.data[0]
        elif self.loss_type == 'pair':
            img_emb_d = self.img_enc(att_feats_d)
            loss = self.pair_loss(img_emb, img_emb_d, cap_emb, whole_batch)
            if not whole_batch:
                self._loss['pair'] = loss.data[0]
        else:
            img_emb_d = self.img_enc(att_feats_d)
            loss_con = self.contrastive_loss(img_emb, cap_emb, whole_batch) 
            loss_pair = self.pair_loss(img_emb, img_emb_d, cap_emb, whole_batch)
            loss = (loss_con + loss_pair) / 2
            if not whole_batch:
                self._loss['contrastive'] = loss_con.data[0]
                self._loss['pair'] = loss_pair.data[0]

        return loss
