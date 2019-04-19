from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm

import torch.nn.functional as F

import numpy as np
from collections import OrderedDict


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    X = X / (X.norm(1, dim=dim, keepdim=True) + eps)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    X = X / (X.norm(2, dim=dim, keepdim=True) + eps)
    return X

class EncoderImage(nn.Module):

    def __init__(self, opt):
        super(EncoderImage, self).__init__()
        self.embed_size = opt.vse_embed_size
        self.no_imgnorm = opt.vse_no_imgnorm
        self.use_abs = opt.vse_use_abs
        self.fc_feat_size = opt.fc_feat_size

        self.fc = nn.Linear(self.fc_feat_size, self.embed_size)

        # self.att1 = Attention(opt)
        # self.att2 = Attention(opt)

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
            features = l2norm(features, dim=-1)

        return features

# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, opt):
        super(EncoderText, self).__init__()
        self.use_abs = opt.vse_use_abs
        self.word_dim = opt.vse_word_dim
        self.embed_size = opt.vse_embed_size
        self.num_layers = 1 #opt.vse_num_layers
        self.rnn_type = 'GRU' #opt.vse_rnn_type
        self.use_bi_gru = True
        self.vocab_size = opt.vocab_size
        self.use_abs = opt.vse_use_abs
        self.no_txtnorm = False
        # word embedding
        self.embed = nn.Embedding(self.vocab_size + 2, self.word_dim)

        # caption embedding
        # self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size, self.embed_size, self.num_layers, batch_first=True)
        self.rnn = nn.GRU(self.word_dim, self.embed_size, 1, batch_first=True, bidirectional=self.use_bi_gru)

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
        padded = pad_packed_sequence(out, batch_first=True, total_length=masks.shape[1])
        I = sorted_lens.view(-1,1,1).expand(seqs.size(0), 1, self.embed_size) - 1
        out = padded[0]
        # out = padded[0].gather(1, I).squeeze(1)

        if self.use_bi_gru:
            out = sum(out.chunk(2, dim=-1)) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            out = l2norm(out, dim=-1)

        out = out[inv_ix].contiguous()

        return out

def func_attention(query, query_mask, context, context_mask, opt, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)


    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    # if opt.raw_feature_norm == "softmax":
    #     # --> (batch*sourceL, queryL)
    #     attn = attn.view(batch_size*sourceL, queryL)
    #     attn = nn.Softmax()(attn)
    #     # --> (batch, sourceL, queryL)
    #     attn = attn.view(batch_size, sourceL, queryL)
    # elif opt.raw_feature_norm == "l2norm":
    #     attn = l2norm(attn, 2)
    # elif opt.raw_feature_norm == "clipped_l2norm":
    if True:
        if query_mask is not None:
            attn = attn * query_mask.unsqueeze(1)
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    # elif opt.raw_feature_norm == "l1norm":
    #     attn = l1norm_d(attn, 2)
    # elif opt.raw_feature_norm == "clipped_l1norm":
    #     attn = nn.LeakyReLU(0.1)(attn)
    #     attn = l1norm_d(attn, 2)
    # elif opt.raw_feature_norm == "clipped":
    #     attn = nn.LeakyReLU(0.1)(attn)
    # elif opt.raw_feature_norm == "no_norm":
    #     pass
    # else:
    #     raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = F.softmax(attn*smooth, -1)
    if context_mask is not None:
        attn = attn * context_mask.float().unsqueeze(1)
        attn = attn / (attn.sum(-1, keepdim=True) + 1e-8) # normalize to 1
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score_t2i(images, img_masks, captions, cap_masks, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = int((cap_masks[i].sum()).item())
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, None, images, img_masks, opt, smooth=opt.lambda_softmax)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if cap_masks is not None:
            row_sim = row_sim * cap_masks
        # if opt.agg_func == 'LogSumExp':
        #     row_sim.mul_(opt.lambda_lse).exp_()
        #     row_sim = row_sim.sum(dim=1, keepdim=True)
        #     row_sim = torch.log(row_sim)/opt.lambda_lse #broken
        # elif opt.agg_func == 'Max':
        #     row_sim = row_sim.max(dim=1, keepdim=True)[0] #broken
        # elif opt.agg_func == 'Sum':
        #     row_sim = row_sim.sum(dim=1, keepdim=True) #broken
        # elif opt.agg_func == 'Mean':
        if True:
            row_sim = row_sim.sum(dim=1, keepdim=True) / (cap_masks.sum(dim=1, keepdim=True)+1e-8)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    
    return similarities


def xattn_score_i2t(images, img_masks, captions, cap_masks, opt):
    """
    Images: (batch_size, max_n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = int((cap_masks[i].sum()).item())
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, img_masks, cap_i_expand, None, opt, smooth=4) # lambda softmax is 4
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if img_masks is not None:
            row_sim = row_sim * img_masks
        # if opt.agg_func == 'LogSumExp':
        #     row_sim.mul_(opt.lambda_lse).exp_()
        #     row_sim = row_sim.sum(dim=1, keepdim=True)
        #     row_sim = torch.log(row_sim)/opt.lambda_lse
        # elif opt.agg_func == 'Max':
        #     row_sim = row_sim.max(dim=1, keepdim=True)[0]
        # elif opt.agg_func == 'Sum':
        #     row_sim = row_sim.sum(dim=1, keepdim=True) borken
        # elif opt.agg_func == 'Mean':
        if True:
            row_sim = row_sim.sum(dim=1, keepdim=True) / (img_masks.sum(dim=1, keepdim=True) + 1e-8)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = opt.vse_margin
        self.max_violation = opt.vse_max_violation

    def forward(self, im, im_l, s, s_l):
        # compute image-sentence score matrix
        # if self.opt.cross_attn == 't2i':
        #     scores = xattn_score_t2i(im, s, s_l, self.opt)
        # elif self.opt.cross_attn == 'i2t':
        if True:
            scores = xattn_score_i2t(im, im_l, s, s_l, self.opt)
        # else:
        #     raise ValueError("unknown first norm type:", opt.raw_feature_norm)
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
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class VSESCANModel(nn.Module):
    """
    rkiros/uvs model
    """

    def __init__(self, opt):
        super(VSESCANModel, self).__init__()
        # tutorials/09 - Image Captioning
        # Build Models
        self.loss_type = opt.vse_loss_type

        self.img_enc = EncoderImage(opt)
        self.txt_enc = EncoderText(opt)
        # Loss and Optimizer
        self.contrastive_loss = ContrastiveLoss(opt)

        self.margin = opt.vse_margin
        self.embed_size = opt.vse_embed_size

        self._loss = {}

    def forward(self, fc_feats, att_feats, att_masks, seq, masks, whole_batch=False, only_one_retrieval='off'):
        img_emb = self.img_enc(att_feats)
        cap_emb = self.txt_enc(seq, masks)

        loss = self.contrastive_loss(img_emb, att_masks, cap_emb, masks)
        if not whole_batch:
            self._loss['contrastive'] = loss.item()

        return loss


# torch.autograd.grad(loss, [img_emb], retain_graph=True, allow_unused=True)[0]
# torch.autograd.grad(loss, [self.img_enc.fc.weight], retain_graph=True, allow_unused=True)[0]

# torch.autograd.grad(loss, [img_emb], retain_graph=True, allow_unused=True)[0]
# torch.autograd.grad(img_emb[ix], [self.img_enc.fc.weight], torch.autograd.grad(loss, [img_emb], retain_graph=True, allow_unused=True)[0][ix], retain_graph=True, allow_unused=True)[0]
