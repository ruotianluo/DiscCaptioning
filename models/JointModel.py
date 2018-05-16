import torch
import torch.nn as nn
from . import load, setup
from torch.autograd import Variable
import misc.utils as utils
import torch.nn.functional as F
import misc.rewards as rewards
import numpy as np

class JointModel(nn.Module):
    def __init__(self, opt):
        super(JointModel, self).__init__()
        self.opt = opt
        self.use_word_weights = getattr(opt, 'use_word_weights' ,0)

        self.caption_generator = setup(opt, opt.caption_model, True)
        
        if opt.vse_model != 'None':
            self.vse = setup(opt, opt.vse_model, False)
            self.share_embed = opt.share_embed
            self.share_fc = opt.share_fc
            if self.share_embed:
                self.vse.txt_enc.embed = self.caption_generator.embed
            if self.share_fc:
                assert self.vse.embed_size == self.caption_generator.input_encoding_size
                if hasattr(self.caption_generator, 'img_embed'):
                    self.vse.img_enc.fc = self.caption_generator.img_embed
                else:
                    self.vse.img_enc.fc = self.caption_generator.att_embed
        else:
            self.vse = lambda x,y,z,w,u: Variable(torch.zeros(1)).cuda()

        if opt.vse_loss_weight == 0 and isinstance(self.vse, nn.Module):
            for p in self.vse.parameters():
                p.requires_grad = False

        self.vse_loss_weight = opt.vse_loss_weight
        self.caption_loss_weight = opt.caption_loss_weight

        self.retrieval_reward = opt.retrieval_reward # none, reinforce, gumbel
        self.retrieval_reward_weight = opt.retrieval_reward_weight # 

        self.reinforce_baseline_type = getattr(opt, 'reinforce_baseline_type', 'greedy')

        self.only_one_retrieval = getattr(opt, 'only_one_retrieval', 'off')

        self.cider_optimization = getattr(opt, 'cider_optimization', 0)

        self._loss = {}

        load(self, opt)
        if getattr(opt, 'initialize_retrieval', None)is not None:
            print("Make sure the vse opt are the same !!!!!\n"*100)
            utils.load_state_dict(self, {k:v for k,v in torch.load(opt.initialize_retrieval).items() if 'vse.' in k})


    def forward(self, fc_feats, att_feats, att_masks, seq, masks, data):
        if self.caption_loss_weight > 0 and not self.cider_optimization:
            loss_cap = self.caption_generator(fc_feats, att_feats, att_masks, seq, masks)
        else:
            loss_cap = Variable(torch.cuda.FloatTensor([0]))
        if self.vse_loss_weight > 0:
            loss_vse = self.vse(fc_feats, att_feats, seq, masks,only_one_retrieval=self.only_one_retrieval)
        else:
            loss_vse = Variable(torch.cuda.FloatTensor([0]))

        loss = self.caption_loss_weight * loss_cap + self.vse_loss_weight * loss_vse

        if self.retrieval_reward_weight > 0:
            if True:
                _seqs, _sampleLogProbs = self.caption_generator.sample(fc_feats, att_feats, att_masks, {'sample_max':0, 'temperature':1})
                gen_result, sample_logprobs = _seqs, _sampleLogProbs
                _masks = torch.cat([Variable(_seqs.data.new(_seqs.size(0), 2).fill_(1).float()), (_seqs > 0).float()[:, :-1]], 1)

                gen_masks = _masks

                _seqs = torch.cat([Variable(_seqs.data.new(_seqs.size(0), 1).fill_(self.caption_generator.vocab_size + 1)), _seqs], 1)

                if True:
                    retrieval_loss = self.vse(fc_feats, att_feats, _seqs, _masks, True, only_one_retrieval=self.only_one_retrieval)
                    if self.reinforce_baseline_type == 'greedy':
                        _seqs_greedy, _sampleLogProbs_greedy = self.caption_generator.sample(*utils.var_wrapper([fc_feats, att_feats, att_masks],volatile=True), opt={'sample_max':1, 'temperature':1})
                        greedy_res = _seqs_greedy
                        # Do we need weights here???
                        if True: #not self.use_word_weights:
                             _masks_greedy = torch.cat([Variable(_seqs_greedy.data.new(_seqs.size(0), 2).fill_(1).float()), (_seqs_greedy > 0).float()[:, :-1]], 1)
                        else:
                            _masks_greedy = self.get_word_weights_mask(_seqs_greedy)

                        _seqs_greedy = torch.cat([Variable(_seqs_greedy.data.new(_seqs_greedy.size(0), 1).fill_(self.caption_generator.vocab_size + 1)), _seqs_greedy], 1)
                            
                        baseline = self.vse(fc_feats, att_feats, _seqs_greedy, _masks_greedy, True, only_one_retrieval=self.only_one_retrieval)
                    elif self.reinforce_baseline_type == 'gt':
                        baseline = self.vse(fc_feats, att_feats, seq, masks,True, only_one_retrieval=self.only_one_retrieval)
                    else:
                        baseline = 0

                sc_loss = _sampleLogProbs * (utils.var_wrapper(retrieval_loss) - utils.var_wrapper(baseline)).detach().unsqueeze(1) * (_masks[:, 1:].detach().float())
                sc_loss = sc_loss.sum() / _masks[:, 1:].data.float().sum()

                loss += self.retrieval_reward_weight * sc_loss

                self._loss['retrieval_sc_loss'] = sc_loss.data[0]

                self._loss['retrieval_loss'] = retrieval_loss.sum().data[0]
                self._loss['retrieval_loss_greedy'] = baseline.sum().data[0] if isinstance(baseline, Variable) else baseline

        if self.cider_optimization:
            if 'gen_result' not in locals():
                gen_result, sample_logprobs = self.caption_generator.sample(fc_feats, att_feats, att_masks, opt={'sample_max':0})
                gen_masks = torch.cat([Variable(gen_result.data.new(gen_result.size(0), 2).fill_(1).float()), (gen_result > 0).float()[:, :-1]], 1)
            if 'greedy_res' not in locals():
                greedy_res, _ =  self.caption_generator.sample(*utils.var_wrapper([fc_feats, att_feats, att_masks], volatile=True), opt={'sample_max':1})
            reward, cider_greedy = rewards.get_self_critical_reward(data, gen_result, greedy_res)
            self._loss['avg_reward'] = reward.mean()
            self._loss['cider_greedy'] = cider_greedy
            loss_cap = sample_logprobs * utils.var_wrapper(-reward.astype('float32')).unsqueeze(1) * (gen_masks[:, 1:].detach())
            loss_cap = loss_cap.sum() / gen_masks[:, 1:].data.float().sum()

            loss += self.caption_loss_weight * loss_cap

        self._loss['loss_cap'] = loss_cap.data[0]
        self._loss['loss_vse'] = loss_vse.data[0]
        self._loss['loss'] = loss.data[0]

        return loss

    def sample(self, fc_feats, att_feats, att_masks, opt={}):
        return self.caption_generator.sample(fc_feats, att_feats, att_masks, opt)

    def loss(self):
        out = {}
        out.update(self._loss)
        out.update({'cap_'+k: v for k,v in self.caption_generator._loss.items()})
        out.update({'vse_'+k: v for k,v in self.vse._loss.items()})

        return out
