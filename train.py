from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
import copy
from misc.rewards import init_scorer

try:
    import tensorflow as tf
    from tensorboardX import SummaryWriter
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

def train(opt):
    opt.use_att = utils.if_use_att(opt)
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tf_summary_writer = tf and SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')) as f:
                histories = cPickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
        best_val_score_vse = infos.get('best_val_score_vse', None)

    model = models.JointModel(opt)
    model.cuda()

    update_lr_flag = True
    # Assure in training mode
    model.train()

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=opt.learning_rate, weight_decay=opt.weight_decay)

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from, 'optimizer.pth')):
        state_dict = torch.load(os.path.join(opt.start_from, 'optimizer.pth'))
        if len(state_dict['state']) == len(optimizer.state_dict()['state']):
            optimizer.load_state_dict(state_dict)
        else:
            print('Optimizer param group number not matched? There must be new parameters. Reinit the optimizer.')

    init_scorer(opt.cached_tokens)
    while True:
        if update_lr_flag:
                # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            else:
                opt.current_lr = opt.learning_rate
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.caption_generator.ss_prob = opt.ss_prob
            # Assign retrieval loss weight
            if epoch > opt.retrieval_reward_weight_decay_start and opt.retrieval_reward_weight_decay_start >= 0:
                frac = (epoch - opt.retrieval_reward_weight_decay_start) // opt.retrieval_reward_weight_decay_every
                model.retrieval_reward_weight = opt.retrieval_reward_weight * (opt.retrieval_reward_weight_decay_rate  ** frac)
            update_lr_flag = False
                
        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')
        print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['fc_feats'], data['att_feats'], data['att_masks'], data['labels'], data['masks']]
        tmp = utils.var_wrapper(tmp)
        fc_feats, att_feats, att_masks, labels, masks = tmp
        
        optimizer.zero_grad()
        
        loss = model(fc_feats, att_feats, att_masks, labels, masks, data)
        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss.data[0]
        torch.cuda.synchronize()
        end = time.time()
        print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
            .format(iteration, epoch, train_loss, end - start))
        prt_str = ""
        for k, v in model.loss().items():
            prt_str += "{} = {:.3f} ".format(k, v)
        print(prt_str)

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            if tf is not None:
                tf_summary_writer.add_scalar('train_loss', train_loss, iteration)
                for k,v in model.loss().items():
                    tf_summary_writer.add_scalar(k, v, iteration)
                tf_summary_writer.add_scalar('learning_rate', opt.current_lr, iteration)
                tf_summary_writer.add_scalar('scheduled_sampling_prob', model.caption_generator.ss_prob, iteration)
                tf_summary_writer.add_scalar('retrieval_reward_weight', model.retrieval_reward_weight, iteration)
                tf_summary_writer.file_writer.flush()

            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.caption_generator.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            # Load the retrieval model for evaluation
            val_loss, predictions, lang_stats = eval_utils.eval_split(model, loader, eval_kwargs)

            # Write validation result into summary
            if tf is not None:
                for k,v in val_loss.items():
                    tf_summary_writer.add_scalar('validation '+k, v, iteration)
                for k,v in lang_stats.items():
                    tf_summary_writer.add_scalar(k, v, iteration)
                tf_summary_writer.add_text('Captions', '.\n\n'.join([_['caption'] for _ in predictions[:100]]), iteration)
                #tf_summary_writer.add_image('images', utils.make_summary_image(), iteration)
                #utils.make_html(opt.id, iteration)
                tf_summary_writer.file_writer.flush()

            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['SPICE']*100
            else:
                current_score = - val_loss['loss_cap']
            current_score_vse = val_loss.get(opt.vse_eval_criterion, 0)*100

            best_flag = False
            best_flag_vse = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                if best_val_score_vse is None or current_score_vse > best_val_score_vse:
                    best_val_score_vse = current_score_vse
                    best_flag_vse = True
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model-%d.pth'%(iteration))
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['best_val_score_vse'] = best_val_score_vse
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-%d.pkl'%(iteration)), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)
                if best_flag_vse:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model_vse-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_vse_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
train(opt)
