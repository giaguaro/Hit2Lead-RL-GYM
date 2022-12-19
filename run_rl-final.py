#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!/usr/bin/env python3

import os
import random

import numpy as np
import dgl
import torch 
from tensorboardX import SummaryWriter

from sacOffPolicy import sac


import gym

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    dgl.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train(args,seed,writer=None):

    workerseed = args.seed
    set_seed(workerseed)


    env = gym.make('molecule-circus')
    
    # to be modified for args
    env.init(complex_="5t0e_clean.pdb", SCHROD_dir="/groups/cherkasvgrp/schrodinger", reward_mmgbsa_min=-5, ratios=dict(),reward_step_total=4,reward_target=0.5,max_action=128,min_action=20)
    env.seed(workerseed)

    SAC = sac(writer, args, env, ac_kwargs=dict(), seed=seed, 
        steps_per_epoch=500, epochs=100, replay_size=int(1e6), gamma=0.99, 
        # polyak=0.995, lr=args.init_lr, alpha=args.init_alpha, batch_size=args.batch_size, start_steps=128,    
        polyak=0.995, lr=args.init_lr, alpha=args.init_alpha, batch_size=args.batch_size, start_steps=args.start_steps,
        update_after=args.update_after, update_every=args.update_every, update_freq=args.update_freq, 
        expert_every=5, num_test_episodes=8, max_ep_len=args.max_action, 
        save_freq=2000, train_alpha=True)
    SAC.train()
    

    env.close()

# 'environment name: molecule; graph'
env = "molecule-circus"

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def molecule_arg_parser():
    parser = arg_parser()

    # Choose RL model
    parser.add_argument('--rl_model', type=str, default='sac') # sac, td3, ddpg
    # env
    parser.add_argument('--seed', help='RNG seed', type=int, default=666)
    parser.add_argument('--num_steps', type=int, default=int(5e7))
    
    # parser.add_argument('--dataset', type=str, default='zinc',help='caveman; grid; ba; zinc; gdb')
    parser.add_argument('--target', type=str, default="5te0_clean.pdb", help='complex containing protein and docked pose')

    
    # rewards
    parser.add_argument('--reward_step_total', type=float, default=0.5)
    
    parser.add_argument('--intr_rew', type=str, default=None) # intr, mc
    parser.add_argument('--intr_rew_ratio', type=float, default=5e-1)
    
    parser.add_argument('--tau', type=float, default=1)
    
    parser.add_argument('--name',type=str,default='')
    parser.add_argument('--output',type=str,default='output')
    parser.add_argument('--name_full_load',type=str,default='')
    
    # model update
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--init_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--update_every', type=int, default=256)
    parser.add_argument('--update_freq', type=int, default=256)
    parser.add_argument('--update_after', type=int, default=2000)
    parser.add_argument('--start_steps', type=int, default=3000)
    
    # model save and load
    parser.add_argument('--save_every', type=int, default=500)
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--load_step', type=int, default=250)
    
    # graph embedding
    parser.add_argument('--graph_emb', type=int, default=0)
    parser.add_argument('--emb_size', type=int, default=64) # default 64

    parser.add_argument('--layer_num_g', type=int, default=3)

    parser.add_argument('--max_action', type=int, default=4) 
    parser.add_argument('--min_action', type=int, default=1) 

    # SAC
    parser.add_argument('--target_entropy', type=float, default=1.)
    parser.add_argument('--init_alpha', type=float, default=1.)
    parser.add_argument('--desc', type=str, default='ecfp') # ecfp
    parser.add_argument('--init_pi_lr', type=float, default=1e-4)
    parser.add_argument('--init_q_lr', type=float, default=1e-4)
    parser.add_argument('--init_alpha_lr', type=float, default=5e-4)
    parser.add_argument('--alpha_max', type=float, default=20.)
    parser.add_argument('--alpha_min', type=float, default=.05)

    # MC dropout
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--n_samples', type=int, default=5)

    # On-policy
    parser.add_argument('--n_cpus', type=int, default=1)
    parser.add_argument('--steps_per_epoch', type=int, default=257)
    
    return parser

def main():
    args = molecule_arg_parser().parse_args()
    print(args)

    ratios = dict()
    ratios['logp'] = 0
    ratios['qed'] = 0
    ratios['sa'] = 0
    ratios['mw'] = 0
    ratios['filter'] = 0
    ratios['mmgbsa'] = 1
    
    args.ratios = ratios
    
    # check and clean
    if not os.path.exists('molecule_gen'):
        os.makedirs('molecule_gen')
    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')

    writer = SummaryWriter(comment='_'+args.name)

    train(args,seed=args.seed,writer=writer)

if __name__ == '__main__':
    main()

