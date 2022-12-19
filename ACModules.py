#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from copy import deepcopy
import math
import time
from scipy import sparse
import scipy.signal
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributions as td
from torch.distributions.normal import Normal

import gym
import dgl
import dgl.function as fn
from dgl.nn.pytorch.glob import SumPooling

from rdkit import Chem

from env_gym.environment import ATOM_VOCAB, CORE_VOCAB, CORE_VOCAB_MOL, FRAG_VOCAB, FRAG_VOCAB_MOL

def get_final_motif(mol):
    m = Chem.DeleteSubstructs(mol, Chem.MolFromSmiles("*"))
    m.UpdatePropertyCache()
    FastFindRings(m)
    return m

def ecfp(molecule):
    molecule = get_final_motif(molecule)
    return [x for x in AllChem.GetMorganFingerprintAsBitVect(
        molecule, 2, 1024)]


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])
    
# DGL operations
msg = fn.copy_src(src='x', out='m')
def reduce_mean(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'x': accum}

def reduce_sum(nodes):
    accum = torch.sum(nodes.mailbox['m'], 1)
    return {'x': accum}  

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)

def create_candidate_motifs(self):
    motif_gs = [self.env.get_observation_mol(mol) for mol in FRAG_VOCAB_MOL]
    return motif_gs

class GCN_MC(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=.3, agg="sum", is_normalize=False, residual=True):
        super().__init__()
        self.residual = residual
        self.agg = agg
        self.is_normalize = is_normalize
        self.linear1 = nn.Linear(in_channels, out_channels, bias=False)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, g):

        g.update_all(msg, reduce_sum)
        h = self.linear1(g.ndata['x'])
        # apply MC dropout
        h = self.dropout(h)
        h = self.activation(h)
        if self.is_normalize:
            h = F.normalize(h, p=2, dim=1)
        
        if self.residual:
            h += h_in
        return h

class GCNEmbed_MC(nn.Module):
    def __init__(self, args):

        ### GCN
        super().__init__()
        self.device = args.device
        self.possible_atoms = ATOM_VOCAB
        self.bond_type_num = 4
        self.d_n = len(self.possible_atoms)+18
        
        self.emb_size = args.emb_size * 2
        self.gcn_aggregate = args.gcn_aggregate

        in_channels = 8
        self.emb_linear = nn.Linear(self.d_n, in_channels, bias=False)

        self.dropout = nn.Dropout(args.dropout)

        self.gcn_type = 'GCN'

        self.gcn_layers = nn.ModuleList([GCN_MC(in_channels, self.emb_size, 
                            dropout=args.dropout, agg="sum", residual=False)])
        for _ in range(args.layer_num_g-1):
            self.gcn_layers.append(GCN_MC(self.emb_size, self.emb_size, 
                            dropout=args.dropout, agg="sum"))

        self.pool = SumPooling()

    def forward(self, ob):
        ## Graph
        
        ob_g = [o['g'] for o in ob]
        ob_att = [o['att'] for o in ob]

        # create attachment point mask as one-hot
        for i, x_g in enumerate(ob_g):
            att_onehot = F.one_hot(torch.LongTensor(ob_att[i]), 
                        num_classes=x_g.number_of_nodes()).sum(0)
            ob_g[i].ndata['att_mask'] = att_onehot.bool()

        g = deepcopy(dgl.batch(ob_g)).to(self.device)
        
        g.ndata['x'] = self.emb_linear(g.ndata['x'])
        g.ndata['x'] = self.dropout(g.ndata['x'])

        for i, conv in enumerate(self.gcn_layers):
            h = conv(g)
            g.ndata['x'] = h
        
        emb_node = g.ndata['x']

        ## Get graph embedding
        emb_graph = self.pool(g, g.ndata['x'])
        
        return g, emb_node, emb_graph
    
class GCNPredictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = GCNEmbed(args)
        self.pred_layer = nn.Sequential(
                    nn.Linear(args.emb_size*2, args.emb_size, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(args.emb_size, 1, bias=True))

    def forward(self, o):
        _, _, graph_emb = self.embed(o)
        pred = self.pred_layer(graph_emb)
        return pred
    
class GCNActive(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embed = GCNEmbed_MC(args)

        self.batch_size = args.batch_size
        self.device = args.device
        self.emb_size = args.emb_size
        self.max_action2 = len(ATOM_VOCAB)
        self.max_action_stop = 2
        
        self.n_samples = args.n_samples
        
        self.pred_layer = nn.Sequential(
                    nn.Linear(args.emb_size*2, args.emb_size, bias=False),
                    nn.Dropout(args.dropout),
                    nn.ReLU(inplace=True))
        self.mean_layer = nn.Linear(args.emb_size, 1, bias=True)
        self.var_layer = nn.Sequential(
                            nn.Linear(args.emb_size, 1, bias=True),
                            nn.Softplus())
        
    def forward(self, o):
        _, _, graph_emb = self.embed(o)
        pred = self.pred_layer(graph_emb)
        pred_mean = self.mean_layer(pred)
        pred_logvar = (self.var_layer(pred) + 1e-12).log()
        return pred_mean, pred_logvar
    
    def forward_n_samples(self, o):
        samples_mean = []
        samples_var = []
        for _ in range(self.n_samples):
            
            _, _, graph_emb = self.embed(o)
            pred = self.pred_layer(graph_emb)
            samples_mean.append(self.mean_layer(pred))
            samples_var.append((self.var_layer(pred) + 1e-12).log())
        samples_mean = torch.stack(samples_mean, dim=1) # bs x n samples x 1
        samples_var = torch.stack(samples_var, dim=1) # bs x n samples x 1
        
        return samples_mean, samples_var
    

class GCNActorCritic(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        # build policy and value functions
        self.embed = GCNEmbed(args)
        ob_space = env.observation_space
        ac_space = env.action_space
        self.env = env
        #(o_g_emb, o_n_emb, o_g, cands)
        self.pi = SFSPolicy(ob_space, ac_space, env, args)
        self.q1 = GCNQFunction(ac_space, args)
        self.q2 = GCNQFunction(ac_space, args, override_seed=True)

        # PER based model
            
        self.p = GCNPredictor(args)

        # curiosity driven model
        if args.intr_rew == 'pe':
            self.p = GCNPredictor(args)
        elif args.intr_rew == 'bu':
            self.p = GCNActive(args)
        
        self.cand = self.create_candidate_motifs()

    def create_candidate_motifs(self):
        motif_gs = [self.env.get_observation_mol(mol) for mol in FRAG_VOCAB_MOL]
        return motif_gs

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            o_g, o_n_emb, o_g_emb = self.embed(obs)
            cands = self.embed(deepcopy(self.cand))
            a, _, _ = self.pi(o_g_emb, o_n_emb, o_g, cands, deterministic)
        return a

class GCNQFunction(nn.Module):
    def __init__(self, ac_space, args, override_seed=False):
        super().__init__()
        if override_seed:
            seed = args.seed + 1
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.batch_size = args.batch_size
        self.device = args.device
        self.emb_size = args.emb_size
        self.max_action2 = len(ATOM_VOCAB)
        self.max_action_stop = 2

        self.d = 2 * args.emb_size + len(FRAG_VOCAB) + len(CORE_VOCAB) + 40
        self.out_dim = 1
        
        self.qpred_layer = nn.Sequential(
                            nn.Linear(self.d, int(self.d//2), bias=False),
                            nn.ReLU(inplace=False),
                            nn.Linear(int(self.d//2), self.out_dim, bias=True))
    
    def forward(self, graph_emb, ac_first_prob, ac_second_hot, ac_third_prob):
        emb_state_action = torch.cat([graph_emb, ac_first_prob, ac_second_hot, ac_third_prob], dim=-1).contiguous()
        qpred = self.qpred_layer(emb_state_action)
        return qpred
    
    
def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class GCNEmbed(nn.Module):
    def __init__(self, args):

        ### GCN
        super().__init__()
        self.device = args.device
        self.possible_atoms = ATOM_VOCAB
        self.bond_type_num = 4
        self.d_n = len(self.possible_atoms)+18
        
        self.emb_size = args.emb_size * 2
        self.gcn_aggregate = args.gcn_aggregate

        in_channels = 8
        self.emb_linear = nn.Linear(self.d_n, in_channels, bias=False)

        self.gcn_type = "GCN"

        self.gcn_layers = nn.ModuleList([GCN(in_channels, self.emb_size, agg="sum", residual=False)])
        for _ in range(args.layer_num_g-1):
            self.gcn_layers.append(GCN(self.emb_size, self.emb_size, agg="sum"))
        self.pool = SumPooling()
        
        
    def forward(self, ob):
        ## Graph
        ob_g = [o['g'] for o in ob]
        ob_att = [o['att'] for o in ob]

        # create attachment point mask as one-hot
        for i, x_g in enumerate(ob_g):
            att_onehot = F.one_hot(torch.LongTensor(ob_att[i]), 
                        num_classes=x_g.number_of_nodes()).sum(0)
            ob_g[i].ndata['att_mask'] = att_onehot.bool()

        g = deepcopy(dgl.batch(ob_g)).to(self.device)
        
        g.ndata['x'] = self.emb_linear(g.ndata['x'])

        for i, conv in enumerate(self.gcn_layers):
            h = conv(g)
            g.ndata['x'] = h
        
        emb_node = g.ndata['x']

        ## Get graph embedding
        emb_graph = self.pool(g, g.ndata['x'])
        
        return g, emb_node, emb_graph

