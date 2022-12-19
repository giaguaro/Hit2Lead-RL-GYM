#!/usr/bin/env python
# coding: utf-8

# In[7]:


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

from env_gym.environment import ATOM_VOCAB, CORE_VOCAB, FRAG_VOCAB, FRAG_VOCAB_MOL # CORE_VOCAB_MOL ?


# to define outside:
# CORE_VOCAB_MOL

def get_final_motif(mol):
    m = Chem.DeleteSubstructs(mol, Chem.MolFromSmiles("*"))
    m.UpdatePropertyCache()
    FastFindRings(m)
    return m

def ecfp(molecule):
    molecule = get_final_motif(molecule)
    return [x for x in AllChem.GetMorganFingerprintAsBitVect(
        molecule, 2, 1024)]


class SFSPolicy(nn.Module):
    def __init__(self, ob_space, ac_space, env, args):
        super().__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.ac_dim = len(FRAG_VOCAB)-1
        self.emb_size = args.emb_size
        self.tau = args.tau
        
        # init candidate atoms
        self.bond_type_num = 4

        self.env = env # env utilized to init cand motif mols
        
        # creates the graphs for each of the fragments
        self.coreCand, self.fragCand = self.create_candidate_motifs()

        self.coreCand_g = dgl.batch([x['g'] for x in self.coreCand])
        self.fragCand_g = dgl.batch([x['g'] for x in self.fragCand])

        self.coreCand_ob_len = self.coreCand_g.batch_num_nodes().tolist()
        self.fragCand_ob_len = self.fragCand_g.batch_num_nodes().tolist()

        # Create candidate descriptors
        desc = ecfp
        self.desc_dim = 1024

        # calculate the ecfp for the core
        self.core_desc = torch.Tensor([desc(Chem.MolFromSmiles(x['smi'])) 
                                for x in self.coreCand]).to(self.device)
        # we should have here up to 10 (max fragmentation of the core)
        self.coreMotif_type_num = len(self.coreCand)
            
        # calculate the ecfp for the frag
        self.frag_desc = torch.Tensor([desc(Chem.MolFromSmiles(x['smi'])) 
                                for x in self.fragCand]).to(self.device)
        # shouldn't we get here like 350 or something
        self.fragMotif_type_num = len(self.fragCand)

        
        #choosing the core
        self.action1_layers = nn.ModuleList([nn.Bilinear(self.desc_dim,args.emb_size, args.emb_size).to(self.device),
                                nn.Linear(self.desc_dim, args.emb_size, bias=False).to(self.device),
                                nn.Linear(args.emb_size, args.emb_size, bias=False).to(self.device), 
                                nn.Sequential(
                                nn.Linear(args.emb_size, args.emb_size, bias=False),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size, args.emb_size, bias=True),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size, 1, bias=True),
                                )])
        
        #choosing the frag
        self.action2_layers = nn.ModuleList([nn.Bilinear(self.desc_dim,args.emb_size, args.emb_size).to(self.device),
                                nn.Linear(self.desc_dim, args.emb_size, bias=False).to(self.device),
                                nn.Linear(args.emb_size, args.emb_size, bias=False).to(self.device), 
                                nn.Sequential(
                                nn.Linear(args.emb_size, args.emb_size, bias=False),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size, args.emb_size, bias=True),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size, 1, bias=True),
                                )])        
        
        
        
        #choosing the att on frag
        self.action3_layers = nn.ModuleList([nn.Bilinear(2*args.emb_size, 2*args.emb_size, args.emb_size).to(self.device),
                                nn.Linear(2*args.emb_size, args.emb_size, bias=False).to(self.device),
                                nn.Linear(2*args.emb_size, args.emb_size, bias=False).to(self.device), 
                                nn.Sequential(
                                nn.Linear(args.emb_size, args.emb_size//2, bias=False),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size//2, 1, bias=True)).to(self.device)])
        
        # Zero padding with max number of actions
        self.max_action = 60 # max atoms
        
        print('number of candidate motifs : ', len(self.fragCand))
        self.ac3_att_len = torch.LongTensor([len(x['att']) 
                                for x in self.fragCand]).to(self.device)
        self.ac3_att_mask = torch.cat([torch.LongTensor([i]*len(x['att'])) 
                                for i, x in enumerate(self.fragCand)], dim=0).to(self.device)

    def create_candidate_motifs(self):
        core_motif_gs = [self.env.get_observation_mol(mol) for mol in CORE_VOCAB_MOL]
        frag_motif_gs = [self.env.get_observation_mol(mol) for mol in FRAG_VOCAB_MOL]
        return core_motif_gs, frag_motif_gs


    def gumbel_softmax(self, logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1, \
                    g_ratio: float = 1e-3) -> torch.Tensor:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels * g_ratio) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)
        
        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

    def forward(self, graph_emb, node_emb, g, coreCands, fragCands, deterministic=False):
        """
        graph_emb : bs x hidden_dim
        node_emb : (bs x num_nodes) x hidden_dim)
        g: batched graph
        att: indexs of attachment points, list of list
        """
        
        g.ndata['node_emb'] = node_emb
        fragCand_g, fragCand_node_emb, fragCand_graph_emb = fragCands
        coreCand_g, coreCand_node_emb, coreCand_graph_emb = coreCands

        # Only acquire node embeddings with attatchment points
        ob_len = g.batch_num_nodes().tolist()
        att_mask = g.ndata['att_mask'] # used to select att embs from node embs
        
        if g.batch_size != 1:
            att_mask_split = torch.split(att_mask, ob_len, dim=0)
            att_len = [torch.sum(x, dim=0) for x in att_mask_split]
        else:
            att_len = torch.sum(att_mask, dim=-1) # used to torch.split for att embs

        coreCand_att_mask = coreCand_g.ndata['att_mask']
        coreCand_att_mask_split = torch.split(coreCand_att_mask, self.coreCand_ob_len, dim=0)
        coreCand_att_len = [torch.sum(x, dim=0) for x in coreCand_att_mask_split]

        fragCand_att_mask = fragCand_g.ndata['att_mask']
        fragCand_att_mask_split = torch.split(fragCand_att_mask, self.fragCand_ob_len, dim=0)
        fragCand_att_len = [torch.sum(x, dim=0) for x in fragCand_att_mask_split]
        
        # =============================== 
        # step 1 : which core to choose
        # =============================== 

        if g.batch_size != 1:
            graph_expand = torch.cat([graph_emb[i].unsqueeze(0).repeat(1, self.coreMotif_type_num, 1) for i in range(g.batch_size)], dim=0).contiguous()
        else:
            graph_expand = graph_emb.repeat(1, self.coreMotif_type_num, 1)
            
        cand_expand = self.coreCand_desc.unsqueeze(0).repeat(g.batch_size, 1, 1)
        
        emb_cat = self.action1_layers[0](cand_expand, graph_expand) + \
                    self.action1_layers[1](cand_expand) + self.action1_layers[2](graph_expand)

        logit_first = self.action1_layers[3](emb_cat).squeeze(-1)
        ac_first_prob = F.softmax(logit_first, dim=-1) + 1e-8
        log_ac_first_prob = ac_first_prob.log()
        
        ac_first_hot = self.gumbel_softmax(ac_first_prob, tau=self.tau, hard=True, g_ratio=1e-3)                                    
        emb_first = torch.matmul(ac_first_hot, coreCand_graph_emb)
        ac_first = torch.argmax(ac_first_hot, dim=-1)

        # Print gumbel otuput
        ac_second_gumbel = self.gumbel_softmax(ac_first_prob, tau=self.tau, hard=False, g_ratio=1e-3)     
        
        # =============================== 
        # step 2 : which frag to choose
        # =============================== 

        emb_first_expand = emb_first.view(-1, 1, self.emb_size).repeat(1, self.fragMotif_type_num, 1)
        cand_expand = self.fragCand_desc.unsqueeze(0).repeat(g.batch_size, 1, 1)
        
        emb_cat = self.action2_layers[0](cand_expand, emb_first_expand) + \
                    self.action2_layers[1](cand_expand) + self.action2_layers[2](emb_first_expand)

        logit_second = self.action2_layers[3](emb_cat).squeeze(-1)
        ac_second_prob = F.softmax(logit_second, dim=-1) + 1e-8
        log_ac_second_prob = ac_second_prob.log()
        
        ac_second_hot = self.gumbel_softmax(ac_second_prob, tau=self.tau, hard=True, g_ratio=1e-3)                                    
        emb_second = torch.matmul(ac_second_hot, fragCand_graph_emb)
        ac_second = torch.argmax(ac_second_hot, dim=-1)

        # Print gumbel otuput
        ac_second_gumbel = self.gumbel_softmax(ac_second_prob, tau=self.tau, hard=False, g_ratio=1e-3)        

        # ===============================  
        # step 3 : where to add on frag motif
        # ===============================

        # Select att points from candidate
        cand_att_emb = torch.masked_select(fragCand_node_emb, fragCand_att_mask.unsqueeze(-1))
        cand_att_emb = cand_att_emb.view(-1, 2*self.emb_size)

        ac3_att_mask = self.ac3_att_mask.repeat(g.batch_size, 1) # bs x (num cands * num att size)
        ac3_att_mask = torch.where(ac3_att_mask==ac_second.view(-1,1),
                            1, 0).view(g.batch_size, -1) # (num_cands * num_nodes)
        ac3_att_mask = ac3_att_mask.bool()

        ac3_cand_emb = torch.masked_select(cand_att_emb.view(1, -1, 2*self.emb_size), 
                                ac3_att_mask.view(g.batch_size, -1, 1)).view(-1, 2*self.emb_size)#.view(1, -1, 2*self.emb_size)
        
        ac3_att_len = torch.index_select(self.ac3_att_len, 0, ac_second).tolist()
        emb_second_expand = torch.cat([emb_second[i].unsqueeze(0).repeat(ac3_att_len[i],1) for i in range(g.batch_size)]).contiguous()

        emb_cat_ac3 = self.action3_layers[0](emb_second_expand, ac3_cand_emb) + self.action3_layers[1](emb_second_expand) \
                  + self.action3_layers[2](ac3_cand_emb)
        
        logits_third = self.action3_layers[3](emb_cat_ac3)

        # predict logit
        if g.batch_size != 1:
            ac_third_prob = [torch.softmax(logit,dim=-1)
                            for i, logit in enumerate(torch.split(logits_third.squeeze(-1), ac3_att_len, dim=0))]
            ac_third_prob = [p+1e-8 for p in ac_third_prob]
            log_ac_third_prob = [x.log() for x in ac_third_prob]

        else:
            logits_third = logits_third.transpose(1,0)
            ac_third_prob = torch.softmax(logits_third, dim=-1) + 1e-8
            log_ac_third_prob = ac_third_prob.log()
        
        # gumbel softmax sampling and zero-padding
        if g.batch_size != 1:
            third_stack = []
            third_ac_stack = []
            for i, node_emb_i in enumerate(torch.split(emb_cat_ac3, ac3_att_len, dim=0)):
                ac_third_hot_i = self.gumbel_softmax(ac_third_prob[i], tau=self.tau, hard=True, dim=-1)
                ac_third_i = torch.argmax(ac_third_hot_i, dim=-1)
                third_stack.append(torch.matmul(ac_third_hot_i, node_emb_i))
                third_ac_stack.append(ac_third_i)

                del ac_third_hot_i
            emb_third = torch.stack(third_stack, dim=0).squeeze(1)
            ac_third = torch.stack(third_ac_stack, dim=0)
            ac_third_prob = torch.cat([
                                torch.cat([ac_third_prob_i, ac_third_prob_i.new_zeros(
                                    self.max_action - ac_third_prob_i.size(0))]
                                        , dim=0).contiguous().view(1,self.max_action)
                                for i, ac_third_prob_i in enumerate(ac_third_prob)], dim=0).contiguous()
            
            log_ac_third_prob = torch.cat([
                                    torch.cat([log_ac_third_prob_i, log_ac_third_prob_i.new_zeros(
                                        self.max_action - log_ac_third_prob_i.size(0))]
                                            , 0).contiguous().view(1,self.max_action)
                                    for i, log_ac_third_prob_i in enumerate(log_ac_third_prob)], dim=0).contiguous()

        else:
            ac_third_hot = self.gumbel_softmax(ac_third_prob, tau=self.tau, hard=True, dim=-1)
            ac_third = torch.argmax(ac_third_hot, dim=-1)
            emb_third = torch.matmul(ac_third_hot, emb_cat_ac3)
            
            ac_third_prob = torch.cat([ac_third_prob, ac_third_prob.new_zeros(
                                        1, self.max_action - ac_third_prob.size(1))] 
                                , -1).contiguous()
            log_ac_third_prob = torch.cat([log_ac_third_prob, log_ac_third_prob.new_zeros(
                                        1, self.max_action - log_ac_third_prob.size(1))]
                                , -1).contiguous()

        # ==== concat everything ====

        ac_prob = torch.cat([ac_first_prob, ac_second_prob, ac_third_prob], dim=1).contiguous()
        log_ac_prob = torch.cat([log_ac_first_prob, 
                            log_ac_second_prob, log_ac_third_prob], dim=1).contiguous()
        ac = torch.stack([ac_first, ac_second, ac_third], dim=1)

        return ac, (ac_prob, log_ac_prob), (ac_first_prob, ac_second_hot, ac_third_prob)
    
    def sample(self, ac, graph_emb, node_emb, g, coreCands, fragCands):
        g.ndata['node_emb'] = node_emb
        fragCand_g, fragCand_node_emb, fragCand_graph_emb = fragCands
        coreCand_g, coreCand_node_emb, coreCand_graph_emb = coreCands

        # Only acquire node embeddings with attatchment points
        ob_len = g.batch_num_nodes().tolist()
        att_mask = g.ndata['att_mask'] # used to select att embs from node embs
        att_len = torch.sum(att_mask, dim=-1) # used to torch.split for att embs

        coreCand_att_mask = coreCand_g.ndata['att_mask']
        coreCand_att_mask_split = torch.split(coreCand_att_mask, self.coreCand_ob_len, dim=0)
        coreCand_att_len = [torch.sum(x, dim=0) for x in coreCand_att_mask_split]

        fragCand_att_mask = fragCand_g.ndata['att_mask']
        fragCand_att_mask_split = torch.split(fragCand_att_mask, self.fragCand_ob_len, dim=0)
        fragCand_att_len = [torch.sum(x, dim=0) for x in fragCand_att_mask_split]


        # =============================== 
        # step 1 : which core to choose
        # =============================== 
        # select only nodes with attachment points
        
        graph_expand = graph_emb.repeat(1, self.coreMotif_type_num, 1)
        cand_expand = self.cand_desc.unsqueeze(0).repeat(g.batch_size, 1, 1)     
        
        emb_cat = self.action1_layers[0](cand_expand, graph_expand) + \
                    self.action1_layers[1](cand_expand) + self.action1_layers[2](graph_expand)

        logit_second = self.action1_layers[3](emb_cat).squeeze(-1)
        ac_first_prob = F.softmax(logit_first, dim=-1) + 1e-8
        log_ac_first_prob = ac_first_prob.log()
        
        ac_first_hot = self.gumbel_softmax(ac_first_prob, tau=self.tau, hard=True, g_ratio=1e-3)                                    
        emb_first = torch.matmul(ac_first_hot, coreCand_graph_emb)
        ac_first = torch.argmax(ac_first_hot, dim=-1)
        
        # =============================== 
        # step 2 : which frag motif to choose
        # ===============================   
        emb_first_expand = emb_first[ac[0]].unsqueeze(0).view(-1, 1, self.emb_size).repeat(1, self.fragMotif_type_num, 1)
        cand_expand = self.cand_desc.unsqueeze(0).repeat(g.batch_size, 1, 1)     
        
        emb_cat = self.action2_layers[0](cand_expand, emb_first_expand) + \
                    self.action2_layers[1](cand_expand) + self.action2_layers[2](emb_first_expand)

        logit_second = self.action2_layers[3](emb_cat).squeeze(-1)
        ac_second_prob = F.softmax(logit_second, dim=-1) + 1e-8
        log_ac_second_prob = ac_second_prob.log()
        
        ac_second_hot = self.gumbel_softmax(ac_second_prob, tau=self.tau, hard=True, g_ratio=1e-3)                                    
        emb_second = torch.matmul(ac_second_hot, fragCand_graph_emb)
        ac_second = torch.argmax(ac_second_hot, dim=-1)

        # ===============================  
        # step 3 : where to add on motif
        # ===============================
        # Select att points from candidates
        
        cand_att_emb = torch.masked_select(cand_node_emb, cand_att_mask.unsqueeze(-1))
        cand_att_emb = cand_att_emb.view(-1, 2*self.emb_size)

        ac3_att_mask = self.ac3_att_mask.repeat(g.batch_size, 1) # bs x (num cands * num att size)
        # torch where currently does not support cpu ops    
        
        ac3_att_mask = torch.where(ac3_att_mask==ac[1], 
                            1, 0).view(g.batch_size, -1) # (num_cands * num_nodes)
        ac3_att_mask = ac3_att_mask.bool()

        ac3_cand_emb = torch.masked_select(cand_att_emb.view(1, -1, 2*self.emb_size), 
                                ac3_att_mask.view(g.batch_size, -1, 1)).view(-1, 2*self.emb_size)
        
        ac3_att_len = self.ac3_att_len[ac[1]]
        emb_second_expand = emb_second.repeat(ac3_att_len,1)
        emb_cat_ac3 = self.action3_layers[0](emb_second_expand, ac3_cand_emb) + self.action3_layers[1](emb_second_expand) \
                  + self.action3_layers[2](ac3_cand_emb)

        logits_third = self.action3_layers[3](emb_cat_ac3)
        logits_third = logits_third.transpose(1,0)
        ac_third_prob = torch.softmax(logits_third, dim=-1) + 1e-8
        log_ac_third_prob = ac_third_prob.log()

        # gumbel softmax sampling and zero-padding
        emb_third = emb_cat_ac3[ac[2]].unsqueeze(0)
        ac_third_prob = torch.cat([ac_third_prob, ac_third_prob.new_zeros(
                                        1, self.max_action - ac_third_prob.size(1))] 
                                , -1).contiguous()
        log_ac_third_prob = torch.cat([log_ac_third_prob, log_ac_third_prob.new_zeros(
                                        1, self.max_action - log_ac_third_prob.size(1))]
                                , -1).contiguous()

        # ==== concat everything ====
        ac_prob = torch.cat([ac_first_prob, ac_second_prob, ac_third_prob], dim=1).contiguous()
        log_ac_prob = torch.cat([log_ac_first_prob, 
                            log_ac_second_prob, log_ac_third_prob], dim=1).contiguous()

        return (ac_prob, log_ac_prob), (ac_first_prob, ac_second_hot, ac_third_prob)


# In[ ]:




