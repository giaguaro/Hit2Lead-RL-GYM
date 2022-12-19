#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys, os
import random
import time
import csv
import itertools
import copy

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import dgl
import dgl.function as fn
import networkx as nx
import pickle
import pandas as pd 

from rdkit import Chem  
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import qed, MolLogP
import sascorer
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog

import gym
import torch

from wrapper_schrodinger import SCHROD_MMGBSA
from overlay_openeye import OverlayOE
from ElabMol import ElabMols
from FragMol import SplitMol


def get_att_points(mol):
    att_points = []
    for a in mol.GetAtoms(): 
        if a.GetSymbol() == '*':
            att_points.append(a.GetIdx())

    return att_points



splitmol=SplitMol(args.target)


ATOM_VOCAB = ['C', 'N', 'O', 'S', 'P', 'F', 'I', 'Cl','Br', '*']

FRAG_VOCAB = open('motifs_350.txt','r').readlines() # n=350
FRAG_VOCAB = [s.strip('\n').split(',') for s in FRAG_VOCAB] 
FRAG_VOCAB_MOL = [Chem.MolFromSmiles(s[0]) for s in FRAG_VOCAB]
FRAG_VOCAB_ATT = [get_att_points(m) for m in FRAG_VOCAB_MOL]
CORE_VOCAB = splitmol.fragment_and_plif()



# From GCPN
def convert_radical_electrons_to_hydrogens(mol):
    """
    Converts radical electrons in a molecule into bonds to hydrogens. Only
    use this if molecule is valid. Results a new mol object
    :param mol: rdkit mol object
    :return: rdkit mol object
    """
    m = copy.deepcopy(mol)
    if Chem.Descriptors.NumRadicalElectrons(m) == 0:  # not a radical
        return m
    else:  # a radical
        for a in m.GetAtoms():
            num_radical_e = a.GetNumRadicalElectrons()
            if num_radical_e > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radical_e)
    return m


def adj2sparse(adj):
    """
        adj: [3, 47, 47] float numpy array
        return: a tuple of 2 lists
    """
    adj = [x*(i+1) for i, x in enumerate(adj)]
    adj = [sparse.dok_matrix(x) for x in adj]
    
    if not all([adj_i is None for adj_i in adj]):
        adj = sparse.dok_matrix(np.sum(adj))
        adj.setdiag(0)   

        all_edges = list(adj.items())
        e_types = np.array([edge[1]-1 for edge in all_edges], dtype=int)
        e = np.transpose(np.array([list(edge[0]) for edge in all_edges]))

        n_edges = len(all_edges)

        e_x = np.zeros((n_edges, 4))
        e_x[np.arange(n_edges),e_types] = 1
        e_x = torch.Tensor(e_x)
        return e, e_x
    else:
        return None

def get_att_points(mol):
    att_points = []
    for a in mol.GetAtoms(): 
        if a.GetSymbol() == '*':
            att_points.append(a.GetIdx())

    return att_points

def map_idx(idx, idx_list, mol):
    abs_id = idx_list[idx]
    neigh_idx = mol.GetAtomWithIdx(abs_id).GetNeighbors()[0].GetIdx()
    return neigh_idx 
  
def reward_vina(smis, predictor, reward_vina_min=0):
    reward = - np.array(predictor.predict(smis))
    reward = np.clip(reward, reward_vina_min, None)
    return reward

class LigEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def init(self, complex_="5t0e_clean.pdb", SCHROD_dir="/groups/cherkasvgrp/schrodinger", reward_mmgbsa_min=-5, ratios=dict(),reward_step_total=4,reward_target=0.5,max_action=128,min_action=20):
        '''
        own init function, since gym does not support passing argument
        '''

        self.mol = FRAG_VOCAB[0]
        
        elabmols = ElabMols()
        
        os.environ["SCHRODINGER"] = SCHROD_dir
        mmgbsa = SCHROD_MMGBSA()

        # init smi
        self.starting_complex = complex_
        splitmol = SplitMol(self.starting_complex)
        
        self.core_list = splitmol.fragment_and_plif()
        
        self.complex_list = []
        self.smile_list = []
        self.smile_old_list = []

        possible_atoms = ATOM_VOCAB
        possible_frags = FRAG_VOCAB
        possible_cores = CORE_VOCAB
        possible_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        self.atom_type_num = len(possible_atoms)
        self.frag_type_num = len(possible_motifs)
        self.core_type_num = len(possible_motifs)
        self.possible_atom_types = np.array(possible_atoms)
        self.possible_frags_types = np.array(possible_frags)
        self.possible_cores_types = np.array(possible_cores)

        self.possible_bond_types = np.array(possible_bonds, dtype=object)

        self.d_n = len(self.possible_atom_types)+18 

        self.max_action = max_action
        self.min_action = min_action

        self.max_atom = 150
        self.action_space = gym.spaces.MultiDiscrete([len(CORE_VOCAB), len(FRAG_VOCAB), 20])

        self.counter = 0
        self.level = 0 # for curriculum learning, level starts with 0, and increase afterwards

        self.predictor = mmgbsa

        self.attach_point = Chem.MolFromSmiles('*')

    def seed(self,seed):
        np.random.seed(seed=seed)
        random.seed(seed)
    
    def reset_batch(self):
        self.mol_list = []
        
    def reward_med(self):
        SA = sascorer.calculateScore(self.mol)
        qed = QED.qed(self.mol)
        reward = (0.5*SA + 0.5*qed)
        return reward

    def reward_batch(self):
        reward = []
        print('mol list', [Chem.MolToSmiles(x) for x in self.mol_list])
        for complex_ in self.complex_list:
            # here we supply the path of the complex (PDB)
            mmgbsa_prot_optimized, mmgbsa_lig_optimized, csv_out = mmgbsa.run(complex_)
            # get optimized complex after mmgbsa calculations and extract scores from the generated csv file
            mmgbsa_scores = mmgbsa.extract_results(csv_out)
            # remove any extra files generated
            shutil.rmtree(mmgbsa.temp) 

            print("MMGBSA complete\n")
            csv_out=os.path.join(mmgbsa.temp, f"pv_{mmgbsa.input}-out-out.csv")

            mmgbsa_scores = mmgbsa.extract_results(csv_out)

            score = mmgbsa_scores['r_psp_MMGBSA_dG_Bind']

            reward = np.clip(score, reward_mmgbsa_min, None) + self.reward_med
            
        return reward

    def reward_single(self, complex_list):
        reward = []
        print('mol list is custom') #, [Chem.MolToSmiles(x) for x in self.mol_list])
        for complex_ in complex_list:
            # here we supply the path of the complex (PDB)
            mmgbsa_prot_optimized, mmgbsa_lig_optimized, csv_out = mmgbsa.run(complex_)
            # get optimized complex after mmgbsa calculations and extract scores from the generated csv file
            mmgbsa_scores = mmgbsa.extract_results(csv_out)
            # remove any extra files generated
            shutil.rmtree(mmgbsa.temp) 

            print("MMGBSA complete\n")
            csv_out=os.path.join(mmgbsa.temp, f"pv_{mmgbsa.input}-out-out.csv")

            mmgbsa_scores = mmgbsa.extract_results(csv_out)

            score = mmgbsa_scores['r_psp_MMGBSA_dG_Bind']

            reward = np.clip(score, reward_mmgbsa_min, None) + self.reward_med
        return reward

    def step(self, ac):
        """
        Perform a given action
        :param action:
        :param action_type:
        :return: reward of 1 if resulting molecule graph does not exceed valency,
        -1 if otherwise
        """
#         ac = ac[0]
         
        ### init
        info = {}  # info we care about
        self.mol_old = copy.deepcopy(self.mol) # keep old mol
        
        stop = False    
        new = False
        
        if (self.counter >= self.max_action) or get_att_points(self.mol) == []:
            new = True
        else:
            self.mol = elabmols.elaborate(CORE_VOCAB[ac[0]], FRAG_VOCAB[ac[1]], ac[2])

        reward_step = 0.05
        if self.mol.GetNumAtoms() > self.mol_old.GetNumAtoms():
            reward_step += 0.005
        self.counter += 1

        if new:            
            reward = 0
            # Only store for obs if attachment point exists in o2
            if get_att_points(self.mol) != []:
                mol_no_att = self.get_final_mol() 
                Chem.SanitizeMol(mol_no_att, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                smi_no_att = Chem.MolToSmiles(mol_no_att)
                info['smile'] = smi_no_att
                print("smi:", smi_no_att)
                self.smile_list.append(smi_no_att)

                # Info for old mol
                mol_old_no_att = self.get_final_mol_ob(self.mol_old)
                Chem.SanitizeMol(mol_old_no_att, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                smi_old_no_att = Chem.MolToSmiles(mol_no_att)
                info['old_smi'] = smi_old_no_att
                self.smile_old_list.append(smi_old_no_att)

                stop = True
            else:
                stop = False
            self.counter = 0      

        ### use stepwise reward
        else:
            reward = reward_step

        info['stop'] = stop

        # get observation
        ob = self.get_observation()
        return ob,reward,new,info

    def reset(self,smile=None):
        '''
        to avoid error, assume an atom already exists
        :return: ob
        '''
        if smile is not None:
            self.mol = Chem.RWMol(Chem.MolFromSmiles(smile))
            Chem.SanitizeMol(self.mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        else:
            # init smi
            self.smi = self.starting_smi
            self.mol = Chem.MolFromSmiles(self.smi) 
        # self.smile_list = [] # only for single motif
        self.counter = 0
        ob = self.get_observation()
        return ob

    def render(self, mode='human', close=False):
        return

    def get_final_smiles_mol(self):
        """
        Returns a SMILES of the final molecule. Converts any radical
        electrons into hydrogens. Works only if molecule is valid
        :return: SMILES
        """
        m = Chem.DeleteSubstructs(self.mol, Chem.MolFromSmiles("*"))
        m = convert_radical_electrons_to_hydrogens(m)
        return m, Chem.MolToSmiles(m, isomericSmiles=True)

    def get_final_mol(self):
        """
        Returns a rdkit mol object of the final molecule. Converts any radical
        electrons into hydrogens. Works only if molecule is valid
        :return: SMILES
        """
        m = Chem.DeleteSubstructs(self.mol, Chem.MolFromSmiles("*"))
        return m
    
    def get_final_mol_ob(self, mol):
        m = Chem.DeleteSubstructs(mol, Chem.MolFromSmiles("*"))
        return m

    def get_observation(self, expert_smi=None):
        """
        ob['adj']:d_e*n*n --- 'E'
        ob['node']:1*n*d_n --- 'F'
        n = atom_num + atom_type_num
        """
        ob = {}

        if expert_smi:
            mol = Chem.MolFromSmiles(expert_smi)
        else:
            ob['att'] = get_att_points(self.mol)
            mol = copy.deepcopy(self.mol)
        
        try:
            Chem.SanitizeMol(mol)
        except:
            pass

        smi = Chem.MolToSmiles(mol)

        n = mol.GetNumAtoms()
        F = np.zeros((1, self.max_atom, self.d_n))

        for a in mol.GetAtoms():
            atom_idx = a.GetIdx()
            
            atom_symbol = a.GetSymbol()
            if self.has_feature:
                float_array = atom_feature(a, use_atom_meta=True)
            else:
                float_array = (atom_symbol == self.possible_atom_types).astype(float)

            F[0, atom_idx, :] = float_array

        d_e = len(self.possible_bond_types)
        E = np.zeros((d_e, self.max_atom, self.max_atom))

        for b in mol.GetBonds(): 
            begin_idx = b.GetBeginAtomIdx()
            end_idx = b.GetEndAtomIdx()
            bond_type = b.GetBondType()
            float_array = (bond_type == self.possible_bond_types).astype(float)
            try:
                assert float_array.sum() != 0
            except:
                print('error',bond_type)
            E[:, begin_idx, end_idx] = float_array
        
        if self.is_normalize:
            E = self.normalize_adj(E)
        
        ob_adj = adj2sparse(E.squeeze())
        ob_node = torch.Tensor(F)
        g = dgl.DGLGraph()

        ob_len = torch.sum(torch.sum(ob_node, dim=-1).bool().float().squeeze(-2), dim=-1)
        g.add_nodes(ob_len)
        if ob_adj is not None and len(ob_adj[0])>0 :
            g.add_edges(ob_adj[0][0], ob_adj[0][1], {'x': ob_adj[1]})
        g.ndata['x'] = ob_node[:, :int(ob_len),:].squeeze(0)
        
        ob['g'] = g
        ob['smi'] = smi
        
        return ob

    def get_observation_mol(self,mol):
        """
        ob['adj']:d_e*n*n --- 'E'
        ob['node']:1*n*d_n --- 'F'
        n = atom_num + atom_type_num
        """
        ob = {}

        ob['att'] = get_att_points(mol)
        
        try:
            Chem.SanitizeMol(mol)
        except:
            pass

        smi = Chem.MolToSmiles(mol)

        n = mol.GetNumAtoms()
        F = np.zeros((1, self.max_atom, self.d_n))

        for a in mol.GetAtoms():
            atom_idx = a.GetIdx()
            
            atom_symbol = a.GetSymbol()
            if self.has_feature:
                float_array = atom_feature(a, use_atom_meta=True)
            else:
                float_array = (atom_symbol == self.possible_atom_types).astype(float)

            F[0, atom_idx, :] = float_array

        d_e = len(self.possible_bond_types)
        E = np.zeros((d_e, self.max_atom, self.max_atom))

        for b in mol.GetBonds(): 

            begin_idx = b.GetBeginAtomIdx()
            end_idx = b.GetEndAtomIdx()
            bond_type = b.GetBondType()
            float_array = (bond_type == self.possible_bond_types).astype(float)

            try:
                assert float_array.sum() != 0
            except:
                print('error',bond_type)
            E[:, begin_idx, end_idx] = float_array
        
        if self.is_normalize:
            E = self.normalize_adj(E)
        
        ob_adj = adj2sparse(E.squeeze())
        ob_node = torch.Tensor(F)
        g = dgl.DGLGraph()

        ob_len = torch.sum(torch.sum(ob_node, dim=-1).bool().float().squeeze(-2), dim=-1)
        g.add_nodes(ob_len)
        if ob_adj is not None and len(ob_adj[0])>0 :
            g.add_edges(ob_adj[0][0], ob_adj[0][1], {'x': ob_adj[1]})
        g.ndata['x'] = ob_node[:, :int(ob_len),:].squeeze(0)
        
        ob['g'] = g
        ob['smi'] = smi
        return ob

