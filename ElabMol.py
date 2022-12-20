#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import re
from glob import glob
from openbabel import pybel
ob = pybel.ob
import numpy as np
import random
import warnings
import requests
import os
import pandas as pd
import openbabel
import copy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors3D 
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdDistGeom
from rdkit.Chem.MolStandardize import rdMolStandardize
from biopandas.pdb import PandasPdb



class ElabMols():
    
    def __init__(self):
        return 
        
    def get_att_pts(self, mol):

        exitVectorIdx=[]
        neighVectorIdx=[]
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                exitVectorIdx.append(atom.GetIdx())
                neighVectorIdx.append([nei.GetIdx() for nei in atom.GetNeighbors()][0]) #Dummy atom only has one neighbour

        #exitVectorPos = np.array(mol.GetConformer().GetAtomPosition(exitVectorIdx))
        return exitVectorIdx, neighVectorIdx
    
    def connectMols(self, mol1, mol2, neigh1, neigh2):
            combined = Chem.CombineMols(mol1, mol2)
            emol = Chem.EditableMol(combined)
            atom1 = mol1.GetAtomWithIdx(neigh1)
            atom2 = mol2.GetAtomWithIdx(neigh2)
            neighbor1_idx = atom1.GetNeighbors()[0].GetIdx()
            neighbor2_idx = atom2.GetNeighbors()[0].GetIdx()
            atom1_idx = atom1.GetIdx()
            atom2_idx = atom2.GetIdx()
            bond_order = atom2.GetBonds()[0].GetBondType()
            emol.AddBond(neighbor1_idx,
                         neighbor2_idx + mol1.GetNumAtoms(),
                         order=bond_order)
            emol.RemoveAtom(atom2_idx + mol1.GetNumAtoms())
            emol.RemoveAtom(atom1_idx)
            mol = emol.GetMol()
            return mol 
    
    def align(self, mol1, mol2, du1, du2, ca1, ca2):
     
        param = rdDistGeom.ETKDGv2()
        param.pruneRmsThresh = 0.1
        cids = rdDistGeom.EmbedMultipleConfs(mol2, 1, param)

        molIndex1=mol1.GetAtomWithIdx(du1).SetAtomicNum(1)
        molIndex2=mol2.GetAtomWithIdx(du2).SetAtomicNum(1)

        aligned=Chem.rdMolAlign.AlignMol(mol2,mol1,atomMap=((ca2,du1),(du2,ca1))) 
        connected=self.connectMols(mol1, mol2, du1, du2)
        
        #################### naturalize dummy atom! ########################
#         for atom in connected.GetAtoms():
#             if atom.GetAtomicNum() == 0:
#                 atom.SetAtomicNum(1)

        return connected
    
    
    # THIS NEEDS REWORK AFTER INTROUDCING THE NEW ACTION OF CHOOSING FRAG ATTACHMENT SITE
    def elaborate(self, mol, frag, att_idx): # pass one index of the mols (mol) and the df of the fragments (frags)
        
        mol_att, mol_ca = self.get_att_pts(mol)

        frag_mol = Chem.MolFromSmiles(frag)
        frag_atts, frag_ca = self.get_att_pts(frag_mol)         

#         elabs = []
#         # if we have more than one possible attachment point
#         if len(frag_atts) > 1 :
#             for du, ca in zip(frag_atts, frag_ca):
        chosen_du = frag_atts[att_idx]
        neigh_ca = frag_ca[att_idx]
#                 elabs.append(self.align(mol, frag_mol, mol_att[0], chosen_du,mol_ca[0],neigh_ca))

#         else:
        elab = self.align(mol, frag_mol, mol_att[0], chosen_du, mol_ca[0], neigh_ca)
        
        return elab
                    

