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
from rdkit.Chem.MolStandardize import rdMolStandardize
from biopandas.pdb import PandasPdb

class SplitMol:
    def __init__(self, PDB: str, MOL_SPLIT_START: int = 70, **kwargs):
        kwargs.setdefault('aggr', 'add')
#         super(SplitMol,self).__init__()
        
        self.MOL_SPLIT_START=MOL_SPLIT_START
        self.pdb=PDB
        self.records=['ATOM']
#         self.values=['HOH','CL','MG','ZN','MN','CA']
        self.ions=['HOH','CL','MG','ZN','MN','CA','FE','CO','K','CU', 'AG', 'NA', 'B', 'SI']

        self.path = os.getcwd()


    def okToBreak(self, bond):
        """
        Here we apply a bunch of rules to judge if the bond is OK to break.

        Parameters
        ----------
        bond :
            RDkit MOL object

        Returns
        -------
        Boolean :
            OK or not to break.
        """
        # See if the bond is in Ring (don't break that)
        if bond.IsInRing():
            return False
        # We OK only single bonds to break
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            return False

        # Get the beginning atom of the bond
        begin_atom = bond.GetBeginAtom()
        # Get the ending atom of the bond
        end_atom = bond.GetEndAtom()
        # What kind of neighbors does these end and begenning atoms have? We need a family of no less than 5!
        neighbor_end=list(end_atom.GetNeighbors())
        neighbor_begin=list(begin_atom.GetNeighbors())
        if (len(neighbor_end) + len(neighbor_begin)) <5:
            return False
        #for atm in neighbor_end:
            #print(atm.GetAtomicNum())
        #print(begin_atom.GetAtomicNum(), end_atom.GetAtomicNum(), MOL_SPLIT_START)
        
        # Now check if end or begenning atoms are in ring (we dont wanna bother those)
        if not(begin_atom.IsInRing() or end_atom.IsInRing()):
            return False
        elif begin_atom.GetAtomicNum() >= self.MOL_SPLIT_START or \
                end_atom.GetAtomicNum() >= self.MOL_SPLIT_START:
            return False
        elif end_atom.GetAtomicNum() == 1:
            return False
        else:
            return True

    def undo_id_label (self, frag, split_id):
        # I am trying to restore Hydrogens where the break happened
        for i, atom in enumerate(frag.GetAtoms()):
            if atom.GetAtomicNum() >= split_id:
                atom.SetAtomicNum(1)

        return frag

#     def assign_dummy (self, frag, split_id):
#         # I am trying to restore Hydrogens where the break happened
#         for i, atom in enumerate(frag.GetAtoms()):
#             if atom.GetAtomicNum() >= split_id:
#                 atom.SetAtomicNum(1)

#         return frag


    def FragID_assign(self, mol):
        invariantID=AllChem.GetHashedMorganFingerprint(mol,radius=2,nBits=1024)
        key=str(''.join([str(item) for item in invariantID]))
        try:
            return FragID[key]
        except:
            FragID[key] = len(FragID)+1
            return [FragID[key]]
        
    def FragID_assign_2(self, mol):
            return list(MACCSkeys.GenMACCSKeys(mol))

# remove the fragment from the main molecule
    def remove_fragment(self, mol, frag):
        coords_frag=[]
        
        indices=[atom.GetIdx() for atom in frag.GetAtoms()]
        frag_conf=frag.GetConformer()
        for idx in indices:
            coords_frag.append(list(frag_conf.GetAtomPosition(idx)))

        to_remove=[]
        indices=[atom.GetIdx() for atom in mol.GetAtoms()]
        mol_conf=mol.GetConformer()
        for idx in indices: 
            if any(coord == tuple(round(y) for y in list(mol_conf.GetAtomPosition(idx))) \
                   for coord in [tuple(round(y) for y in x) for x in coords_frag]):
                to_remove.append(idx)

        # now remove the atoms on identified indices of fragments

        m = copy.deepcopy(mol)
        em1 = Chem.EditableMol(m)
        atomsToRemove=to_remove
        atomsToRemove.sort(reverse=True)
        for atom in atomsToRemove:
            em1.RemoveAtom(atom)

        m2 = em1.GetMol()
        Chem.SanitizeMol(m2)
        Chem.MolToSmiles(m2)

        return Chem.rdmolops.GetMolFrags(m2, asMols=True)

    def decomposed_atoms_idx(self, mol, coords_att):
        
        mol_idx=[]
        indices=[atom.GetIdx() for atom in mol.GetAtoms()]
        mol_conf=mol.GetConformer()
        for idx in indices: 
            if any(coord == tuple(round(y) for y in list(mol_conf.GetAtomPosition(idx))) \
                   for coord in [tuple(round(y) for y in x) for x in coords_att]):
                mol_idx.append(idx)
        return mol_idx
        
    
    # Divide a molecule into fragments
    def split_molecule(self, mol, pdb):
                
        # we may need to overwrite with rdkit sanitized mol
        w = Chem.PDBWriter(f"tmp_{pdb}_ligand.pdb")
        w.write(mol)
        w.close()
        
        
        split_id = self.MOL_SPLIT_START

        res = []
        res_no_id=[]
        ok_bonds_list=[]
        res_with_dummy = [ [] for _ in range(10) ]
        
        to_check = [mol]
        while len(to_check) > 0:
            ok_bonds, ms = self.spf(mol, to_check.pop(), split_id)
            ok_bonds_list.append(ok_bonds)
            if len(ms) == 1:
                res += ms
            else:
                to_check += ms
                split_id += 1
        
        ok_bonds=sorted([number for group in ok_bonds_list for number in group])
        largest_Fragment = rdMolStandardize.LargestFragmentChooser(True)
        mol=Chem.rdmolops.AddHs(mol, addCoords=True)
        for n, bond in enumerate(ok_bonds):
            res_with_dummy[n]=largest_Fragment.choose(Chem.FragmentOnBonds(mol, [bond]))

        

#         for frag in res:
            
#             trimmed_mol=self.remove_fragment(mol,frag)
            
#             res_no_id.append(frag)
#             res_no_id.append(self.undo_id_label(frag, self.MOL_SPLIT_START))
        
        #stop everything if frags exceed 10
#         if len(res_no_id) > 10:
#             raise Exception(f"sorry does not support large ligands with more than 10 fragments possible")
                         
#         # Now we index each of the 3D feats into a slot of the allocated 10 fragments (max). If exceed return error.
#         # i.e. we discard that sample
#         trail=[0,0,0,0,0,0,0,0,0,0,0]
#         trail=[trail]*10
#         for idx, pos in enumerate(zip(trail,descriptors3D)):
#             trail[idx]=descriptors3D[idx]    
            
#         descriptors3D=trail
        
        
        return res_with_dummy


    # Function for doing all the nitty gritty splitting work.
    # loops over bonds until bonds get exhausted or bonds are ok to break, whichever comes first. If ok to break, then each
    # fragment needs to be checked individually again through the loop
    def spf(self, original, mol, split_id):
        
        ok_bonds=[]
        bonds = mol.GetBonds()
        for i in range(len(bonds)):
            if self.okToBreak(bonds[i]):
                
                mol = Chem.FragmentOnBonds(mol, [i])
                # Dummy atoms are always added last
                n_at = mol.GetNumAtoms()
                print('Split ID', split_id)
                att1=mol.GetAtomWithIdx(n_at-1)
                att2=mol.GetAtomWithIdx(n_at-2)
                
                mol_conf=mol.GetConformer()
                att1_idx=self.decomposed_atoms_idx(original, [list(mol_conf.GetAtomPosition(n_at-1))])
                att2_idx=self.decomposed_atoms_idx(original, [list(mol_conf.GetAtomPosition(n_at-2))])
                
                bond_idx=original.GetBondBetweenAtoms(att1_idx[0],att2_idx[0]).GetIdx()
                ok_bonds.append(bond_idx)
                
                att1.SetAtomicNum(split_id)
                att2.SetAtomicNum(split_id)
                return ok_bonds, Chem.rdmolops.GetMolFrags(mol, asMols=True)

        # If the molecule could not been split, return original molecule
        return [], [mol]

    def pdb_2_sdf(self, pdb):
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("pdb", "sdf")
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, pdb)   # Open Babel will uncompress automatically

        mol.AddHydrogens()


        obConversion.WriteFile(mol, f"{pdb.split('.')[0]}.sdf")
        return f"{pdb.split('.')[0]}.sdf"
    
    def pdb_2_mol2(self, pdb):
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("pdb", "mol2")
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, pdb)   # Open Babel will uncompress automatically

        mol.AddHydrogens()


        obConversion.WriteFile(mol, f"{pdb.split('.')[0]}.mol2")
        return f"{pdb.split('.')[0]}.mol2"
    
    def sdf_2_pdb(self, sdf):
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("sdf", "pdb")
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, sdf)   # Open Babel will uncompress automatically

        mol.AddHydrogens()
        obConversion.WriteFile(mol, f"{sdf.split('.')[0]}.pdb")
        return f"{sdf.split('.')[0]}.pdb"

    def save_bpdb(self, pdb,ppdb, record):  
        ppdb.to_pdb(path=f"{record}_{pdb.split('.')[0].split('_')[0]}.pdb",
                    records=[record],
                    gz=False, 
                    append_newline=True)

#     def get_HOH_pdb(self, pdb):
#         ppdb = PandasPdb() 
#         ppdb.read_pdb(pdb) 
#         ppdb.df['HETATM']=ppdb.df['HETATM'].loc[ppdb.df['HETATM']['residue_name'].isin(self.values)]
#         ppdb.to_pdb(path=f"HOH_{pdb.split('.')[0].split('_')[0]}.pdb",
#                 records=['HETATM'],
#                 gz=False, 
#                 append_newline=True)

    def keep_relevant_hetatm(self, pdb):
        raw=str(self.pdb).split('/')[-1]
        with open(pdb) as f1, open(f"{raw.split('.')[0]}_protein.pdb", 'w') as f2:
            for line in f1:
                if 'ATOM' in line:
                    f2.write(line)
        with open(pdb) as f1, open(f"{raw.split('.')[0]}_ligand.pdb", 'w') as f2:
            for line in f1:
                if ('HETATM' in line) and not any(ion in line for ion in self.ions):
                    f2.write(line)
        return
    
    
    def fragment_and_plif(self):
        path = os.getcwd()
        if not os.path.exists('fragment'):
            os.mkdir(f'{path}/fragment')
        os.chdir("fragment")
        
        raw=self.pdb.split('/')[-1].split('.')[0]
        self.keep_relevant_hetatm(self.pdb)
        lig_sdf=self.pdb_2_sdf(f'{raw}_ligand.pdb')
        lig_mol2=self.pdb_2_mol2(f'{raw}_ligand.pdb')
        
#         # in case the original pdb file is corrupted
#         content = open(f'{raw}_ligand.pdb').read()
#         hets=re.findall("^HETATM (.*)", content, re.M)
#         if len(hets)<5:

#             with open(f'{raw}_ligand.pdb', 'r') as file :
#                 filedata = file.read()

#             # Replace the target string
#             filedata = filedata.replace('ATOM  ', 'HETATM')

#             # Write the file out again
#             with open(f'{raw}_ligand.pdb', 'w') as file:
#                 file.write(filedata)
        
        try: 
            fragment_mols = Chem.RemoveHs(fragment_mols[0])
            output_frags = self.split_molecule(fragment_mols,raw)
            
        except:  
            try:
                fragment_mols = Chem.SDMolSupplier(lig_sdf, removeHs=True, sanitize=False)
                output_frags = self.split_molecule(fragment_mols[0],raw)
            except:
                try: 
                    fragment_mols_alt = Chem.MolFromMol2File(lig_mol2, sanitize=True, removeHs=True)
                    output_frags = self.split_molecule(fragment_mols_alt,raw)
                except:
                    try: 
                        fragment_mols = AllChem.MolFromPDBFile(f'{raw}_ligand.pdb')
                        output_frags = self.split_molecule(fragment_mols,raw)
                    except:
                        raise Exception(f"Something wrong with ligand")
        os.chdir(f'{path}')
        
        return output_frags

