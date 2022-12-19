import openeye
from openeye.oechem import *
from openeye.oeiupac import *
from openeye.oeomega import *
from openeye.oeshape import *
from openeye.oedepict import *
from openeye import oechem
from openeye import oeshape
from openeye import oeomega
from openeye import oequacpac
import sys, os, subprocess



class OverlayOE:
    
    def __init__(self, fitMolDir, refMolDir, outputMol, license_filename= 'oe_license_1.txt'):

        self.temp = "OE_dir/"
        self.fitMolDir=fitMolDir
        self.refMolDir=refMolDir
        self.fitMol = fitMolDir.split("/")[-1]
        self.refMol = refMolDir.split("/")[-1]
        self.outputMol = outputMol
        
        
        if os.path.isfile(license_filename):
            license_file = open(license_filename, 'r')
            openeye.OEAddLicenseData(license_file.read())
            license_file.close()
        else:
            print("Error: Your OpenEye license is not readable; please check your filename and that you have mounted your Google Drive")

        licensed = openeye.oechem.OEChemIsLicensed()
        print("Was your OpenEye license correctly installed (True/False)? " + str(licensed))
        if not licensed:
            print("Error: Your OpenEye license is not correctly installed.")
            raise Exception("Error: Your OpenEye license is not correctly installed.")
    
        if not os.path.isdir(f"./{self.temp}"):
            os.makedirs(f"./{self.temp}")
     
    def Tautomerize(self, tauto_prefix="tauto"):
        
        os.chdir("./OE_dir/")
        ifs = oechem.oemolistream()
        if not ifs.open(self.fitMolDir):
            oechem.OEThrow.Fatal("Unable to open %s for reading" % self.fitMolDir)
        ofs = oechem.oemolostream()
        if not ofs.open(f"{tauto_prefix}_{self.fitMol}"):
            oechem.OEThrow.Fatal("Unable to open %s for reading" % f"{tauto_prefix}_{self.fitMol}")
        
        tautomerOptions = oequacpac.OETautomerOptions()
        tautomerOptions.SetMaxZoneSize(50)
#         tautomerOptions.SetMaxTautomericAtoms(50)
    #    tautomerOptions.SetClearCoordinates(True)
#         tautomerOptions.maxToReturn(100)
#         tautomerOptions.SetLevel(3)
#         tautomerOptions.SetHybridizationLevel(1)
        pKaNorm = True
        
        
        
        for mol in ifs.GetOEGraphMols():
            oequacpac.OEHypervalentNormalization(mol)
            oequacpac.OERemoveFormalCharge(mol)
            
            for tautomer in oequacpac.OEEnumerateTautomers(mol, tautomerOptions):
                oechem.OEWriteMolecule(ofs, tautomer)
        
        
        os.chdir("../")
        
        return f"{tauto_prefix}_{self.fitMol}"
    
    def ConformerGen(self, in_mol2, omega2_exe="/groups/cherkasvgrp/openeye/bin/omega2", prefix="confs"):
        """ 
        Runs OpenEye OMEGA2 in a shell environment to 
        generate up to 200 low-energy concormers.
        Keyword Arguments:
            omega2_exe (str): path to the OMEGA2 executable
            fraglib (str): path to the OMEGA2 fragment library
            in_mol2 (str): path to the input mol2 structure
            out_mol2 (str): path to the multi-conformer output file
        """
        out_mol2=f"{prefix}_{in_mol2}"
        os.chdir("./OE_dir/")
        subprocess.call("{0} " \
                "-in {1} " \
                "-out {2} " \
                "-warts true -mpi_np 16 " \
                "-commentEnergy true " \
                "-strictstereo false -maxconfs 100 -flipper True -rms 1.0 -maxRot 9999 "\
                "-maxTime 1000 -enumNitrogen all " \
                "-flipper_maxCenters 31".format(
                omega2_exe, in_mol2, out_mol2), shell=True)
        os.chdir("../")
        return f"{prefix}_{in_mol2}"
    
    def Charge(self, in_mol, out_prefix="charge", chargeName = "mmff94"):
        os.chdir("./OE_dir/")

        def AssignChargesByName(mol, name = "mmff94"):
            if name == "noop":
                return oequacpac.OEAssignCharges(mol, oequacpac.OEChargeEngineNoOp())
            elif name == "mmff" or name == "mmff94":
                return oequacpac.OEAssignCharges(mol, oequacpac.OEMMFF94Charges())
            elif name == "am1bcc":
                return oequacpac.OEAssignCharges(mol, oequacpac.OEAM1BCCCharges())
            elif name == "am1bccnosymspt":
                optimize = True
                symmetrize = True
                return oequacpac.OEAssignCharges(mol,
                                                 oequacpac.OEAM1BCCCharges(not optimize, not symmetrize))
            elif name == "amber" or name == "amberff94":
                return oequacpac.OEAssignCharges(mol, oequacpac.OEAmberFF94Charges())
            elif name == "am1bccelf10":
                return oequacpac.OEAssignCharges(mol, oequacpac.OEAM1BCCELF10Charges())
            return False
        
        ifs = oechem.oemolistream()
        if not ifs.open(in_mol):
            oechem.OEThrow.Fatal("Unable to open %s for reading" % in_mol)
        
        ofs = oechem.oemolostream()
        if not ofs.open(f"{out_prefix}_{in_mol.split('.')[0]}.mol2"):
            oechem.OEThrow.Fatal("Unable to open %s for writing" % f"{out_prefix}_{in_mol.split('.')[0]}.mol2")
        

        mol = oechem.OEMol()
        while oechem.OEReadMolecule(ifs, mol):
            if not AssignChargesByName(mol, chargeName):
                oechem.OEThrow.Warning("Unable to assign %s charges to mol %s"
                                       % (chargeName, mol.GetTitle()))
            oechem.OEWriteMolecule(ofs, mol)

        ifs.close()
        ofs.close()
        os.chdir("../")
        
        return f"{out_prefix}_{in_mol.split('.')[0]}.mol2"
  
    def FlexOverlay(self, fitMol, refMol=None, output=None):
        
        if refMol==None: refMol=self.refMolDir
        if output==None: output=self.outputMol
            
        os.chdir("./OE_dir/")
        overlayOpts = oeshape.OEFlexiOverlayOptions()

        ifs = oechem.oemolistream()
        if not ifs.open(fitMol):
            oechem.OEThrow.Fatal("Unable to open %s for reading" % fitMol)

        rfs = oechem.oemolistream()
        if not rfs.open(refMol):
            oechem.OEThrow.Fatal("Unable to open %s for reading" % refMol)

        ofs = oechem.oemolostream()
        if not ofs.open(output):
            oechem.OEThrow.Fatal("Unable to open %s for writing" % output)

        refmol = oechem.OEMol()
        oechem.OEReadMolecule(rfs, refmol)
        print("Ref. Title:", refmol.GetTitle())

        overlay = oeshape.OEFlexiOverlay(overlayOpts)
        overlay.SetupRef(refmol)
        
        print("Overlaying starts now ...")
        os.chdir("../")
        for fitmol in ifs.GetOEMols():
            results = oeshape.OEFlexiOverlapResults()
            if overlay.BestOverlay(results, fitmol):
                print("Fit Title: %-4s Tanimoto Combo: %.2f Energy: %2f"
                      % (fitmol.GetTitle(), results.GetTanimotoCombo(), results.GetInternalEnergy()))
                oechem.OESetSDData(fitmol, "Tanimoto Combo", "%.2f" % results.GetTanimotoCombo())
                oechem.OESetSDData(fitmol, "Energy", "%.2f" % results.GetInternalEnergy())

                if results.GetTanimotoCombo() >= 0.95: 
                    oechem.OEWriteMolecule(ofs, fitmol)
                    print("FOUND good overlay")
                    break
                else: 
                    print("Mol not written due to low score")
            else:
                print("Failed Title: %-4s" % (fitmol.GetTitle()))
                
#         os.chdir("../")
        return output
