import os
import pandas as pd



class SCHROD_MMGBSA(object):
    """
    Minimal wrapper for SCHRODINGER MMGBSA implementation on the command line through python
    NB: Currently this method is only available to linux users
    Please ensure you have set the following environment variable:
    >>>export SCHRODINGER='<path_to_SCHRODINGER_HOME>'
    
    returns optimized protein, optimized ligand, and csv file of mmgbsa scores
    """
    def __init__(self):
        self.temp = "mmgbsa_dir/"
        # here assuming the input will be named complex.pdb
        self.input = "complex"
        self.output = "mmgbsa_out"

    def run(self, complex_, job_type="SITE_OPT"):
        """
        executes the command line call
        :param protein-ligand to be optimized and scored: complex
        :int target flexibility cutoff: degree of freedom as we optimize the ligand in the pocket
        :type job type: EX: ENERGY,REAL_MIN,SIDE_PRED,SIDE_COMBI,SITE_OPT,PGL, etc...
        """
        if not os.path.isdir(f"./{self.temp}"):
            os.makedirs(f"./{self.temp}")
        os.chdir(f"./{self.temp}")

        cmd0 = f"{os.environ['SCHRODINGER']}/utilities/prepwizard -NOJOBID -disulfides -mse -fillsidechains -fillloops -propka_pH 7 -rehtreat -rmsd 5.0 ../{self.input}.pdb prepped_{self.input}.pdb"
        os.system(cmd0)
        cmd1 = f"{os.environ['SCHRODINGER']}/utilities/structconvert -ipdb prepped_{self.input}.pdb -omae pv_{self.input}.mae"
        os.system(cmd1)
        cmd2 = f"{os.environ['SCHRODINGER']}/run pv_convert.py pv_{self.input}.mae -mode split_pv -lig_last_mol"
        os.system(cmd2)
        #ENERGY,REAL_MIN,SIDE_PRED,SIDE_COMBI,SITE_OPT,PGL
        cmd3 = f"{os.environ['SCHRODINGER']}/prime_mmgbsa pv_{self.input}-out_pv.mae -WAIT -NOJOBID -job_type {job_type} -target_flexibility -target_flexibility_cutoff 20 -out_type COMPLETE"
        os.system(cmd3)
        cmd4 = f"{os.environ['SCHRODINGER']}/utilities/structconvert -imae pv_{self.input}-out-out.maegz -opdb pv_{self.output}-out-out.pdb"
        os.system(cmd4)

        os.chdir("../")
        
        return os.path.join(self.temp, f"pv_{self.output}-out-out-1.pdb"), os.path.join(self.temp, f"pv_{self.output}-out-out-3.pdb"),\
                                os.path.join(self.temp, f"pv_{self.input}-out-out.csv")

    @staticmethod # for now
    def extract_results(csv_path):
        """
        returns the df of mmgbsa results.TODO: parse the column names.
        """

        try:
            mmgbsa=pd.read_csv(csv_path, sep=",")
            if len(mmgbsa)==3:
                mmgbsa=mmgbsa.drop("title", axis=1)
                mmgbsa=mmgbsa.drop(index=[0,2],axis=1)
                #mmgbsa.to_csv(f"{str(sys.argv[1]).rsplit( ".", 1 )[ 0 ]}_cleaned.csv",index=False)
            elif len(mmgbsa)==1:
                mmgbsa=mmgbsa.drop("title", axis=1)
        except:
            print("no mmgbsa score file found!")
            
        return mmgbsa
