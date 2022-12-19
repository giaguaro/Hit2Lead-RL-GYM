{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "28be4ca2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "from utilties.wrapper_schrodinger import SCHROD_MMGBSA\n",
    "\n",
    "\n",
    "license_filename = 'oe_license_1.txt'\n",
    "import openeye\n",
    "\n",
    "import os\n",
    "if os.path.isfile(license_filename):\n",
    "    license_file = open(license_filename, 'r')\n",
    "    openeye.OEAddLicenseData(license_file.read())\n",
    "    license_file.close()\n",
    "else:\n",
    "    print(\"Error: Your OpenEye license is not readable; please check your filename and that you have mounted your Google Drive\")\n",
    "\n",
    "licensed = openeye.oechem.OEChemIsLicensed()\n",
    "print(\"Was your OpenEye license correctly installed (True/False)? \" + str(licensed))\n",
    "if not licensed:\n",
    "    print(\"Error: Your OpenEye license is not correctly installed.\")\n",
    "    raise Exception(\"Error: Your OpenEye license is not correctly installed.\")\n",
    "    \n",
    "    \n",
    "class Ligand_Gym():\n",
    "\n",
    "    class Settings(object):\n",
    "        \"\"\"\n",
    "        adjusts the default settings for the ligand optimization\n",
    "        :param int something\n",
    "        :param float soemthing .Default ..\n",
    "        :param bool foo: description\n",
    "        \"\"\"\n",
    "\n",
    "        def __init__(self, setting1=, setting2=, etc):\n",
    "            self.setting1=setting1\n",
    "\n",
    "\n",
    "    def __init__(self, settings=None):\n",
    "        \n",
    "        if settings is None:\n",
    "            self.sampler_settings = self.Settings()\n",
    "        else:\n",
    "            self.sampler_settings = settings\n",
    "\n",
    "    self.mmgbsa_method = mmgbsa_method\n",
    "\n",
    "\n",
    "    @property\n",
    "    def mmgbsa_method(self):\n",
    "        \"\"\"\n",
    "        calculate MM-GBSA and optimize ligand provided a 3D complex of protein-ligand\n",
    "        :return:\n",
    "        \"\"\"\n",
    "        return self._mmgbsa_method\n",
    "\n",
    "    @mmgbsa_method.setter\n",
    "    def mmgbsa_method(self, method):\n",
    "        \"\"\"\n",
    "        leave option to change mmgbsa method in case we need to update the code in the future\n",
    "        :return:\n",
    "        \"\"\"\n",
    "        method = method.lower()\n",
    "        if method == 'schrodinger':\n",
    "            if sys.platform == 'linux' or sys.platform == 'linux2':\n",
    "                if 'SCHRODINGER' in environ:\n",
    "                    self._mmgbsa_method = method\n",
    "                else:\n",
    "                    raise EnvironmentError(\"Must set SCHRODINGER environment variable\")\n",
    "            else:\n",
    "                raise OSError('SCHRODINGER is only supported on linux')\n",
    "\n",
    "        else:\n",
    "            raise ValueError(\"Provided method is not supported yet come back in an minute\")\n",
    "\n",
    "    def calc_mmgbsa(self, method):\n",
    "        \"\"\"\n",
    "        Do actual calculations in the linux environment \n",
    "        :return:\n",
    "        \"\"\"\n",
    "        mmgbsa = SCHROD_MMGBSA()\n",
    "\n",
    "        # here we supply the path of the complex (PDB)\n",
    "        self.mmgbsa_prot_optimized, self.mmgbsa_lig_optimized, csv_out = mmgbsa.run(self.complex)\n",
    "        # get optimized complex after mmgbsa calculations and extract scores from the generated csv file\n",
    "        self.mmgbsa_scores = mmgbsa.extract_results(csv_out)\n",
    "        # remove any extra files generated\n",
    "        shutil.rmtree(mmgbsa.temp) \n",
    "        \n",
    "        print(\"MMGBSA complete\\n\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
