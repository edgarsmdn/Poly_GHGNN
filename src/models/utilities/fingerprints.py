'''
Project: PolyGNN
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
-------------------------------------------------------------------------------
'''
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

def get_fp_ECFP_bitvector(mol, radius=4, nBits=2048, info={}):
    return GetMorganFingerprintAsBitVect(mol, radius, nBits, bitInfo=info) 

