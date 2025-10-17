import torch
import datamol as dm
from torch_geometric.data import Data
import numpy as np
from rdkit.Chem import Draw

ATOM_LIST = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]

def atom_features(atom):
    Z = atom.GetAtomicNum()
    onehot = [1 if Z == a else 0 for a in ATOM_LIST]
    return torch.tensor(onehot + [atom.GetTotalDegree(), atom.GetTotalValence(), 1 if atom.GetIsAromatic() else 0], dtype=torch.float)

def smiles_to_graph(smiles):
    mol = dm.to_mol(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    x = torch.stack([atom_features(a) for a in mol.GetAtoms()])
    edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    edges += [[b[1], b[0]] for b in edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data

def compute_descriptors(smiles):
    mol = dm.to_mol(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    return {
        "MolWt": dm.mol.mol_weight(mol),
        "LogP": dm.mol.logp(mol),
        "HDonors": dm.mol.num_hdonors(mol),
        "HAcceptors": dm.mol.num_hacceptors(mol),
        "TPSA": dm.mol.tpsa(mol),
        "RotatableBonds": dm.mol.num_rotatable_bonds(mol)
    }

def draw_molecule(smiles):
    mol = dm.to_mol(smiles)
    return Draw.MolToImage(mol, size=(300, 300))

def tanimoto_similarity(smiles1, smiles2):
    mol1, mol2 = dm.to_mol(smiles1), dm.to_mol(smiles2)
    fp1, fp2 = dm.fingerprint.morgan_fp(mol1), dm.fingerprint.morgan_fp(mol2)
    return float(np.dot(fp1, fp2) / (np.linalg.norm(fp1) * np.linalg.norm(fp2)))
