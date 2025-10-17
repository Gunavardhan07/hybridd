from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors
import torch
from torch_geometric.data import Data
from rdkit.Chem import AllChem
import numpy as np

ATOM_LIST = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]

def atom_features(atom):
    Z = atom.GetAtomicNum()
    onehot = [1 if Z == a else 0 for a in ATOM_LIST]
    return torch.tensor(onehot + [
        atom.GetTotalDegree(),
        atom.GetTotalValence(),
        1 if atom.GetIsAromatic() else 0
    ], dtype=torch.float)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    x = torch.stack([atom_features(a) for a in mol.GetAtoms()])
    edges = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[a1, a2], [a2, a1]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data

# ---------------- Molecular Descriptors ---------------- #
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    desc = {
        "MolecularWeight": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol)
    }
    return desc

# ---------------- Molecular Similarity ---------------- #
def tanimoto_similarity(smiles1, smiles2):
    mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
    fp1, fp2 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048), AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return DataStructs.FingerprintSimilarity(fp1, fp2)

# ---------------- Graph Visualization ---------------- #
def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=(300,300))
    return img
