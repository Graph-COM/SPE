import os
import os.path as osp
import pickle
import shutil
from typing import Callable, List, Optional
from rdkit import Chem
import rdkit
import mmcv
import torch
from tqdm import tqdm
import numpy as np

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


BOND_TYPE_TO_FEAT = {Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2,
                     Chem.rdchem.BondType.TRIPLE : 3}

ATOM_TYPE_TO_FEAT = {1: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 11: 6, 14: 7, 15: 8, 16: 9, 17: 10, 19: 11, 20: 12, 30: 13,
                     33: 14, 34: 15, 35: 16, 47: 17, 53: 18, 3: 19}

class DrugOOD(InMemoryDataset):
    def __init__(
        self,
        root: str,
        curator: str,
        split: str = 'train',
        processed_suffix: str = '',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.root = root
        self.curator = curator
        self.processed_name = 'processed'+processed_suffix
        assert split in ['train', 'iid_val', 'iid_test', 'ood_val', 'ood_test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            self.curator + '.json'
        ]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.processed_name)

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'iid_val.pt', 'iid_test.pt', 'ood_val.pt', 'ood_test.pt']

    def process(self):
        Smile2Graph = SmileToGraph([])
        raw_data = mmcv.load(osp.join(self.raw_dir, self.raw_file_names[0]))
        for split in ['train', 'iid_val', 'iid_test', 'ood_val', 'ood_test']:
            raw_data_list = raw_data['split'][split]

            pbar = tqdm(total=len(raw_data_list))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in range(len(raw_data_list)):
                data = raw_data_list[idx]
                # x, edge_index, edge_attr = smiles_to_graph(data["smiles"])
                x, edge_index, edge_attr = Smile2Graph.smile2graph(data["smiles"])

                y = torch.tensor(data["cls_label"]).to(torch.float)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))



def smiles_to_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol) # get rid of aromatic bonds
    atom_types = [ATOM_TYPE_TO_FEAT[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        edge_attr.append(BOND_TYPE_TO_FEAT[bond.GetBondType()])
        edge_attr.append(BOND_TYPE_TO_FEAT[bond.GetBondType()])

    atom_types = torch.tensor(atom_types).view(-1, 1)
    edge_index = torch.tensor(edge_index).T
    edge_attr = torch.tensor(edge_attr)

    # sanity check: comment once passed
    for atom in mol.GetAtoms():
        assert atom.GetDegree() <= 4
        #if atom.GetAtomicNum() == 1:
        #    print(smi)


    return atom_types, edge_index, edge_attr



class SmileToGraph(object):
    # borrowed from drugood
    """Transform smile input to graph format

    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = self.smile2graph(results[key])
        return results

    def get_atom_features(self, atom):
        # The usage of features is along with the Attentive FP.
        feature = np.zeros(39)
        # Symbol
        symbol = atom.GetSymbol()
        symbol_list = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At']
        if symbol in symbol_list:
            loc = symbol_list.index(symbol)
            feature[loc] = 1
        else:
            feature[15] = 1

        # Degree
        degree = atom.GetDegree()
        if degree > 5:
            print("atom degree larger than 5. Please check before featurizing.")
            raise RuntimeError

        feature[16 + degree] = 1

        # Formal Charge
        charge = atom.GetFormalCharge()
        feature[22] = charge

        # radical electrons
        radelc = atom.GetNumRadicalElectrons()
        feature[23] = radelc

        # Hybridization
        hyb = atom.GetHybridization()
        hybridization_list = [rdkit.Chem.rdchem.HybridizationType.SP,
                              rdkit.Chem.rdchem.HybridizationType.SP2,
                              rdkit.Chem.rdchem.HybridizationType.SP3,
                              rdkit.Chem.rdchem.HybridizationType.SP3D,
                              rdkit.Chem.rdchem.HybridizationType.SP3D2]
        if hyb in hybridization_list:
            loc = hybridization_list.index(hyb)
            feature[loc + 24] = 1
        else:
            feature[29] = 1

        # aromaticity
        if atom.GetIsAromatic():
            feature[30] = 1

        # hydrogens
        hs = atom.GetNumImplicitHs()
        feature[31 + hs] = 1

        # chirality, chirality type
        if atom.HasProp('_ChiralityPossible'):
            # TODO what kind of error
            feature[36] = 1

            try:
                chi = atom.GetProp('_CIPCode')
                chi_list = ['R', 'S']
                loc = chi_list.index(chi)
                feature[37 + loc] = 1
            except KeyError:
                feature[37] = 0
                feature[38] = 0

        return feature

    def get_bond_features(self, bond):
        feature = np.zeros(10)

        # bond type
        type = bond.GetBondType()
        bond_type_list = [rdkit.Chem.rdchem.BondType.SINGLE,
                          rdkit.Chem.rdchem.BondType.DOUBLE,
                          rdkit.Chem.rdchem.BondType.TRIPLE,
                          rdkit.Chem.rdchem.BondType.AROMATIC]
        if type in bond_type_list:
            loc = bond_type_list.index(type)
            feature[0 + loc] = 1
        else:
            print("Wrong type of bond. Please check before feturization.")
            raise RuntimeError

        # conjugation
        conj = bond.GetIsConjugated()
        feature[4] = conj

        # ring
        ring = bond.IsInRing()
        feature[5] = ring

        # stereo
        stereo = bond.GetStereo()
        stereo_list = [rdkit.Chem.rdchem.BondStereo.STEREONONE,
                       rdkit.Chem.rdchem.BondStereo.STEREOANY,
                       rdkit.Chem.rdchem.BondStereo.STEREOZ,
                       rdkit.Chem.rdchem.BondStereo.STEREOE]
        if stereo in stereo_list:
            loc = stereo_list.index(stereo)
            feature[6 + loc] = 1
        else:
            print("Wrong stereo type of bond. Please check before featurization.")
            raise RuntimeError

        return feature

    def smile2graph(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if (mol is None):
            return None
        src = []
        dst = []
        atom_feature = []
        bond_feature = []

        try:
            for atom in mol.GetAtoms():
                one_atom_feature = self.get_atom_features(atom)
                atom_feature.append(one_atom_feature)

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                one_bond_feature = self.get_bond_features(bond)
                src.append(i)
                dst.append(j)
                bond_feature.append(one_bond_feature)
                src.append(j)
                dst.append(i)
                bond_feature.append(one_bond_feature)

            src = torch.tensor(src).long().view(1, -1)
            dst = torch.tensor(dst).long().view(1, -1)
            edge_index = torch.cat([src, dst], dim=0)
            atom_feature = torch.tensor(atom_feature).float()
            bond_feature = torch.tensor(bond_feature).float()
            return atom_feature, edge_index, bond_feature

        except RuntimeError:
            return None

    def featurize_atoms(self, mol):
        feats = []
        for atom in mol.GetAtoms():
            feats.append(atom.GetAtomicNum())
        return {'atomic': torch.tensor(feats).reshape(-1).to(torch.int64)}

    def featurize_bonds(self, mol):
        feats = []
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                      Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        for bond in mol.GetBonds():
            btype = bond_types.index(bond.GetBondType())
            # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
            feats.extend([btype, btype])
        return {'type': torch.tensor(feats).reshape(-1).to(torch.int64)}