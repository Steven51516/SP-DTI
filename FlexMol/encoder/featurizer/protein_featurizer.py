from sklearn.preprocessing import OneHotEncoder
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel, BertTokenizer, BertModel, AlbertTokenizer, AlbertModel
from itertools import zip_longest
import re
from FlexMol.util.biochem.protein.subpockets import *
from FlexMol.util.biochem.BPEEncoder import BPEEncoder
from FlexMol.util.biochem.pybiomed_helper import _GetPseudoAAC, CalculateAADipeptideComposition, \
CalculateConjointTriad, GetQuasiSequenceOrder, CalculateCTD, CalculateAutoTotal

from .base import Featurizer

# several methods adapted from https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/utils.py

class ProteinOneHotFeaturizer(Featurizer):
    def __init__(self, max_seq = 1000):
        super(ProteinOneHotFeaturizer, self).__init__()
        amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
                      'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']
        self.onehot_enc = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))
        self.amino_char = amino_char
        self.transform_modes["initial"] = self.initial_transform
        self.transform_modes["loadtime"] = self.loadtime_transform
        self.max_seq = max_seq

    def initial_transform(self, x):
        temp = list(x.upper())
        temp = [i if i in self.amino_char else '?' for i in temp]
        if len(temp) <  self.max_seq:
            temp = temp + ['?'] * ( self.max_seq - len(temp))
        else:
            temp = temp[: self.max_seq]
        return temp

    def loadtime_transform(self, x):
        return self.onehot_enc.transform(np.array(x).reshape(-1, 1)).toarray().T

    def transform(self, x):
        x = self.initial_transform(x)
        return self.loadtime_transform(x)


class ProteinAACFeaturizer(Featurizer):
    def transform(self, x):
        try:
            features = CalculateAADipeptideComposition(x)
        except:
            print('AAC fingerprint not working for protein: ' + x + ' convert to 0 vectors')
            features = np.zeros((8420, ))
        return np.array(features)


class ProteinGraphFeaturizer(Featurizer):
    def __init__(self, data_dir):
        super(ProteinGraphFeaturizer, self).__init__()
        from FlexMol.util.biochem.protein.prot_graph import create_prot_dgl_graph
        self.data_dir = data_dir
        self._transform = create_prot_dgl_graph

    def transform(self, x):
        return self._transform(x, self.data_dir)
    

class ProteinGraphESMFeaturizer(Featurizer):
    def __init__(self, data_dir):
        super(ProteinGraphESMFeaturizer, self).__init__()
        from FlexMol.util.biochem.protein.prot_graph import create_prot_esm_dgl_graph
        self.data_dir = data_dir
        self._transform = create_prot_esm_dgl_graph

    def transform(self, x):
        return self._transform(x, self.data_dir)



class SubpocketFeaturizer(Featurizer):
    def __init__(self, pdb_dir, subpocket_dir, pocket_num = 30):
        super(SubpocketFeaturizer, self).__init__()
        self.max_pockets = pocket_num
        self.pdb_dir = pdb_dir
        self.subpocket_dir = subpocket_dir
    def transform(self, x):
        return process_protein_subpocket(x, self.pdb_dir, self.subpocket_dir, self.max_pockets)


