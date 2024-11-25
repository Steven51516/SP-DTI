
from FlexMol.util.biochem.BPEEncoder import BPEEncoder
from .base import Featurizer
from sklearn.preprocessing import OneHotEncoder
from functools import partial
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdReducedGraphs
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_complete_graph, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
import torch
from FlexMol.util.biochem.pybiomed_helper import calcPubChemFingerAll
import torch.nn as nn
from tokenizers.models import WordLevel
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import AutoTokenizer, AutoModel
import re


# several methods adapted from https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/utils.py

class DrugOneHotFeaturizer(Featurizer):
    def __init__(self, max_seq = 100):
        super(DrugOneHotFeaturizer, self).__init__()
        smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
                       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
                       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
                       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
                       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']
        self.onehot_enc = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))
        self.smiles_char = smiles_char
        self.transform_modes["initial"] = self.initial_transform
        self.transform_modes["loadtime"] = self.loadtime_transform
        self.max_seq = 100

    def initial_transform(self, x):
        temp = list(x)
        temp = [i if i in self.smiles_char else '?' for i in temp]
        if len(temp) <  self.max_seq:
            temp = temp + ['?'] * (self.max_seq - len(temp))
        else:
            temp = temp[:self.max_seq]
        return temp

    def loadtime_transform(self, x):
        return self.onehot_enc.transform(np.array(x).reshape(-1, 1)).toarray().T

    def transform(self, x):
        x = self.initial_transform(x)
        return self.loadtime_transform(x)



class BaseGraphFeaturizer(Featurizer):
    def __init__(self, virtual_nodes=False, max_node=50):
        super().__init__()
        self.virtual_nodes = virtual_nodes
        self.max_node = max_node

    def add_virtual_nodes(self, graph):
        actual_node_feats = graph.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_node - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        graph.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, actual_node_feats.shape[1] - 1), torch.ones(num_virtual_nodes, 1)), 1)
        graph.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        graph = graph.add_self_loop()
        return graph



class DrugCanonicalFeaturizer(BaseGraphFeaturizer):
    def __init__(self, virtual_nodes=False, max_node=50):
        super().__init__(virtual_nodes, max_node)
        self.node_featurizer = CanonicalAtomFeaturizer()
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.transform_func = partial(smiles_to_bigraph,
                                      node_featurizer=self.node_featurizer,
                                      edge_featurizer=self.edge_featurizer,
                                      add_self_loop=True)

    def transform(self, x):
        graph = self.transform_func(x)
        if self.virtual_nodes:
            graph = self.add_virtual_nodes(graph)
        return graph




class MorganFeaturizer(Featurizer):
    def __init__(self, radius=2, nbits=1024):
        super().__init__()
        self.radius = radius
        self.nBits = nbits

    def transform(self, s):
        try:
            mol = Chem.MolFromSmiles(s)
            features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.nBits)
            features = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, features)
        except:
            print('rdkit not found this smiles for morgan: ' + s + ' convert to all 0 features')
            features = np.zeros((self.nBits,))
        return features



class ChemBERTaFeaturizer(Featurizer):
    """Featurizer using ChemBERTa model to generate molecular embeddings."""
    def __init__(self, model_path="DeepChem/ChemBERTa-77M-MTR", mode='mean'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model._modules["lm_head"] = nn.Identity() 
        self.mode = mode

    def transform(self, smiles):
        """Transforms a SMILES string into a ChemBERTa embedding."""
        try:
            encoded_input = self.tokenizer(smiles, return_tensors="pt")
            model_output = self.model(**encoded_input)
            if self.mode == 'cls':
                # Use the embedding from the CLS token
                embedding = model_output[0][:, 0, :]  # CLS token is the first token
            elif self.mode == 'mean':
                # Calculate the mean of all token embeddings
                embedding = torch.mean(model_output[0], dim=1)
            else:
                raise ValueError("Unsupported mode. Choose 'cls' or 'mean'.")
            return embedding.squeeze().tolist()
        except Exception as e:
            print(f'Error processing SMILES {smiles}: {str(e)}')
            return []


class DrugChemBertFeaturizer(Featurizer):
    def __init__(self):
        super().__init__()
        model_name = "/root/sliu/BioEncoder/ChemBERTa-77M-MLM"
        new_tokenizer = Tokenizer(
            WordLevel.from_file(
                '/root/sliu/BioEncoder/ChemBERTa-77M-MLM/vocab.json', 
                unk_token='[UNK]'
        ))

        from tokenizers import Regex
        self.pre_tokenizer = Split(
            pattern=Regex("\[(.*?)\]|.*?"),
            behavior='isolated'
        )
        new_tokenizer.pre_tokenizer = self.pre_tokenizer
        self.tokenizer = new_tokenizer

        self.model = AutoModel.from_pretrained(model_name)
        self.atom_finder = re.compile(r"""
        (
         Cl? |             # Cl and Br are part of the organic subset
         Br? |
         Mg? |
         [NOSPFIbcnosp*] | # as are these single-letter elements
         \[[^]]+\]         # everything else must be in []s
        )
        """, re.X)

    def transform(self, x):

        vocab = {"[PAD]":0,"[unused1]":1,"[unused2]":2,"[unused3]":3,"[unused4]":4,"[unused5]":5,"[unused6]":6,"[unused7]":7,"[unused8]":8,"[unused9]":9,"[unused10]":10,"[UNK]":11,"[CLS]":12,"[SEP]":13,"[MASK]":14,"c":15,"C":16,"(":17,")":18,"O":19,"1":20,"2":21,"=":22,"N":23,".":24,"n":25,"3":26,"F":27,"Cl":28,">>":29,"~":30,"-":31,"4":32,"[C@H]":33,"S":34,"[C@@H]":35,"[O-]":36,"Br":37,"#":38,"/":39,"[nH]":40,"[N+]":41,"s":42,"5":43,"o":44,"P":45,"[Na+]":46,"[Si]":47,"I":48,"[Na]":49,"[Pd]":50,"[K+]":51,"[K]":52,"[P]":53,"B":54,"[C@]":55,"[C@@]":56,"[Cl-]":57,"6":58,"[OH-]":59,"\\":60,"[N-]":61,"[Li]":62,"[H]":63,"[2H]":64,"[NH4+]":65,"[c-]":66,"[P-]":67,"[Cs+]":68,"[Li+]":69,"[Cs]":70,"[NaH]":71,"[H-]":72,"[O+]":73,"[BH4-]":74,"[Cu]":75,"7":76,"[Mg]":77,"[Fe+2]":78,"[n+]":79,"[Sn]":80,"[BH-]":81,"[Pd+2]":82,"[CH]":83,"[I-]":84,"[Br-]":85,"[C-]":86,"[Zn]":87,"[B-]":88,"[F-]":89,"[Al]":90,"[P+]":91,"[BH3-]":92,"[Fe]":93,"[C]":94,"[AlH4]":95,"[Ni]":96,"[SiH]":97,"8":98,"[Cu+2]":99,"[Mn]":100,"[AlH]":101,"[nH+]":102,"[AlH4-]":103,"[O-2]":104,"[Cr]":105,"[Mg+2]":106,"[NH3+]":107,"[S@]":108,"[Pt]":109,"[Al+3]":110,"[S@@]":111,"[S-]":112,"[Ti]":113,"[Zn+2]":114,"[PH]":115,"[NH2+]":116,"[Ru]":117,"[Ag+]":118,"[S+]":119,"[I+3]":120,"[NH+]":121,"[Ca+2]":122,"[Ag]":123,"9":124,"[Os]":125,"[Se]":126,"[SiH2]":127,"[Ca]":128,"[Ti+4]":129,"[Ac]":130,"[Cu+]":131,"[S]":132,"[Rh]":133,"[Cl+3]":134,"[cH-]":135,"[Zn+]":136,"[O]":137,"[Cl+]":138,"[SH]":139,"[H+]":140,"[Pd+]":141,"[se]":142,"[PH+]":143,"[I]":144,"[Pt+2]":145,"[C+]":146,"[Mg+]":147,"[Hg]":148,"[W]":149,"[SnH]":150,"[SiH3]":151,"[Fe+3]":152,"[NH]":153,"[Mo]":154,"[CH2+]":155,"%10":156,"[CH2-]":157,"[CH2]":158,"[n-]":159,"[Ce+4]":160,"[NH-]":161,"[Co]":162,"[I+]":163,"[PH2]":164,"[Pt+4]":165,"[Ce]":166,"[B]":167,"[Sn+2]":168,"[Ba+2]":169,"%11":170,"[Fe-3]":171,"[18F]":172,"[SH-]":173,"[Pb+2]":174,"[Os-2]":175,"[Zr+4]":176,"[N]":177,"[Ir]":178,"[Bi]":179,"[Ni+2]":180,"[P@]":181,"[Co+2]":182,"[s+]":183,"[As]":184,"[P+3]":185,"[Hg+2]":186,"[Yb+3]":187,"[CH-]":188,"[Zr+2]":189,"[Mn+2]":190,"[CH+]":191,"[In]":192,"[KH]":193,"[Ce+3]":194,"[Zr]":195,"[AlH2-]":196,"[OH2+]":197,"[Ti+3]":198,"[Rh+2]":199,"[Sb]":200,"[S-2]":201,"%12":202,"[P@@]":203,"[Si@H]":204,"[Mn+4]":205,"p":206,"[Ba]":207,"[NH2-]":208,"[Ge]":209,"[Pb+4]":210,"[Cr+3]":211,"[Au]":212,"[LiH]":213,"[Sc+3]":214,"[o+]":215,"[Rh-3]":216,"%13":217,"[Br]":218,"[Sb-]":219,"[S@+]":220,"[I+2]":221,"[Ar]":222,"[V]":223,"[Cu-]":224,"[Al-]":225,"[Te]":226,"[13c]":227,"[13C]":228,"[Cl]":229,"[PH4+]":230,"[SiH4]":231,"[te]":232,"[CH3-]":233,"[S@@+]":234,"[Rh+3]":235,"[SH+]":236,"[Bi+3]":237,"[Br+2]":238,"[La]":239,"[La+3]":240,"[Pt-2]":241,"[N@@]":242,"[PH3+]":243,"[N@]":244,"[Si+4]":245,"[Sr+2]":246,"[Al+]":247,"[Pb]":248,"[SeH]":249,"[Si-]":250,"[V+5]":251,"[Y+3]":252,"[Re]":253,"[Ru+]":254,"[Sm]":255,"*":256,"[3H]":257,"[NH2]":258,"[Ag-]":259,"[13CH3]":260,"[OH+]":261,"[Ru+3]":262,"[OH]":263,"[Gd+3]":264,"[13CH2]":265,"[In+3]":266,"[Si@@]":267,"[Si@]":268,"[Ti+2]":269,"[Sn+]":270,"[Cl+2]":271,"[AlH-]":272,"[Pd-2]":273,"[SnH3]":274,"[B+3]":275,"[Cu-2]":276,"[Nd+3]":277,"[Pb+3]":278,"[13cH]":279,"[Fe-4]":280,"[Ga]":281,"[Sn+4]":282,"[Hg+]":283,"[11CH3]":284,"[Hf]":285,"[Pr]":286,"[Y]":287,"[S+2]":288,"[Cd]":289,"[Cr+6]":290,"[Zr+3]":291,"[Rh+]":292,"[CH3]":293,"[N-3]":294,"[Hf+2]":295,"[Th]":296,"[Sb+3]":297,"%14":298,"[Cr+2]":299,"[Ru+2]":300,"[Hf+4]":301,"[14C]":302,"[Ta]":303,"[Tl+]":304,"[B+]":305,"[Os+4]":306,"[PdH2]":307,"[Pd-]":308,"[Cd+2]":309,"[Co+3]":310,"[S+4]":311,"[Nb+5]":312,"[123I]":313,"[c+]":314,"[Rb+]":315,"[V+2]":316,"[CH3+]":317,"[Ag+2]":318,"[cH+]":319,"[Mn+3]":320,"[Se-]":321,"[As-]":322,"[Eu+3]":323,"[SH2]":324,"[Sm+3]":325,"[IH+]":326,"%15":327,"[OH3+]":328,"[PH3]":329,"[IH2+]":330,"[SH2+]":331,"[Ir+3]":332,"[AlH3]":333,"[Sc]":334,"[Yb]":335,"[15NH2]":336,"[Lu]":337,"[sH+]":338,"[Gd]":339,"[18F-]":340,"[SH3+]":341,"[SnH4]":342,"[TeH]":343,"[Si@@H]":344,"[Ga+3]":345,"[CaH2]":346,"[Tl]":347,"[Ta+5]":348,"[GeH]":349,"[Br+]":350,"[Sr]":351,"[Tl+3]":352,"[Sm+2]":353,"[PH5]":354,"%16":355,"[N@@+]":356,"[Au+3]":357,"[C-4]":358,"[Nd]":359,"[Ti+]":360,"[IH]":361,"[N@+]":362,"[125I]":363,"[Eu]":364,"[Sn+3]":365,"[Nb]":366,"[Er+3]":367,"[123I-]":368,"[14c]":369,"%17":370,"[SnH2]":371,"[YH]":372,"[Sb+5]":373,"[Pr+3]":374,"[Ir+]":375,"[N+3]":376,"[AlH2]":377,"[19F]":378,"%18":379,"[Tb]":380,"[14CH]":381,"[Mo+4]":382,"[Si+]":383,"[BH]":384,"[Be]":385,"[Rb]":386,"[pH]":387,"%19":388,"%20":389,"[Xe]":390,"[Ir-]":391,"[Be+2]":392,"[C+4]":393,"[RuH2]":394,"[15NH]":395,"[U+2]":396,"[Au-]":397,"%21":398,"%22":399,"[Au+]":400,"[15n]":401,"[Al+2]":402,"[Tb+3]":403,"[15N]":404,"[V+3]":405,"[W+6]":406,"[14CH3]":407,"[Cr+4]":408,"[ClH+]":409,"b":410,"[Ti+6]":411,"[Nd+]":412,"[Zr+]":413,"[PH2+]":414,"[Fm]":415,"[N@H+]":416,"[RuH]":417,"[Dy+3]":418,"%23":419,"[Hf+3]":420,"[W+4]":421,"[11C]":422,"[13CH]":423,"[Er]":424,"[124I]":425,"[LaH]":426,"[F]":427,"[siH]":428,"[Ga+]":429,"[Cm]":430,"[GeH3]":431,"[IH-]":432,"[U+6]":433,"[SeH+]":434,"[32P]":435,"[SeH-]":436,"[Pt-]":437,"[Ir+2]":438,"[se+]":439,"[U]":440,"[F+]":441,"[BH2]":442,"[As+]":443,"[Cf]":444,"[ClH2+]":445,"[Ni+]":446,"[TeH3]":447,"[SbH2]":448,"[Ag+3]":449,"%24":450,"[18O]":451,"[PH4]":452,"[Os+2]":453,"[Na-]":454,"[Sb+2]":455,"[V+4]":456,"[Ho+3]":457,"[68Ga]":458,"[PH-]":459,"[Bi+2]":460,"[Ce+2]":461,"[Pd+3]":462,"[99Tc]":463,"[13C@@H]":464,"[Fe+6]":465,"[c]":466,"[GeH2]":467,"[10B]":468,"[Cu+3]":469,"[Mo+2]":470,"[Cr+]":471,"[Pd+4]":472,"[Dy]":473,"[AsH]":474,"[Ba+]":475,"[SeH2]":476,"[In+]":477,"[TeH2]":478,"[BrH+]":479,"[14cH]":480,"[W+]":481,"[13C@H]":482,"[AsH2]":483,"[In+2]":484,"[N+2]":485,"[N@@H+]":486,"[SbH]":487,"[60Co]":488,"[AsH4+]":489,"[AsH3]":490,"[18OH]":491,"[Ru-2]":492,"[Na-2]":493,"[CuH2]":494,"[31P]":495,"[Ti+5]":496,"[35S]":497,"[P@@H]":498,"[ArH]":499,"[Co+]":500,"[Zr-2]":501,"[BH2-]":502,"[131I]":503,"[SH5]":504,"[VH]":505,"[B+2]":506,"[Yb+2]":507,"[14C@H]":508,"[211At]":509,"[NH3+2]":510,"[IrH]":511,"[IrH2]":512,"[Rh-]":513,"[Cr-]":514,"[Sb+]":515,"[Ni+3]":516,"[TaH3]":517,"[Tl+2]":518,"[64Cu]":519,"[Tc]":520,"[Cd+]":521,"[1H]":522,"[15nH]":523,"[AlH2+]":524,"[FH+2]":525,"[BiH3]":526,"[Ru-]":527,"[Mo+6]":528,"[AsH+]":529,"[BaH2]":530,"[BaH]":531,"[Fe+4]":532,"[229Th]":533,"[Th+4]":534,"[As+3]":535,"[NH+3]":536,"[P@H]":537,"[Li-]":538,"[7NaH]":539,"[Bi+]":540,"[PtH+2]":541,"[p-]":542,"[Re+5]":543,"[NiH]":544,"[Ni-]":545,"[Xe+]":546,"[Ca+]":547,"[11c]":548,"[Rh+4]":549,"[AcH]":550,"[HeH]":551,"[Sc+2]":552,"[Mn+]":553,"[UH]":554,"[14CH2]":555,"[SiH4+]":556,"[18OH2]":557,"[Ac-]":558,"[Re+4]":559,"[118Sn]":560,"[153Sm]":561,"[P+2]":562,"[9CH]":563,"[9CH3]":564,"[Y-]":565,"[NiH2]":566,"[Si+2]":567,"[Mn+6]":568,"[ZrH2]":569,"[C-2]":570,"[Bi+5]":571,"[24NaH]":572,"[Fr]":573,"[15CH]":574,"[Se+]":575,"[At]":576,"[P-3]":577,"[124I-]":578,"[CuH2-]":579,"[Nb+4]":580,"[Nb+3]":581,"[MgH]":582,"[Ir+4]":583,"[67Ga+3]":584,"[67Ga]":585,"[13N]":586,"[15OH2]":587,"[2NH]":588,"[Ho]":589,"[Cn]":590}
        tokenized_inputs = self.tokenizer.encode(x)
        token_ids = tokenized_inputs.ids
        token_ids = [vocab['[CLS]']] + token_ids + [vocab['[SEP]']]
        attention_mask = [1] + tokenized_inputs.attention_mask + [1]

        inputs = {
            'input_ids': torch.tensor([token_ids]),  # Wrap in a list to add batch dimension
            'attention_mask': torch.tensor([attention_mask ])  # Wrap in a list to add batch dimension
        }

        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state[:, 1:-1, :][0].detach()
        is_atom_array = []
        output= self.tokenizer.pre_tokenizer.pre_tokenize_str(x)
        for token, _ in output:
            if  token == "H" or token == "[H]":
                is_atom_array.append(False)
            elif self.atom_finder.match(token):
                is_atom_array.append(True)
            else:
                is_atom_array.append(False)

        mean_embeddings = []
        for i, index in enumerate(is_atom_array):
            if index:
                range_embeddings = last_hidden_states[i]
                mean_embeddings.append(range_embeddings)

        mean_embeddings_tensor = torch.stack(mean_embeddings)
        return mean_embeddings_tensor




class DrugChemBertGNNFeaturizer(Featurizer):
    def __init__(self):
        super().__init__()
        self.node_featurizer = CanonicalAtomFeaturizer()
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.transform_func = partial(smiles_to_bigraph,
                                      canonical_atom_order=False,
                                      node_featurizer=self.node_featurizer,
                                      edge_featurizer=self.edge_featurizer,
                                      add_self_loop=True)
        self.bert = DrugChemBertFeaturizer()

    def transform(self, x):
        mol = Chem.MolFromSmiles(x)
        graph = self.transform_func(x)
        bert_embedding = self.bert(x)
        graph.ndata['bert'] = bert_embedding
        return graph