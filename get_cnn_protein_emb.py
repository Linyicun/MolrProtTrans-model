from DeepPurpose.utils import trans_protein, protein_2_embed
from DeepPurpose import utils
from DeepPurpose.DTI import CNN
import torch
config = utils.generate_config(
                         target_encoding = 'CNN',
                         cnn_target_filters = [32,64,96],
                         cnn_target_kernels = [4,8,12]
                        )
print("config=",config)
model_protein = CNN('protein', **config)

f=open("../data/GPCR_train.txt","r")
for line in f.readlines():
    smiles,protein,label=line.split()
    p=trans_protein(protein)
    p=protein_2_embed(p)
    p=torch.tensor(p)
    p=p.unsqueeze(dim=0)
    print(p.shape)
    emb=model_protein(p)
    emb=emb[0]

    print("emb=",emb)
    print(emb.shape)
    break
