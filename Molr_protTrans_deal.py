from Molr_src.featurizer import MolEFeaturizer
from transformers import T5EncoderModel, T5Tokenizer
from rdkit import Chem

import torch

import numpy as np
import os

model_drug=MolEFeaturizer(path_to_model='D:/MolR-master/saved/gcn_1024')

model_protein = T5EncoderModel.from_pretrained("D:\DeepPurpose-master\protTrans\hugging_face")
tokenizer_protein_ProtTrans = T5Tokenizer.from_pretrained('D:\DeepPurpose-master\protTrans\hugging_face', do_lower_case=False)

DATASET = "GPCR_train_Molr_protTrans"
with open("../data/GPCR_train.txt","r") as f:
    data_list = f.read().strip().split('\n')
"""Exclude data contains '.' in the SMILES format."""
data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
N = len(data_list)

num_atom_feat = 34

def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)+np.eye(adjacency.shape[0])

def mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    #mol = Chem.AddHs(mol)
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    # try:
        # smiles_emb,flag = model_drug.transform([smiles])
        # smiles_emb=torch.tensor(smiles_emb)
        # drug_fc=torch.nn.Linear(1024,num_atom_feat)
        # smiles_emb=drug_fc(smiles_emb)
        # smiles_emb=smiles_emb.numpy()
        # smiles_emb=list(smiles_emb)
    # except:
    #     smiles_emb=[0]*num_atom_feat
    
    try:

        

        smiles_emb,flag = model_drug.transform([smiles])
        smiles_emb=torch.tensor(smiles_emb)
        smiles_emb=smiles_emb[0]
        print(smiles_emb.shape)
        drug_fc=torch.nn.Linear(1024,num_atom_feat,dtype=float)
        smiles_emb=drug_fc(smiles_emb)

        if(abs(torch.max(smiles_emb))>20):
            smiles_emb/=100

        smiles_emb=smiles_emb.detach().numpy()

        

        smiles_emb=list(smiles_emb)
    except Exception as e:
        print(str(e))
        smiles_emb=[0]*num_atom_feat
    
    print('smiles emb=',smiles_emb)

    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = smiles_emb
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix


def protTrans_embedding(sequence):
    token_encoding = tokenizer_protein_ProtTrans.batch_encode_plus(sequence, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(token_encoding['input_ids'])
    attention_mask = torch.tensor(token_encoding['attention_mask'])
    with torch.no_grad():
        # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
        embedding_repr = model_protein(input_ids, attention_mask=attention_mask).last_hidden_state
        # print('emb: ',embedding_repr)   
        # print('emb shape: ',embedding_repr.shape)
        protein_emb=embedding_repr.mean(dim=0).mean(dim=0)
        
        print('protein emb shape=',protein_emb.shape)
        protein_fc=torch.nn.Linear(1024,100)
        protein_emb=protein_fc(protein_emb)
    
    print('prot_emb=',protein_emb)
    protein_emb=list(protein_emb)
    total_protens=[]
    for i in range( int(len(sequence)/3)):
        total_protens.append(protein_emb)
    return total_protens


############################################################
compounds, adjacencies,proteins,interactions = [], [], [], []

cnt=0
limit_cnt=3000

for no, data in enumerate(data_list):
    # print('/'.join(map(str, [no + 1, N])))
    print('/'.join(map(str, [no + 1, limit_cnt])),'('+str(N)+')')
    smiles, sequence, interaction = data.strip().split(" ")

    atom_feature, adj = mol_features(smiles)
    compounds.append(atom_feature)
    adjacencies.append(adj)

    interactions.append(np.array([float(interaction)]))

    protein_embedding = protTrans_embedding(sequence)
    proteins.append(protein_embedding)
    
    cnt+=1
    print('cnt=',cnt)
    if cnt>=limit_cnt:
        print('break')
        break
dir_input = ('dataset/' + DATASET + '/word2vec_30/')
os.makedirs(dir_input, exist_ok=True)
for i in compounds:
    print(len(i))
np.save(dir_input + 'compounds', compounds)
np.save(dir_input + 'adjacencies', adjacencies)
np.save(dir_input + 'proteins', proteins)
np.save(dir_input + 'interactions', interactions)
print('The preprocess of ' + DATASET + ' dataset has finished!')
