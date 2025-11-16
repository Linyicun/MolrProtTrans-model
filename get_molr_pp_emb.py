import pysmiles
from Molr_src.data_processing import networkx_to_dgl
from Molr_src.model import GNN
import torch
import pickle
import csv

use_cuda=False
path="Molr_saved/gcn_1024/"

with open(path+'feature_enc.pkl', 'rb') as f:
    feature_encoder = pickle.load(f)

with open(path+'hparams.pkl', 'rb') as f:
    hparams = pickle.load(f)

    mole = GNN(hparams['gnn'], hparams['layer'], hparams['feature_len'], hparams['dim'])
    if torch.cuda.is_available() and use_cuda:
        mole.load_state_dict(torch.load( path + 'model.pt', map_location=torch.device('cuda:0')))
        mole = mole.cuda(0)
    else:
        mole.load_state_dict(torch.load(path + 'model.pt', map_location=torch.device('cpu')))

f=open("../data/GPCR_train.txt","r")
f_err=open("../data/GPCR_err.txt","w")
row_no=0
for line in f.readlines():
    row_no +=1
    smiles,protein,label=line.split()
    #smiles="[Cl].CC(C)NCC(O)COc1cccc2ccccc12"
    try:
        graph = pysmiles.read_smiles(smiles, zero_order_bonds=False)
        graph=networkx_to_dgl(graph,feature_encoder)

        if torch.cuda.is_available() and use_cuda:
            graph = graph.to("cuda:0")

        pred=mole(graph)
        print("smiles=",smiles)
        print("pred=",pred)

        linear=torch.nn.Linear(1024,256)
        pred2=linear(pred)

        print("pred2=",pred2)

        print()
    except:
        print("cant deal smiles: ",smiles,"row_no=",row_no)
        f_err.write(str(row_no)+smiles+"\n")
    
    break

    

f.close()
f_err.close()
