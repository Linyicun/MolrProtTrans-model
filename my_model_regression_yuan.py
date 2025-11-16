from torch.utils.data import Dataset,DataLoader
from torch import nn
# import csv

import pysmiles
from Molr_src.data_processing import networkx_to_dgl
from Molr_src.model import GNN
import torch
import pickle

from DeepPurpose.utils import trans_protein, protein_2_embed
from DeepPurpose import utils
from DeepPurpose.DTI import CNN

from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc 
from scipy.stats import pearsonr


import sklearn

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score


class MyMolProtDataset(Dataset):
    def __init__(self,path) :
        self.path=path
        f=open(self.path,"r")
        self.molecules=[]
        self.proteins=[]
        self.labels=[]
        
        # print(self.path)
        # print(f)
        for line in f.readlines():
            #print("line=",line)
            line=line.split()
            self.molecules.append(line[0])
            self.proteins.append(line[1])
            
            self.labels.append(line[2])
        
    def __len__(self):
        return len(self.molecules)
    
    def __getitem__(self, idx) :
        return self.molecules[idx], self.proteins[idx], self.labels[idx]


class MyModel(nn.Sequential):
    def __init__(self):
        super(MyModel,self).__init__()
        self.use_cuda=False
        # mol_model_path="Molr_saved/gcn_1024/"
        mol_model_path="Molr_saved/gcn_1024Rhea/"
        with open(mol_model_path+'hparams.pkl', 'rb') as f:
            hparams = pickle.load(f)

            mole = GNN(hparams['gnn'], hparams['layer'], hparams['feature_len'], hparams['dim'])
            if torch.cuda.is_available() and self.use_cuda:
                mole.load_state_dict(torch.load( mol_model_path + 'model.pt', map_location=torch.device('cuda:0')))
                mole = mole.cuda(0)
            else:
                mole.load_state_dict(torch.load(mol_model_path + 'model.pt', map_location=torch.device('cpu')))

        self.mole_model=mole

        with open('Molr_saved/gcn_pp/feature_enc.pkl', 'rb') as f:
            self.feature_encoder = pickle.load(f)


        config = utils.generate_config(
                         target_encoding = 'CNN',
                         cnn_target_filters = [32,64,96],
                         cnn_target_kernels = [4,8,12]
                        )
        #print("config=",config)
        model_protein = CNN('protein', **config)

        self.protein_model=model_protein

        self.linear_smiles=nn.Linear(1024,256)
        
        self.fc1=nn.Linear(256+256,256)
        self.fc2=nn.Linear(256,64)
        self.fc3=nn.Linear(64,16)
        self.fc4=nn.Linear(16,1)
        
        self.acti=nn.Sigmoid()
      
    def forward(self,smiles,protein):
        graphs=[]
        smiles_embs=[]
        protein_embs=[]

        for s in smiles:
            
            try:
                graph = pysmiles.read_smiles(s, zero_order_bonds=False)
                graph=networkx_to_dgl(graph,self.feature_encoder)

                if torch.cuda.is_available() and self.use_cuda:
                    graph = graph.to("cuda:0")
                
                smiles_emb=self.mole_model(graph)
                smiles_emb=self.linear_smiles(smiles_emb)
                # smiles_emb=self.acti(smiles_emb)
                # emb2=[]
                # print("smiles emb",smiles_emb)
                # print(smiles_emb.shape)
                # for i in smiles_emb:
                #     print("i",i)
                #     if(abs(i)>20):
                #         emb2.append(i/100)
                #     else:
                #         emb2.append(i)
                
                # smiles_emb=torch.tensor(emb2)

                if(abs(torch.max(smiles_emb))>20):
                    smiles_emb/=100

                # print("s=",s)
                #print("smile_emb=",smiles_emb)

                smiles_embs.append(smiles_emb)
            except:
                #smiles_embs.append(torch.zeros(1,1024))
                smiles_embs.append(torch.zeros(1,256))

                
        for p in protein:
            
            p=trans_protein(p)
            p=protein_2_embed(p)
            p=torch.tensor(p)
            p=p.unsqueeze(dim=0)
            #print(p.shape)
            emb=self.protein_model(p)
            # protein_emb=emb
            protein_emb=emb*10
            protein_embs.append(protein_emb)
        
        smiles_embs=torch.cat(smiles_embs,dim=0)
        protein_embs=torch.cat(protein_embs,dim=0)
        
        #print("smiles embs=",smiles_embs)
        # print(smiles_embs.shape)
        #print("protein_embs=",protein_embs)
        # print(protein_embs.shape)

        combine_embs=torch.cat((smiles_embs,protein_embs),dim=1)
        #print("combine_embs=",combine_embs)
        #print(combine_embs.shape)
        c=self.fc1(combine_embs)
        c=self.fc2(c)
        c=self.fc3(c)
        c=self.fc4(c)
        #preds=self.acti(c)
        preds=c
        return preds
    

    
        
ds=MyMolProtDataset("../data/davis.txt")
model=MyModel()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_f=nn.MSELoss()

print(ds.__len__())
loader=DataLoader(ds,batch_size=64,shuffle=True)

def test(ds):
    test_loader=DataLoader(ds,batch_size=64,shuffle=True)
    for batch in test_loader:
        smiles, protein, label=batch
        # print(smiles,"\n")
        # print(protein,"\n")
        label=[[float(i)] for i in label]
        print("label=",label,"\n")
        label=torch.tensor(label)
        preds=model(smiles,protein)
        T=label
        S=preds.detach()
        lr=LinearRegression()
        performance=lr.fit(S,T)
        s2=S.squeeze(1)
        t2=[i[0] for i in T]
        R_score,p=pearsonr(t2,s2)
        r2=r2_score(t2,s2)
        print("R_score=",R_score," p=",p," r2=",r2)
        
        y_pred=lr.predict(S)
        
        plt.figure()
        plt.scatter(S,T,c='r')

        
        plt.plot(S,y_pred,c='b',label='R score %.3f p %.3f'%(R_score,p))
        plt.legend()
        plt.savefig("regression.jpg")

        break

for epoch in range(100):

    f_train_loss=open("output/result/my_model_davis_regression.txt","a")
    loss=0

    for batch in loader:
        smiles, protein, label=batch
        # print(smiles,"\n")
        # print(protein,"\n")
        label=[[float(i)] for i in label]
        print("label=",label,"\n")
        label=torch.tensor(label)

        preds=model(smiles,protein)
        
        print("preds=",preds)
        print(preds.shape)

        loss=loss_f(preds,label)
        print("loss=",loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    f_train_loss.write(str(loss)+"\n")
    f_train_loss.close()
        
    print("trained a epoch, begin test......")
    test(ds)
    print()



    

    



