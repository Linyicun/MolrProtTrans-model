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

from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc, roc_curve

import sklearn

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


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
      
    def forward(self,smiles,protein,train=True):
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
        emb1=c
        c=self.fc2(c)
        emb2=c
        c=self.fc3(c)
        emb=c
        c=self.fc4(c)
        #preds=self.acti(c)
        preds=c

        if train:
            return preds
        else:
            return preds,emb,emb1,emb2
    

    
        
ds=MyMolProtDataset("../Human,C.elegans/dataset/human_data.txt")
model=MyModel()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_f=nn.MSELoss()

print(ds.__len__())
loader=DataLoader(ds,batch_size=64,shuffle=True)

for epoch in range(3):
    for batch in loader:
        smiles, protein, label=batch
        # print(smiles,"\n")
        # print(protein,"\n")
        label=[[float(i)] for i in label]
        #print("label=",label,"\n")
        label=torch.tensor(label)

        preds=model(smiles,protein,train=True)
        
        
        # print("preds=",preds)
        # print(preds.shape)

        loss=loss_f(preds,label)
        print("loss=",loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        


print("train over, begin test......")
print()
ds_test=MyMolProtDataset("../data/GPCR_test.txt")
test_loader=DataLoader(ds_test,batch_size=64,shuffle=True)

import matplotlib.pyplot as plt

plt.figure()

import csv
f_csv=open("my_model_emb_kinase.csv","w")
writer=csv.writer(f_csv)

tot_emb=[]
tot_emb1=[]
tot_emb2=[]
tot_label=[]

for batch in test_loader:
    smiles, protein, label=batch
    # print(smiles,"\n")
    # print(protein,"\n")
    label=[[float(i)] for i in label]
    print("label=",label,"\n")
    label=torch.tensor(label)

    preds,emb,emb1,emb2=model(smiles,protein,train=False)

    for i in range(emb.shape[0]):
        row=[]
        for j in range(emb.shape[1]):
            row.append(emb[i][j].item())
        row.append(label[i].item())
        writer.writerow(row)
    
    fc=torch.nn.Linear(16,2)
    fc1=torch.nn.Linear(256,2)
    fc2=torch.nn.Linear(64,2)
    emb=fc(emb)
    emb=emb.detach()

    emb1=fc1(emb1)
    emb1=emb1.detach()
    emb2=fc2(emb2)
    emb2=emb2.detach()

    tot_emb.append(emb)
    tot_emb1.append(emb1)
    tot_emb2.append(emb2)
    tot_label.append(label)
    

    # for i in range(emb.shape[0]):
    #     if label[i].item()==0:
    #         plt.scatter(emb[i][0],emb[i][1],c='r')
    #     else:
    #         plt.scatter(emb[i][0],emb[i][1],c='b')


    
    T=label
    S=preds.detach()

    print("T shape= ",T.shape)

    if(T.shape[0]<=1):
        break

    T[-1]=0
    T[-2]=1
    

    AUC = roc_auc_score(T, S)
    tpr, fpr, _ = precision_recall_curve(T, S)
    PRC = auc(fpr, tpr)
    print("S=",S)
    print("AUC=",AUC)
    print("PRC=",PRC)
    acc=0
    s2=[0 if i<0.5 else 1 for i in S]
    f1=sklearn.metrics.f1_score(T,s2,average='macro')
    for i in range(len(s2)):
        if s2[i]==T[i]:
            acc+=1
    acc/=len(s2)

    print("f1=",f1)
    print("acc=",acc)

    fr,tr,thresh=roc_curve(T,S) 
    r_a=auc(fr,tr)
    plt.plot(fr,tr,label='AUC=%0.3f'%r_a)
    plt.legend()
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.savefig("roc_human.jpg")
    plt.show()

    f_roc_data=open('my_model_human_ROC_data.txt',"w")
    f_roc_data.write(str(fr)+"\n")
    f_roc_data.write(str(tr)+"\n")
    f_roc_data.close()
    
f_csv.close()

# plt.figure()
# for cnt,emb in enumerate(tot_emb):
#     label=tot_label[cnt]
#     for i in range(emb.shape[0]):
#         if label[i].item()==0:
#             plt.scatter(emb[i][0],emb[i][1],c='r')
#         else:
#             plt.scatter(emb[i][0],emb[i][1],c='b')

# plt.savefig("emb_vision_kinase.jpg")


    
# plt.figure()
# for cnt,emb1 in enumerate(tot_emb1):
#     label=tot_label[cnt]
#     for i in range(emb1.shape[0]):
#         if label[i].item()==0:
#             plt.scatter(emb1[i][0],emb1[i][1],c='r')
#         else:
#             plt.scatter(emb1[i][0],emb1[i][1],c='b')

# plt.savefig("emb1_vision_kinase.jpg")
    

# plt.figure()
# for cnt,emb2 in enumerate(tot_emb2):
#     label=tot_label[cnt]
#     for i in range(emb2.shape[0]):
#         if label[i].item()==0:
#             plt.scatter(emb2[i][0],emb2[i][1],c='r')
#         else:
#             plt.scatter(emb2[i][0],emb2[i][1],c='b')

# plt.savefig("emb2_vision_kanase.jpg")


