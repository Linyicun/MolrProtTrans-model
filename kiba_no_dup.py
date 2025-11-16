f=open("../data/kiba.txt","r")
f2=open("../data/kiba_simple.txt","w")
smile_dict=dict()
cnt=0
for line in f.readlines():
    smiles,protein,label=line.split()
    if smiles not in smile_dict:
        cnt+=1
        if cnt<=300:
            f2.write(smiles+" "+protein+" "+label+"\n")
            smile_dict[smiles]=1
    else:
        smile_dict[smiles]+=1
        if smile_dict[smiles]<=1:
            f2.write(smiles+" "+protein+" "+label+"\n")

f.close()
f2.close()