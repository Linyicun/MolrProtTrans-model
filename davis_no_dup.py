f=open("../data/davis.txt","r")
f2=open("../data/davis_simple.txt","w")
smile_dict=dict()
for line in f.readlines():
    smiles,protein,label=line.split()
    if smiles not in smile_dict:
        f2.write(smiles+" "+protein+" "+label+"\n")
        smile_dict[smiles]=1
    else:
        smile_dict[smiles]+=1
        if smile_dict[smiles]<=5:
            f2.write(smiles+" "+protein+" "+label+"\n")

f.close()
f2.close()