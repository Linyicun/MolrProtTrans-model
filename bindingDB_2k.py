

f_deal=open('bingdingDB_2k.txt','w')



cnt=0
f=open('train.txt','r')
for line in f.readlines():
    smiles,protein,label=line.split()
    
    cnt+=1
    
    if cnt>2000:
        break
    else:
        
        smiles,protein,label=line.split()
        
        f_deal.write(smiles+'\t'+protein+'\t'+label+'\n')

f_deal.close()


