import csv
f=open("my_model_emb.csv","r")
reader=csv.reader(f)
f2=open("my_model_emb2.csv","w",newline="")
writer=csv.writer(f2)
for line in reader:
    if len(line)>0:
        writer.writerow(line)
f.close()
f2.close()