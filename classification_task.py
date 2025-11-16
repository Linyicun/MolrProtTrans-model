from sklearn import svm, tree, neighbors, neural_network
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.model_selection import KFold
import warnings
from sklearn import datasets
import csv

import matplotlib.pyplot as plt

# x_train=[[1],[2],[3]]
# y_train=[4,5,6]

# x_test=[[7],[8]]
# y_test=[10,11]

x_train, y_train, x_test, y_test=[],[],[],[]

f=open("my_model_emb2.csv","r")
reader=csv.reader(f)

f_classify=open("classify_result.txt","w",encoding="utf-8")

cnt=0
l=0
lines=[]
for line in reader:
    l+=1
    lines.append(line)

for line in lines:
    cnt+=1
    if cnt<=l*0.8:
        x_train.append(line[0:16])
        y_train.append(line[16])
    else:
        x_test.append(line[0:16])
        y_test.append(line[16])
f.close()

models=[
    # SVM
    svm.SVC(C=2, kernel='poly',degree=5, decision_function_shape='ovo',probability=True),
    # 决策树
    tree.DecisionTreeClassifier(),
    # KNN
    neighbors.KNeighborsClassifier(10),
    # 人工神经网络
    neural_network.MLPClassifier(alpha=1, max_iter=1000),
    # 高斯分类器
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    # Logistic回归
    LogisticRegression (multi_class='multinomial'),
    # 随机森林
    RandomForestClassifier(),
]

model_names=[
    'SVM',
    '决策树',
    'KNN',
    'MLP',
    '高斯过程分类器',
    'Logistic',
    'RandomForest',
    'Voting',
]

for i in range(len(models)):
    model_iter=models[i]
    model_iter.fit(x_train,y_train)
    pre_svm=model_iter.predict_proba(x_test)[:,1]

    p=model_iter.predict(x_test)

    yt=[int(float(i)) for i in y_test]
    
    print("pre_svm=",pre_svm)
    ps=[float(i) for i in pre_svm]
    print("yt=",yt)
    print("pre_svm=",pre_svm)
    fr,tr,thresh=roc_curve(yt,ps)
    r_a=auc(fr,tr)

    
    plt.plot(fr,tr,label='AUC=%0.3f'%r_a)
    plt.legend(loc='lower right')
    plt.savefig("roc_classification_"+model_names[i]+".jpg")
    plt.show()

    auc1=accuracy_score(y_test,p)
    precision1=precision_score(y_test,p,average='macro')
    recall1=recall_score(y_test,p,average="macro")
    f1_score1=f1_score(y_test,p,average="macro")

    print("model=",model_names[i])
    print("auc=",auc1)
    print("precision=",precision1)
    print("recall_score=",recall1)
    print("f1_score=",f1_score1)

    f_classify.write("model: "+model_names[i]+"\n")
    f_classify.write("auc: "+str(auc1)+"\n")
    f_classify.write("precision: "+str(precision1)+"\n")
    f_classify.write("recall score: "+str(recall1)+"\n")
    f_classify.write("f1_score="+str(f1_score1)+"\n")
    f_classify.write("##############################\n")


    
    print("###################")




