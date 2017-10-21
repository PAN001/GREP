from sklearn.svm import SVR
import numpy as np
import math
def processing(data,info):
    group_data={}
    for key,value in data.items():
        group_name,group_label=info[key]
        if group_name in group_data:
            feature,num,label=group_data[group_name]
            feature+=value
            group_data[group_name]=(feature,num+1,label)
        else:
            group_data[group_name]=(value,1,group_label)
    X=[]
    Y=[]
    for key,(feature,num,label) in group_data.items():
        feature=np.reshape(feature,(64))
        X.append(feature/num)
        Y.append(label)
    return np.array(X),np.array(Y)
    
train=np.load('features/train_1nets_LSTM_features.npy').item()
val=np.load('features/val_1nets_LSTM_features.npy').item()
train_info=np.load('../group_train.npy').item()
val_info=np.load('../group_val.npy').item()

train_X,train_Y=processing(train,train_info)
val_X,val_Y=processing(val,val_info)
min_=1e9
for C in [1.0,2.0,3.0,4.0,5.0,6.0]:
    for e in [0.1,0.2,0.3,0.4,0.5]:
        clf = SVR(C=C, epsilon=e)
        clf.fit(train_X,train_Y)
        predict_Y=clf.predict(val_X)
        min_=min(math.sqrt(((predict_Y-val_Y)**2).sum()/val_Y.shape[0]),min_)
print min_

