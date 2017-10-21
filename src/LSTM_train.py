# -*- coding: utf-8 -*-

"""

This module is used to train the LSTM model for feature aggregation of those extracted from ensemble models.

"""

import tensorflow as tf
import numpy as np
import argparse
from LSTM import LSTM

def load_data(path,feature_path,flag=False):
    """Load data into model

    Args:
        path (string): The path to the data
        feature_path (string): The path to the geometric features (of faces)
        flag (boolean): Flag indicating whether to add feature whose label is -1

    """
    data_=np.load(path)
    features=None
    if feature_path!=None:
        features=np.load(feature_path).item()
    data=[] # image features
    target=[] # labels
    features_=[] # geometric features
    total=0
    for instance in data_:
        if features!=None and instance['name'] not in features:
            continue
        if instance['label']==-1:
            if not flag:
                continue
            else:
                data.append(np.array(instance['feature']))
                target.append([])
                if features!=None:
                    features_.append(np.reshape(features[instance['name']],(1,1176)))
        else:
            data.append(np.array(instance['feature']))
            target.append(instance['label'])
            if features!=None:
                features_.append(np.reshape(features[instance['name']],(1,1176)))
                
    print total
    return data,target,features_

def parse_args():
    """Parse arguments to the main
    """
    parser=argparse.ArgumentParser(description='group emotion analysis')
    parser.add_argument('--net_number',dest='number_of_nets',default=1,type=int)
    parser.add_argument('--train_path',dest='train_path')
    parser.add_argument('--val_path',dest='val_path')
    parser.add_argument('--iter',dest='max_step',type=int)
    parser.add_argument('--weights',dest='weights')
    parser.add_argument('--geo_feature_path',dest='geo_feature_path')
    return parser.parse_args()    

def main():
    args=parse_args()

    # initialize LSTM
    model=LSTM(args.number_of_nets)
    x=[]

    if args.geo_feature_path!=None:
        x.append(tf.placeholder(tf.float32,(1,1176)))

    model.inference(x)

    if args.weights!=None:
        model.assign_weights(args.weights)

    # load training and validation set
    train_data,train_target,train_features=load_data(args.train_path,args.geo_feature_path)
    train_data1,_,train_features1=load_data(args.train_path,args.geo_feature_path,True)
    val_data,val_target,val_features=load_data(args.val_path,args.geo_feature_path)
    val_data1,_,val_features1=load_data(args.val_path,args.geo_feature_path,True)

    # train LSTM
    for i in xrange(args.max_step):
        model.train(train_data,train_target,x,train_features)

    # save trained LSTM
    model.save('weights/%dnets_group_label_lstm_new.npy'%(args.number_of_nets))

    # # test model
    model.test(val_data,val_target,x,val_features)

    # extract training features
    features=model.extract(train_data1,x,train_features1)
    features_={}

    for f,feature in zip(np.load(args.train_path),features):
        features_[f['name']]=feature

    print len(features_)

    np.save('features/train_{}nets_LSTM_features_to_validate6.npy'.format(args.number_of_nets),features_)

    # extract validation features
    features=model.extenract(val_data1,x,val_features1)
    features_={}
    
    for f,feature in zip(np.load(args.val_path),features):
        features_[f['name']]=feature
    
    print len(features_)
    
    np.save('features/val_{}nets_LSTM_features_to_validate6.npy'.format(args.number_of_nets),features_)

if __name__=='__main__':
    main() 
