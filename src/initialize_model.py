import tensorflow as tf
import numpy as np
def save_model(model_path,sess):
    data={}
    for v in tf.get_collection(tf.GraphKeys.VARIABLES):
        data[v.name]=sess.run(v)
    np.save(model_path,data)
    print "model saved"

def initialize_with_npy(model_path,sess):
    #init=tf.initialize_all_variables()
    #sess.run(init)
    data=np.load(model_path).item()
    for v in tf.trainable_variables():
        sess.run(v.assign(data[v.name]))

def initialize_with_caffe_resnet(logits,caffe_model_path,sess):
    data=np.load(caffe_model_path).item()
    for v in tf.get_collection(tf.GraphKeys.VARIABLES):
        words=str(v.name[0:-2]).split('/')
        words=words[1:]
        key_in_caffe_model=None
        subkey=None
        if words[0]=='1':
            if words[1]=='res':
                key_in_caffe_model='conv1'
            elif words[1]=='bn':
                key_in_caffe_model='bn_conv1'
            subkey=words[2]
        elif words[0]=='fc':
            continue
        else:
            if len(words)<5:
                print words
                
            
            key_in_caffe_model=words[3]+words[0]+words[1]+'_'+words[2]
            subkey=words[4]
        sess.run(v.assign(data[key_in_caffe_model][subkey]))
    
        """vals=data[key_in_caffe_model]
        del vals[subkey]
        if len(vals)==0:
            del data[key_in_caffe_model]"""
         
     
