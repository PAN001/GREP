# -*- coding: utf-8 -*-

"""

This module is used for defining the LSTM model for feature aggregation of those extracted from ensemble models.

"""

import tensorflow as tf
import numpy as np
import math

class LSTM:
    def __init__(self,max_step):
        """Initialize a LSTM model

        Args:
            self (objecdt): self
            max_step (int): maximum steps

        """
        self.input_size=64
        self.batch_size=1
        self.max_step=max_step
        self.number_of_classes=1
        self.num_hidden=128
        self.x=tf.placeholder(tf.float32,(self.batch_size,self.max_step,self.input_size))
        self.y=tf.placeholder(tf.float32,(self.number_of_classes,))
        self.length=tf.placeholder(tf.int32,1)
        self.sess=tf.Session() 

    def inference(self,tensors):
        def fc(x,num_out):
            num_in=x.get_shape()[1]
            w=tf.get_variable("weights",(num_in,num_out) ,
                                initializer=tf.random_normal_initializer())
            return tf.matmul(x,w)

        x=self.x

        for tensor in tensors:
            x=fc(tensor,self.input_size)
            x=tf.reshape(x,(1,1,self.input_size))
            x=tf.concat([self.x,x],1)
            self.max_step+=1

        cell=tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(self.num_hidden)
        cells=tf.contrib.rnn.MultiRNNCell([cell]*2)
        state=cell.zero_state((self.batch_size),tf.float32)
        
        x=tf.split(x,self.max_step,1)
        x=[tf.reshape(i,(self.batch_size,self.input_size)) for i in x]
        out,last_state=tf.contrib.rnn.static_rnn(cell=cells,dtype=tf.float32,inputs=x,sequence_length=self.length)
        #last = tf.gather(val, int(val.get_shape()[0]) - 1)
        last=tf.gather(out,self.length-1)
        last=tf.reshape(last,(1,self.num_hidden))
        
        weight = tf.Variable(tf.truncated_normal([self.num_hidden, self.number_of_classes]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.number_of_classes]))
        prediction=tf.matmul(last,weight)+bias
        self.feature=last
        loss=tf.reduce_sum(tf.pow(tf.subtract(self.y,prediction),2))
        optimizer = tf.train.AdamOptimizer()
        minimize = optimizer.minimize(loss)
        
        self.sess.run(tf.global_variables_initializer())
        self.prediction=prediction
        self.minimize=minimize
        self.state=last_state 

    def assign_weights(self,path):
        op=[]
        data=np.load(path).item()
        for v in tf.get_collection(tf.GraphKeys.VARIABLES):
            op.append(v.assign(data[v.name]))
        self.sess.run(op)
        print("weighted loaded")
    
    def train(self,data,target,tensors,features):
        ptr = 0
        for x,y in zip(data,target):
            length=self.max_step
            x=np.reshape(x,(x.shape[1],x.shape[0],x.shape[2]))
            feed_dict={self.x:x,self.y:(y,)}
            if len(tensors)>0:
                feed_dict[tensors[0]]=features[ptr]
            
                if features[ptr]==None:
                    feed_dict[tensors[0]]=np.zeros((1,1176))        
                    length-=1
            feed_dict[self.length]=(length,)
            self.sess.run(self.minimize,feed_dict)
            ptr+=1
            print ptr
    
    def test(self,data,target,tensors,features):
        error=0
        ptr=0
        for x,y in zip(data,target):
            #print x
            flag=False 
            length=self.max_step
            x=np.reshape(x,(x.shape[1],x.shape[0],x.shape[2]))
            feed_dict={self.x:x,self.y:(y,)}
            if len(tensors)>0:
                feed_dict[tensors[0]]=features[ptr]
                if features[ptr]==None:
                    feed_dict[tensors[0]]=np.zeros((1,1176))        
                    length-=1 
            feed_dict[self.length]=(length,)
            #import random
            #x_zero[0,1,0]=random.randint(0,100)
            #print x_zero
            
            out=self.sess.run(self.prediction,feed_dict)
            error+=abs(out-y)**2
            ptr+=1
        print math.sqrt(1.0*error/len(data))

    def extract(self,data,tensors,features):
        ptr=0
        features_=[]
        for x in data:
            length=self.max_step
            x=np.reshape(x,(x.shape[1],x.shape[0],x.shape[2]))
            feed_dict={self.x:x}
            if len(tensors)>0:
                feed_dict[tensors[0]]=features[ptr]
                if features[ptr]==None:
                    feed_dict[tensors[0]]=np.zeros((1,1176))        
                    length-=1
            feed_dict[self.length]=(length,)

            out=self.sess.run(self.feature,feed_dict)
            features_.append(out)
            ptr+=1
        return features_

    def save(self,path):
        import initialize_model
        initialize_model.save_model(path,self.sess)

    def load(self,path):
        import initialize_model
        initialize_model.initialize_with_npy(path, self.sess)

def generate_data():
    import random
    train_input=[]
    for j in range(10,15):
        train_input =train_input+ ['{0:b}'.format(i) for i in range(2**j)]
        
    from random import shuffle
    shuffle(train_input)
    train_input = [map(int,i) for i in train_input]
    ti  = []
    for i in train_input:
        temp_list = []
        for j in i:
                temp_list.append([j])
        ti.append(np.array(temp_list))
    
    train_input = ti
    
    train_output = []
    
    for i in train_input:
        ze=np.zeros(2)
        a=0
        if len(i)>=5:
            a=i[4]
        ze[a]=1
        train_output.append(ze)
    num_of_train=int(len(train_input)*0.1)

    return train_input[:num_of_train],train_output[:num_of_train],train_input[num_of_train:],train_output[num_of_train:]
 

    
   
