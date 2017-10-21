import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
from resnet_train import train
import tensorflow as tf
import time
import os
import sys
import re
import numpy as np
import resnet
from synset import *
from image_processing import image_preprocessing
import argparse
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '/home/ryan/data/ILSVRC2012/ILSVRC2012_img_train',
                           'imagenet dir')


def file_list(data_dir):
    dir_txt = data_dir + ".txt"
    filenames = []
    with open(dir_txt, 'r') as f:
        for line in f:
            if line[0] == '.': continue
            line = line.rstrip()
            fn = os.path.join(data_dir, line)
            filenames.append(fn)
    return filenames

def extract_filename(res,data,dir_):
    import os
    for instance in data[1:]:
        
        for jpg in instance[3][1:]:
            
            jpg_name=jpg[12][0]
            jpg_path=dir_+str(jpg_name)
            if jpg[1][0].shape[0]==1:
                res.append({'filename':jpg_path,'label_index':int(jpg[1][0][0])})
            else:
                res.append({'filename':jpg_path,'label_index':int(-1)})
    return res 

def load_mat_data():
    mat_path="../HAPPEI_DATA_NEW.mat"
    from scipy.io import loadmat
    data=loadmat(mat_path)
    train_data=data['DATA_TR_NEW']
    val_data=data['DATA_VA_NEW']
    train=[]
    val=[]
        
    return extract_filename(train,train_data,""),extract_filename(val,val_data,"")      

def load_data(data_dir):
    data = []
    i = 0

    print "listing files in", data_dir
    start_time = time.time()
    files = file_list(data_dir)
    duration = time.time() - start_time
    print "took %f sec" % duration

    for img_fn in files:
        ext = os.path.splitext(img_fn)[1]
        if ext != '.JPEG': continue

        label_name = re.search(r'(n\d+)', img_fn).group(1)
        fn = os.path.join(data_dir, img_fn)

        label_index = synset_map[label_name]["index"]

        data.append({
            "filename": fn,
            "label_name": label_name,
            "label_index": label_index,
            "desc": synset[label_index],
        })

    return data

def distorted_inputs(data,image_dir):
    #data = load_data(FLAGS.data_dir)
    
    filenames = [ image_dir+d['filename'] for d in data ]
    label_indexes = [ d['label_index'] for d in data ]
    filename, label_index = tf.train.slice_input_producer([filenames, label_indexes], shuffle=True)

    num_preprocess_threads = 4
    images_and_labels = []
    for thread_id in range(num_preprocess_threads):
        image_buffer = tf.read_file(filename)

        bbox = []
        train = True
        image = image_preprocessing(image_buffer, bbox, train, thread_id)
        images_and_labels.append([image, label_index])

    images, label_index_batch = tf.train.batch_join(
        images_and_labels,
        batch_size=FLAGS.batch_size,
        capacity=2 * num_preprocess_threads * FLAGS.batch_size,
        allow_smaller_final_batch=True)

    height = FLAGS.input_size
    width = FLAGS.input_size
    depth = 3

    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[FLAGS.batch_size, height, width, depth])

    return images, tf.reshape(label_index_batch, [FLAGS.batch_size])

def parse_args():
    parser=argparse.ArgumentParser(description='group emotion analysis')
    parser.add_argument('--net_number',dest='number_of_nets',default=1,type=int)
    parser.add_argument('--loss',dest='loss_function',type=str)
    parser.add_argument('--weights',dest='weights',type=str) 
    return parser.parse_args() 
    
def main(_):
    
    args=parse_args()
    train_set,val_set=load_mat_data()
    
    table=[[] for i in range(6)]
    for f in train_set:
        if f['label_index']!=-1:
            table[f['label_index']].append(f)
    import random 
    train_set_selected=[[] for i in range(args.number_of_nets)]
    train_images=[None for i in range(args.number_of_nets)]
    train_labels=[None for i in range(args.number_of_nets)]
    for j in xrange(args.number_of_nets):
        for i in xrange(380):
            for x in xrange(6):
                ranint=random.randint(0,len(table[x])-1)
                train_set_selected[j].append(table[x][ranint])

        train_images[j],train_labels[j] = distorted_inputs(train_set_selected,"../train/")
    import get_net
    images=tf.placeholder(tf.float32,shape=[FLAGS.batch_size,FLAGS.input_size,FLAGS.input_size,3])
    is_training=tf.placeholder(tf.bool)
    logits=get_net.preprocessing(args,images,'final_layer',is_training)
    train(args,logits,is_training,images,train_images, train_labels,val_set,args.loss_function)


if __name__ == '__main__':
    tf.app.run()
