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
    parser.add_argument('--test_set',dest='test_set',type=str)
    parser.add_argument('--model_path',dest='model_path',type=str)
    parser.add_argument('--data_path',dest='data_path',type=str)
    parser.add_argument('--loss',dest='loss_function',type=str) 
    return parser.parse_args()

def image_preprocessing(image_path,sess):
    import cv2
    im=cv2.imread(image_path)
    im_tensor=tf.convert_to_tensor(im)
    im_tensor=tf.image.convert_image_dtype(im_tensor,dtype=tf.float32)
    im_tensor=tf.image.resize_images(im_tensor,(FLAGS.input_size,FLAGS.input_size),0)
    im_tensor=tf.reshape(im_tensor,shape=(1,FLAGS.input_size,FLAGS.input_size,3))
    im_tensor=tf.subtract(im_tensor,0.5)
    im_tensor=tf.multiply(im_tensor,2)
    return sess.run(im_tensor)

def main(_):
    
    args=parse_args()
    train_set,val_set=load_mat_data()
    images=tf.placeholder(tf.float32,shape=[1,FLAGS.input_size,FLAGS.input_size,3])
    import get_net
    is_training=tf.placeholder(tf.bool)
    logits=get_net.preprocessing(args,images,'feature',is_training)
    #logits=get_net.preprocessing(args,images,'feature',is_training)
 
    import classify
    test_set=None
    if args.test_set=='train':
        test_set=train_set
    elif args.test_set=='val':
        test_set=val_set
    
    import os
    images_batch=np.empty([0,FLAGS.input_size,FLAGS.input_size,3])
    labels_batch=[]
    labels=tf.placeholder(tf.int32,shape=(FLAGS.batch_size,))
    
    sess=tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    data=np.load(args.model_path).item()
    op=[]
    for v in tf.get_collection(tf.GraphKeys.VARIABLES):
        op.append(v.assign(data[v.name]))
    sess.run(op)
        #if a[-1]=='biases:0':
    
    correct_num=0
    total_num=0
    features=[]
    import data_input
    ims=data_input.get_filequeue([args.data_path+f['filename'] for f in test_set])
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    for logit,i in zip(logits,range(1,args.number_of_nets+1)):
        ii=0
        while ii<len(test_set):    
            im=sess.run(ims)
            print ii
            #im=cv2.resize(cv2.imread(path),(FLAGS.input_size,FLAGS.input_size),interpolation=cv2.INTER_CUBIC)
                   
            #im=im/255.0*2-1 
            
            #im=np.resize(im,(1,FLAGS.input_size,FLAGS.input_size,3))
            
            out=sess.run(logit,{is_training:False,images:im})
            if i==1:
                features.append({'name':test_set[ii]['filename'],'feature':[out],'label':test_set[ii]['label_index']})            
            else:
                features[ii]['feature'].append(out)
            ii+=1
        
        np.save('features/'+args.test_set+'_%dnets_feature.npy'%i,features)

if __name__ == '__main__':
    tf.app.run()
