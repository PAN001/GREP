import scipy.io as sio
import numpy as np

def convert(input):
    if isinstance(input, dict):
        return {convert(key): convert(value) for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [convert(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

dataFile1 = '/Users/PAN/Downloads/features/train_features.mat'
dataFile2 = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/HappeiDetectedFaces/HAPPEI_DATA_NEW.mat'
data1 = sio.loadmat(dataFile1)
data2 = sio.loadmat(dataFile2)

print('done')
# sio.savemat('/Users/PAN/Downloads/features/train_features_new.mat', )