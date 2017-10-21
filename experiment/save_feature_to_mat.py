import scipy.io as scio
import numpy as np
import collections
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
# from __future__ import division

trainFeatPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/train_l2loss_features.mat'
validationFeatPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/val_l2loss_features.mat'
dataMainPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/HappeiDetectedFaces/HAPPEI_MAIN_INFO.mat'
allPath = "/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/HappeiDetectedFaces/HAPPEI_DATA_NEW.mat"

trainFeat = scio.loadmat(trainFeatPath)
validationFeat = scio.loadmat(validationFeatPath)
dataMain = scio.loadmat(dataMainPath)
allData = scio.loadmat(allPath)

# train
trainMain = dataMain['TR_Main_Info']
trainAllData = allData['DATA_TR_NEW']
trainMainDic = {} # faceName as key, intensity and fileName as values
trainGroupFaceDic = {} # fileName as key, faceName as values
trainGroup = [] # a list of group files (as a order)
trainLabels = [] # a list of group labels under trainGroup order
trainKeys = trainFeat.keys()
trainFeatures = []
trainFeaturesWithoutUnlabeled = []
trainLabels = []
trainLabelsWithoutUnlabeled = []
i = 0

# convert trainMain to dic
for sample in trainMain:
    faceFileName = sample[0][0]
    faceFileName = faceFileName.encode("ascii")

    fileName = sample[1][0]
    fileName = fileName.encode("ascii")

    if sample[2].size != 0:
        label = sample[2][0][0]

    else:
        label = -1

    trainMainDic[faceFileName] = [fileName, label]

for key in trainKeys:
    i = i + 1
    if (key == "__header__") | (key == "__version__") | (key == "__globals__"):
        continue

    feature = trainFeat[key][0]
    trainFeatures.append(feature)

    label = trainMainDic[key][1]
    trainLabels.append(label)

    if label != -1:
        trainFeaturesWithoutUnlabeled.append(feature)
        trainLabelsWithoutUnlabeled.append(label)

# validation
validationMain = dataMain['VA_Main_Info']
validationMainDic = {}
validationKeys = []
validationFeatures = []
validationLabels = []
validationFeaturesWithoutUnlabeled = []
trainLabels = []
validationLabelsWithoutUnlabeled = []

# convert validationMain to dic
for sample in validationMain:
    # i = i + 1
    # print(i)
    faceFileName = sample[0][0]
    faceFileName = faceFileName.encode("ascii")
    validationKeys.append(faceFileName)

    fileName = sample[1][0]
    fileName = fileName.encode("ascii")

    if sample[2].size != 0:
        label = sample[2][0][0]
    else:
        label = -1

    validationMainDic[faceFileName] = [fileName, label]

for key in validationKeys:
    i = i + 1
    if (key == "__header__") | (key == "__version__") | (key == "__globals__"):
        continue

    feature = validationFeat[key][0]
    validationFeatures.append(feature)

    label = validationMainDic[key][1]
    validationLabels.append(label)

    if label != -1:
        validationFeaturesWithoutUnlabeled.append(feature)
        validationLabelsWithoutUnlabeled.append(label)

# store features back into mat
newColumn = 1
for element in trainAllData[1:]:
    faces = element[3]
    faceFileNames = faces[1:,12]


    newColumn = np.array(['Deep_Feature'], dtype=object)

    for faceFileName in faceFileNames:
        thisFaceFileName = faceFileName[0].encode('ascii')

        deppFeature = trainFeat[thisFaceFileName][0]
        newColumn = np.vstack((newColumn, deppFeature))

    newColumn = newColumn.reshape(newColumn.size, 1)
    break




