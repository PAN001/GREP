import scipy.io as sio
import numpy as np
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

def getCENTRISTFeatAsList(originFeat):
    CENTRISTFeatList = []
    for element in originFeat:
        thisFeat = element.reshape(1, 4064)
        CENTRISTFeatList.append(thisFeat[0])
    return CENTRISTFeatList

def getGroupIntensityAsList(originIntensity):
    groupIntensityList = []
    for element in originIntensity:
        groupIntensityList.append(element[0][0])
    return groupIntensityList

matPath = "/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/HappeiDetectedFaces/HAPPEI_DATA_NEW.mat"
matFile = sio.loadmat(matPath)

trainData = matFile['DATA_TR_NEW']
validationData = matFile['DATA_VA_NEW']
testData = matFile['DATA_TE_NEW']

# get CENTRIST_feat
trainCENTRISTFeat = getCENTRISTFeatAsList(trainData[1:, 4])
valCENTRISTFeat = getCENTRISTFeatAsList(validationData[1:, 4])

# get group intensity
trainGroupIntensity = getGroupIntensityAsList(trainData[1:, 1])
validationGroupIntensity = getGroupIntensityAsList(validationData[1:, 1])

# train SVM
clf = SVR(C=1.0, epsilon=0.2, kernel=chi2_kernel)
print("train starts")
clf.fit(trainCENTRISTFeat, trainGroupIntensity)
print("train ends")

validationPredications = clf.predict(valCENTRISTFeat)

# The mean squared error
print("Root Mean squared error before normalizing: %.2f"
      % np.sqrt(mean_squared_error(validationPredications, validationGroupIntensity)))

# The mean squared error after normalizing
min_max_scaler = preprocessing.MinMaxScaler((0, 5))
validationPredicationsNormalized = min_max_scaler.fit_transform(validationPredications)
print("Root Mean squared error after normalizing: %.2f"
      % np.sqrt(mean_squared_error(validationPredicationsNormalized, validationGroupIntensity)))