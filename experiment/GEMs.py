from __future__ import division
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
import matplotlib
import matplotlib.pyplot as plt
import random
import math
from plot_cm import *
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV

# helper function
def getFeatureAndLabel(mainDic, origFeatDic):
    keys = origFeatDic.keys()
    featList = []
    labelList = []
    featListWithoutUnlabeled = []
    labelListWithoutUnlabeled = []
    for key in keys:
        if (key == "__header__") | (key == "__version__") | (key == "__globals__"):
            continue

        feature = origFeatDic[key][0]
        featList.append(feature)

        label = mainDic[key][1]
        labelList.append(label)

        if label != -1:
            featListWithoutUnlabeled.append(feature)
            labelListWithoutUnlabeled.append(label)

    return featList, labelList, featListWithoutUnlabeled, labelListWithoutUnlabeled

def groupPredict(groupList, groupFaceDic, faceFeatDic, groupLabel, model):
    Prediction = []
    index = 0
    numberOfNoFaces = 0
    for element in groupList:
        faces = groupFaceDic[element]['face']
        averageLabel = 0
        i = 0

        for face in faces:
            thisFeature = faceFeatDic[face]
            thisFeature = np.array(thisFeature).reshape((1, -1))
            thisPrediction = model.predict(thisFeature)[0]
            averageLabel += thisPrediction
            i = i + 1

        if i == 0: # no face detected in the group
            averageLabel = groupLabel[index]
            numberOfNoFaces = numberOfNoFaces + 1
        else:
            averageLabel = averageLabel / i

        Prediction.append(averageLabel)
        index = index + 1

    return Prediction

# Distance function
def distance(A, B):
    sq1 = (A[0]- B[0]) ** 2
    sq2 = (A[1]- B[1]) ** 2
    dist = math.sqrt(sq1 + sq2)
    return dist

def roundPrediction(predictions):
    roundedPrediction = []
    for prediction in predictions:
        if prediction <= 0:
            roundedPrediction.append(0)
        elif prediction >= 5:
            roundedPrediction.append(5)
        else:
            roundedPrediction.append(int(round(prediction)))
    return roundedPrediction

def groupPredictWeighted(groupList, groupFaceDic, faceFeatDic, groupLabel, model, mainDic, alpha, beta):
    predictions = []
    weightsList = []
    weightedPredictionList = []
    numberOfNoFaces = 0
    for element in groupList:
        faces = groupFaceDic[element]['face']
        i = 0
        boundSizeList = []
        centroidList = []
        thesePredictions = []
        weightedPrediction = 0
        for face in faces:
            thisFeature = faceFeatDic[face]
            thisFeature = np.array(thisFeature).reshape((1, -1))
            thiePrediction = model.predict(thisFeature)[0]
            thesePredictions.append(thiePrediction)
            i = i + 1

            # append locations
            location = mainDic[face][2]
            if len(location) > 0:
                boundSize = location[2]
                centroidX = location[0]
                centroidY = location[1]
                centroid = (centroidX + boundSize/2, centroidY + boundSize/2)
                boundSizeList.append(boundSize)
                centroidList.append(centroid)

        if i == 0: # no face detected in the group
            weightedPrediction = groupLabel[index]
            numberOfNoFaces = numberOfNoFaces + 1
            print("no faces in this group")
        elif i == 1: # only one face
            weightedPrediction = thesePredictions[0]
            # print("only one face")
        else: # more than one face

            # calculate weights
            weights = []
            for i in range(0, len(thesePredictions)):
                thisCentroid = centroidList[i]
                dist = 0
                for j in range(0, len(centroidList)):
                    if j == i:
                        continue

                    dist = dist + distance(thisCentroid, centroidList[j])

                thisWeight = (boundSizeList[i] * alpha) / (dist * beta)
                weights.append(thisWeight)

            # calculate weighted predictions
            weights = np.asarray(weights)
            normalizedWeights = weights / (np.sum(weights))

            thesePredictions = np.array(thesePredictions)
            weightedPredictions = thesePredictions * normalizedWeights
            weightedPrediction = np.sum(weightedPredictions)

            weightsList.append(normalizedWeights)
            weightedPredictionList.append(weightedPredictions)

        predictions.append(weightedPrediction)

    return predictions, weightsList, weightedPredictionList

def meanEncodingTrain(groupList, groupFaceDic, faceFeatDic, groupLabel):
    model = SVR(C=0.5, cache_size=200, coef0=0.0, degree=3, epsilon=0.13, gamma='auto', kernel='linear')
    averageFeatList = []
    index = 0
    numberOfNoFaces = 0

    for element in groupList:
        faces = groupFaceDic[element]['face']
        i = 0

        averageFace = None
        for face in faces:
            thisFeature = faceFeatDic[face]
            thisFeature = np.array(thisFeature).reshape((1, -1))[0]

            if i == 0:
                averageFace = thisFeature
                i = i + 1
                continue

            averageFace = averageFace + thisFeature
            i = i + 1

        if i == 0: # no face detected in the group
            numberOfNoFaces = numberOfNoFaces + 1
            print("no face ")
            print(index)
            groupLabel.pop(index)  # remove this sample
        else:
            averageFace = averageFace / i
            averageFeatList.append(averageFace)

        index = index + 1

    model.fit(averageFeatList, groupLabel)
    return model


# average feature vector
def meanEncodingPredict(groupList, groupFaceDic, faceFeatDic, groupLabel, model):
    Prediction = []
    index = 0

    for element in groupList:
        faces = groupFaceDic[element]['face']
        averageLabel = 0
        i = 0

        averageFace = None
        for face in faces:
            thisFeature = faceFeatDic[face]
            thisFeature = np.array(thisFeature).reshape((1, -1))
            if i == 0:
                averageFace = thisFeature
                i = i + 1
                continue

            averageFace = averageFace + thisFeature
            i = i + 1

        if i == 0: # no face detected in the group
            numberOfNoFaces = numberOfNoFaces + 1
        else:
            averageFace = averageFace / i

        averageLabel = model.predict(averageFace)[0]
        Prediction.append(averageLabel)
        index = index + 1

    return Prediction


def generateMeanFeatureVector(groupList, groupFaceDic, faceFeatDic, mainDic):
    weightsList = []
    weightedGroupFeaatureList = []
    numberOfNoFaces = 0
    for element in groupList:
        faces = groupFaceDic[element]['face']
        i = 0
        boundSizeList = []
        centroidList = []
        theseFeatureList = []
        weights = []
        weightedFeature = None
        for face in faces:
            thisFeature = faceFeatDic[face]
            thisFeature = np.array(thisFeature).reshape((1, -1))
            theseFeatureList.append(thisFeature)
            i = i + 1

            # append locations
            location = mainDic[face][2]
            if len(location) > 0:
                boundSize = location[2]
                centroidX = location[0]
                centroidY = location[1]
                centroid = (centroidX + boundSize/2, centroidY + boundSize/2)
                boundSizeList.append(boundSize)
                centroidList.append(centroid)

        if i == 0: # no face detected in the group
            # weightedPrediction = groupLabel[index]
            numberOfNoFaces = numberOfNoFaces + 1
            weightedFeature = []
            weightsList.append([0])
            weightedGroupFeaatureList.append(weightedFeature)
            print("no faces in this group")
            continue
        elif i == 1: # only one face
            weightedFeature = theseFeatureList[0][0]
            weightsList.append([1])
            weightedGroupFeaatureList.append(weightedFeature)
            # print("only one face")
            continue
        else: # more than one face

            # calculate weights
            for i in range(0, len(theseFeatureList)):
                thisCentroid = centroidList[i]
                dist = 0
                for j in range(0, len(centroidList)):
                    if j == i:
                        continue

                    dist = dist + distance(thisCentroid, centroidList[j])

                thisWeight = boundSizeList[i] / dist
                weights.append(thisWeight)

            # calculate weighted predictions
            weights = np.asarray(weights)
            normalizedWeights = weights / (np.sum(weights))

            weightedFeatureList = []
            for i in range(0, len(theseFeatureList)):
                thisFeature = theseFeatureList[i][0]
                # return thisFeature
                weightedFeature = thisFeature * normalizedWeights[i]
                weightedFeatureList.append(weightedFeature)

        weightedFeatureList = np.array(weightedFeatureList)
        weightedFeature = np.sum(weightedFeatureList, axis=0)
        weightedGroupFeaatureList.append(weightedFeature)
        weightsList.append(normalizedWeights)

    return weightsList, weightedGroupFeaatureList

# weighted average feature vector
def weightedMeanEncoding(trainGroupList, trainGroupFaceDic, trainFaceFeatDic, trainMainDic, trainGroupLabels, validationGroupList, validationGroupFaceDic, validationFaceFeatDic, validationMainDic):
    model = SVR(C=0.5, cache_size=200, coef0=0.0, degree=3, epsilon=0.13, gamma='auto', kernel='linear')

    trainWeights, trainWeightedFeatures = generateMeanFeatureVector(trainGroupList, trainGroupFaceDic, trainFaceFeatDic, trainMainDic)
    # print(trainWeightedFeatures[0])
    model.fit(trainWeightedFeatures, trainGroupLabels)

    # features and predictions should adhere to the order of validationGroupList
    validationWeights, validationWeightedFeatures = generateMeanFeatureVector(validationGroupList, validationGroupFaceDic, validationFaceFeatDic, validationMainDic);
    validationWeightedPredict = model.predict(validationWeightedFeatures)

    return validationWeightedPredict


def individualFacePredict(faceList, faceLabel, model):
    predication = model.predict(faceList)
    # The mean squared error
    rmse = np.sqrt(mean_squared_error(predication, faceLabel))
    print("Root Mean squared error for individual level before normalizing: %.2f" %rmse)

    # Accuracy
    predicationRounded = [round(elem) for elem in predication]
    comparison = [elem1 == elem2 for elem1, elem2 in zip(predicationRounded, faceLabel)]
    accuracy = len(find(comparison, True)) / len(comparison)
    print("Accuracy is: %.2f" % accuracy)

    return rmse, accuracy

def find(lst, target):
    result = []
    for i, x in enumerate(lst):
        if x == target:
            result.append(i)
    return result

def balanceData(origFeatList, origLabelList, num):
    minNum = min(np.histogram(origLabelList, bins=[0,1,2,3,4,5,6])[0])
    if(num > minNum):
        num = minNum

    classes = np.histogram(origLabelList)[1]
    balancedFeats = []
    balancedLabels = []
    for oneClass in classes:
        # print(oneClass)
        indices = find(origLabelList, oneClass)[:num]
        # print(indices)
        for index in indices:
            # print(index)
            balancedFeats.append(origFeatList[index])
            balancedLabels.append(origLabelList[index])

    return balancedFeats, balancedLabels

def testAll(trainPath, valPath):
    trainFeat = np.load(trainPath)
    trainFeatInDic = trainFeat.item()

    trainFeatList, trainLabelList, trainFeatListWithoutUnlabeled, trainLabelListWithoutUnlabeled = getFeatureAndLabel(trainMainDic, trainFeatInDic)

    validationFeat = np.load(valPath)
    validationFeatInDic = validationFeat.item()

    validationFeatList, validationLabelList, validationFeatListWithoutUnlabeled, validationLabelListWithoutUnlabeled = getFeatureAndLabel(validationMainDic, validationFeatInDic)

    # individual level
    # SVM training
    clf = SVR(C=0.5, cache_size=200, coef0=0.0, degree=3, epsilon=0.13, gamma='auto', kernel='linear')
    print("train starts")
    clf.fit(trainFeatListWithoutUnlabeled, trainLabelListWithoutUnlabeled)
    print("train ends")

    # individual face level
    rmseFaceLevel, accuracyFaceLevel = individualFacePredict(validationFeatListWithoutUnlabeled, validationLabelListWithoutUnlabeled, clf)

    # group level

    # mean encoding
    # SVM training
    model = meanEncodingTrain(trainGroup, trainGroupFaceDic, trainFeatInDic, trainGroupLabels)
    validationMeanEncodingPrediction = meanEncodingPredict(validationGroup, validationGroupFaceDic, validationFeatInDic, validationGroupLabels, model)
    # The mean squared error
    rmseMeanEncoding = np.sqrt(mean_squared_error(validationGroupLabels, validationMeanEncodingPrediction))
    print("Root Mean squared error with mean encoding %f" % rmseMeanEncoding)


    # weighted mean encoding
    validationWeightedMeanEncodingPrediction = weightedMeanEncoding(trainGroup, trainGroupFaceDic, trainFeatInDic, trainMainDic, trainGroupLabels, validationGroup, validationGroupFaceDic, validationFeatInDic, validationMainDic)
    rmseWeightedMeanEncoding = np.sqrt(mean_squared_error(validationGroupLabels, validationWeightedMeanEncodingPrediction))
    print("Root Mean squared error using weighted meaning encoding: %f" % rmseWeightedMeanEncoding)


    # mean of face-level estimations
    validationFaceMeanPrediction = groupPredict(validationGroup, validationGroupFaceDic, validationFeatInDic, validationGroupLabels, clf)
    rmseFaceMean = np.sqrt(mean_squared_error(validationGroupLabels, validationFaceMeanPrediction))
    print("Root Mean squared error with mean of face-level estimations: %f" % rmseFaceMean)


    # weighted mean of face-level estimations
    validationWeightedFaceMeanPrediction, weights, weightedPredictions = groupPredictWeighted(validationGroup, validationGroupFaceDic, validationFeatInDic, validationGroupLabels, clf, validationMainDic, 1, 1)
    rmseWeightedFaceMean = np.sqrt(mean_squared_error(validationGroupLabels, validationWeightedFaceMeanPrediction))
    print("Root Mean squared error with weighted mean of face-level estimations: %f" % rmseWeightedFaceMean)

    return validationMeanEncodingPrediction, rmseMeanEncoding, validationWeightedMeanEncodingPrediction, rmseWeightedMeanEncoding, validationFaceMeanPrediction, rmseFaceMean, validationWeightedFaceMeanPrediction, rmseWeightedFaceMean



dataMainPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/HappeiDetectedFaces/HAPPEI_MAIN_INFO.mat'
allPath = "/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/HappeiDetectedFaces/HAPPEI_DATA_NEW.mat"


# preloaded data
dataMain = scio.loadmat(dataMainPath)
allData = scio.loadmat(allPath)

# train
trainMain = dataMain['TR_Main_Info']
trainAllData = allData['DATA_TR_NEW']
trainMainDic = {} # faceName as key, intensity, fileName, locations as values
trainGroupFaceDic = {} # fileName as key, faceName as values
trainGroup = [] # a list of group files (as a order) (group that does not have face will be removed)
trainGroupLabels = [] # a list of group labels under trainGroup order (group that does not have face will be removed)

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

# get trainGroupFaceDic and trainGroupLabels
for element in trainAllData[1:]:
    fileName = element[0][0].encode('ascii')
    groupLabel = element[1][0][0]

    faces = element[3]
    faceFileNames = faces[1:, 12]
    locations = faces[1:, 0]

    faceFileList = []
    hasFace = False
    index = 0
    for faceFileName in faceFileNames:
        hasFace = True
        thisFaceFileName = faceFileName[0].encode('ascii')
        faceFileList.append(thisFaceFileName)

        # add bound and vertex into trainMainDic
        trainMainDic[thisFaceFileName].append(locations[index][0])
        index = index + 1

    if hasFace:
        value = {'face': faceFileList}
        value['groupLabel'] = groupLabel

        trainGroupFaceDic[fileName] = value
        trainGroup.append(fileName)
        trainGroupLabels.append(groupLabel)



# validation
validationMain = dataMain['VA_Main_Info']
validationAllData = allData['DATA_VA_NEW']
validationMainDic = {}  # faceName as key, intensity and fileName as values
validationGroupFaceDic = {}  # fileName as key, faceName as values
validationGroup = []  # a list of group files (as a order)
validationGroupLabels = []  # a list of group labels under validationGroup order
validationMainDic = {}

# convert validationMain to dic
for sample in validationMain:
    # i = i + 1
    # print(i)
    faceFileName = sample[0][0]
    faceFileName = faceFileName.encode("ascii")

    fileName = sample[1][0]
    fileName = fileName.encode("ascii")

    if sample[2].size != 0:
        label = sample[2][0][0]
    else:
        label = -1

    validationMainDic[faceFileName] = [fileName, label]

# get trainGroupFaceDic and trainGroupLabels
for element in validationAllData[1:]:
    fileName = element[0][0].encode('ascii')
    groupLabel = element[1][0][0]

    faces = element[3]
    faceFileNames = faces[1:, 12]
    locations = faces[1:, 0]

    faceFileList = []
    hasFace = False
    index = 0
    for faceFileName in faceFileNames:
        hasFace = True
        thisFaceFileName = faceFileName[0].encode('ascii')
        faceFileList.append(thisFaceFileName)

        # add bound and vertex into validationMainDic
        validationMainDic[thisFaceFileName].append(locations[index][0])
        index = index + 1

    if hasFace:
        value = {'face': faceFileList}
        value['groupLabel'] = groupLabel

        validationGroupFaceDic[fileName] = value
        validationGroup.append(fileName)
        validationGroupLabels.append(groupLabel)


# # test train_1nets_LSTM_features
# trainPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/LSTM_features2/train_1nets_LSTM_features.npy'
# valPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/LSTM_features2/val_1nets_LSTM_features.npy'
#
# validationMeanEncodingPrediction1, rmseMeanEncoding1, validationWeightedMeanEncodingPrediction1, rmseWeightedMeanEncoding1, validationFaceMeanPrediction1, rmseFaceMean1, validationWeightedFaceMeanPrediction1, rmseWeightedFaceMean1 = testAll(trainPath, valPath)
#
#
# # test train_2nets_LSTM_features
# trainPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/LSTM_features2/train_2nets_LSTM_features.npy'
# valPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/LSTM_features2/val_2nets_LSTM_features.npy'
#
# validationMeanEncodingPrediction2, rmseMeanEncoding2, validationWeightedMeanEncodingPrediction2, rmseWeightedMeanEncoding2, validationFaceMeanPrediction2, rmseFaceMean2, validationWeightedFaceMeanPrediction2, rmseWeightedFaceMean2 = testAll(trainPath, valPath)
#
#
# # test train_3nets_LSTM_features
# trainPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/LSTM_features2/train_3nets_LSTM_features.npy'
# valPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/LSTM_features2/val_3nets_LSTM_features.npy'
#
# validationMeanEncodingPrediction3, rmseMeanEncoding3, validationWeightedMeanEncodingPrediction3, rmseWeightedMeanEncoding3, validationFaceMeanPrediction3, rmseFaceMean3, validationWeightedFaceMeanPrediction3, rmseWeightedFaceMean3 = testAll(trainPath, valPath)
#
#
# # test train_4nets_LSTM_features
# trainPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/LSTM_features2/train_4nets_LSTM_features.npy'
# valPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/LSTM_features2/val_4nets_LSTM_features.npy'
#
# validationMeanEncodingPrediction4, rmseMeanEncoding4, validationWeightedMeanEncodingPrediction4, rmseWeightedMeanEncoding4, validationFaceMeanPrediction4, rmseFaceMean4, validationWeightedFaceMeanPrediction4, rmseWeightedFaceMean4 = testAll(trainPath, valPath)
#
#
# test train_5nets_LSTM_features
# trainPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/LSTM_features2/train_5nets_LSTM_features.npy'
# valPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/LSTM_features2/val_5nets_LSTM_features.npy'
#
# validationMeanEncodingPrediction5, rmseMeanEncoding5, validationWeightedMeanEncodingPrediction5, rmseWeightedMeanEncoding5, validationFaceMeanPrediction5, rmseFaceMean5, validationWeightedFaceMeanPrediction5, rmseWeightedFaceMean5 = testAll(trainPath, valPath)


# # test train_1nets_all_data_LSTM_features
# trainPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/lstm_all_data/train_1nets_all_data_LSTM_features.npy'
# valPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/lstm_all_data/val_1nets_all_data_LSTM_features.npy'
#
# validationMeanEncodingPredictionA1, rmseMeanEncodingA1, validationWeightedMeanEncodingPredictionA1, rmseWeightedMeanEncodingA1, validationFaceMeanPredictionA1, rmseFaceMeanA1, validationWeightedFaceMeanPredictionA1, rmseWeightedFaceMeanA1 = testAll(trainPath, valPath)
