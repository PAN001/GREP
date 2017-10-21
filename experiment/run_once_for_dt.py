from GEMs import *

trainPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/LSTM_features2/train_5nets_LSTM_features.npy'
valPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/LSTM_features2/val_5nets_LSTM_features.npy'

trainFeat = np.load(trainPath)
trainFeatInDic = trainFeat.item()

trainFeatList, trainLabelList, trainFeatListWithoutUnlabeled, trainLabelListWithoutUnlabeled = getFeatureAndLabel(
    trainMainDic, trainFeatInDic)

validationFeat = np.load(valPath)
validationFeatInDic = validationFeat.item()

validationFeatList, validationLabelList, validationFeatListWithoutUnlabeled, validationLabelListWithoutUnlabeled = getFeatureAndLabel(
    validationMainDic, validationFeatInDic)

# individual level
# SVM training
clf = SVR(C=0.5, cache_size=200, coef0=0.0, degree=3, epsilon=0.13, gamma='auto', kernel='linear')
print("train starts")
clf.fit(trainFeatListWithoutUnlabeled, trainLabelListWithoutUnlabeled)
print("train ends")

# individual face level
rmseFaceLevel, accuracyFaceLevel = individualFacePredict(validationFeatListWithoutUnlabeled,
                                                         validationLabelListWithoutUnlabeled, clf)

# group level

# mean encoding
# SVM training
model = meanEncodingTrain(trainGroup, trainGroupFaceDic, trainFeatInDic, trainGroupLabels)
validationMeanEncodingPrediction = meanEncodingPredict(validationGroup, validationGroupFaceDic, validationFeatInDic,
                                                       validationGroupLabels, model)
# # The mean squared error
# rmseMeanEncoding = np.sqrt(mean_squared_error(validationGroupLabels, validationMeanEncodingPrediction))
# print("Root Mean squared error with mean encoding %f" % rmseMeanEncoding)

# weighted mean encoding
validationWeightedMeanEncodingPrediction = weightedMeanEncoding(trainGroup, trainGroupFaceDic, trainFeatInDic,
                                                                trainMainDic, trainGroupLabels, validationGroup,
                                                                validationGroupFaceDic, validationFeatInDic,
                                                                validationMainDic)

rmseWeightedMeanEncoding = np.sqrt(mean_squared_error(validationGroupLabels, validationWeightedMeanEncodingPrediction))
print("Root Mean squared error using weighted meaning encoding: %f" % rmseWeightedMeanEncoding)

# mean of face-level estimations
validationFaceMeanPrediction = groupPredict(validationGroup, validationGroupFaceDic, validationFeatInDic,
                                            validationGroupLabels, clf)
rmseFaceMean = np.sqrt(mean_squared_error(validationGroupLabels, validationFaceMeanPrediction))
print("Root Mean squared error with mean of face-level estimations: %f" % rmseFaceMean)

# weighted mean of face-level estimations
validationWeightedFaceMeanPrediction, weights, weightedPredictions = groupPredictWeighted(validationGroup,
                                                                                          validationGroupFaceDic,
                                                                                          validationFeatInDic,
                                                                                          validationGroupLabels, clf,
                                                                                          validationMainDic, 1, 1)
rmseWeightedFaceMean = np.sqrt(mean_squared_error(validationGroupLabels, validationWeightedFaceMeanPrediction))
print("Root Mean squared error with weighted mean of face-level estimations: %f" % rmseWeightedFaceMean)






groupName = '488779608_ce29b2eda0_218_8166339@N08.xml.jpg'
gourpIndex = 324 # index in validationGroup

predictLabel = validationWeightedMeanEncodingPrediction[gourpIndex]
label = validationGroupLabels[gourpIndex]

face1Name = "488779608_ce29b2eda0_218_8166339@N08.xml_1.jpg"
face1Feat = validationFeatInDic[face1Name]
face1Label = validationMainDic[face1Name][1]

face2Name = "488779608_ce29b2eda0_218_8166339@N08.xml_2.jpg"
face2Feat = validationFeatInDic[face2Name]
face2Label = validationMainDic[face2Name][1]

face3Name = "488779608_ce29b2eda0_218_8166339@N08.xml_3.jpg"
face3Feat = validationFeatInDic[face3Name]
face3Label = validationMainDic[face3Name][1]

face4Name = "488779608_ce29b2eda0_218_8166339@N08.xml_4.jpg"
face4Feat = validationFeatInDic[face4Name]
face4Label = validationMainDic[face4Name][1]

face5Name = "488779608_ce29b2eda0_218_8166339@N08.xml_5.jpg"
face5Feat = validationFeatInDic[face5Name]
face5Label = validationMainDic[face5Name][1]

face6Name = "488779608_ce29b2eda0_218_8166339@N08.xml_6.jpg"
face6Feat = validationFeatInDic[face6Name]
face6Label = validationMainDic[face6Name][1]

face7Name = "488779608_ce29b2eda0_218_8166339@N08.xml_7.jpg"
face7Feat = validationFeatInDic[face7Name]
face7Label = validationMainDic[face7Name][1]

face8Name = "488779608_ce29b2eda0_218_8166339@N08.xml_8.jpg"
face8Feat = validationFeatInDic[face8Name]
face8Label = validationMainDic[face8Name][1]

# face9Name = "488779608_ce29b2eda0_218_8166339@N08.xml_9.jpg"
# face9Feat = validationFeatInDic[face9Name]
# face9Label = validationMainDic[face9Name][1]

face1Predict = clf.predict(face1Feat)
face2Predict = clf.predict(face2Feat)
face3Predict = clf.predict(face3Feat)
face4Predict = clf.predict(face4Feat)
face5Predict = clf.predict(face5Feat)
face6Predict = clf.predict(face6Feat)
face7Predict = clf.predict(face7Feat)
face8Predict = clf.predict(face8Feat)
# face9Predict = clf.predict(face9Feat)



