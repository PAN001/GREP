from GEMs import *
from sklearn.externals import joblib

def generateWeightedMeanEncodingGroupSVR(trainGroupList, trainGroupFaceDic, trainFaceFeatDic, trainMainDic, trainGroupLabels):
    model = SVR(C=0.5, cache_size=200, coef0=0.0, degree=3, epsilon=0.13, gamma='auto', kernel='linear')

    trainWeights, trainWeightedFeatures = generateMeanFeatureVector(trainGroupList, trainGroupFaceDic, trainFaceFeatDic, trainMainDic)

    model.fit(trainWeightedFeatures, trainGroupLabels)

    return model

# def main():
modelPath = '/Users/PAN/PycharmProjects/untitled/models/'

# 5-networks
networks5TrainPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/LSTM_features2/train_5nets_LSTM_features.npy'

trainFeat = np.load(networks5TrainPath)
trainFeatInDic = trainFeat.item()
trainFeatList, trainLabelList, trainFeatListWithoutUnlabeled, trainLabelListWithoutUnlabeled = getFeatureAndLabel(trainMainDic, trainFeatInDic)

# groupModel = generateWeightedMeanEncodingGroupSVR(trainGroup, trainGroupFaceDic, trainFeatInDic, trainMainDic, trainGroupLabels)
# joblib.dump(groupModel, modelPath + 'weighted_meaning_encoding_group_5netsRRDE_SVR.pkl')
#
# individual SVR
# individualModel = SVR(C=0.5, cache_size=200, coef0=0.0, degree=3, epsilon=0.13, gamma='auto', kernel='linear')
# individualModel.fit(trainFeatListWithoutUnlabeled, trainLabelListWithoutUnlabeled)
# # joblib.dump(individualModel, modelPath + 'individual_5netsRRDE_SVR.pkl')

individualModel = joblib.load(modelPath + 'individual_5netsRRDE_SVR.pkl')

face1 = trainFeatInDic['389679202_326a1884d4_183_33403234@N00.xml_1.jpg']
prediction1 = individualModel.predict(face1)