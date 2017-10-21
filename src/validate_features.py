import tensorflow as tf
import numpy as np
import argparse
from LSTM import LSTM
from sklearn.externals import joblib

rootPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/my-group-emotion-analysis-master/'

# data1 = np.load("features/train_5nets_feature.npy")
# feat1 = None
# for element in data1:
#     if element['name'] == '389679202_326a1884d4_183_33403234@N00.xml_1.jpg':
#         feat1 = element
# feat1_1 = feat1['feature'][0]
#
# data2 = np.load("features/train_5nets_feature_correct.npy")
# feat2 = None
# for element in data2:
#     if element['name'] == '389679202_326a1884d4_183_33403234@N00.xml_1.jpg':
#         feat2 = element
# feat2_1 = feat2['feature'][0]




data1 = np.load('features/train_5nets_LSTM_features_correct.npy')
data1 = data1.item()
feat1 = data1['389757423_76e0c258fe_128_95849130@N00.xml_3.jpg']

data2 = np.load('features/train_5nets_LSTM_features_to_validate6.npy')
data2 = data2.item()
feat2 = data2['389757423_76e0c258fe_128_95849130@N00.xml_3.jpg']
#
# data3 = np.load('features/train_5nets_LSTM_features_to_validate2.npy')
# data3 = data3.item()
# feat3 = data3['389679202_326a1884d4_183_33403234@N00.xml_1.jpg']
#
# data4 = np.load('features/train_5nets_LSTM_features_to_validate3.npy')
# data4 = data4.item()
# feat4 = data4['389679202_326a1884d4_183_33403234@N00.xml_1.jpg']
#
# data5 = np.load('features/train_5nets_LSTM_features_to_validate4.npy')
# data5 = data5.item()
# feat5 = data5['389679202_326a1884d4_183_33403234@N00.xml_1.jpg']
#
# data6 = np.load('features/train_5nets_LSTM_features_to_validate5.npy')
# data6 = data6.item()
# feat6 = data6['389679202_326a1884d4_183_33403234@N00.xml_1.jpg']



# data1 = np.load('features/train_5nets_LSTM_features_correct.npy')
# data1 = data1.item()
# feat1 = data1['389757423_76e0c258fe_128_95849130@N00.xml_2.jpg']
# #
# # feat2 = data1['389757423_76e0c258fe_128_95849130@N00.xml_2.jpg']
# #
# # feat3 = data1['389757423_76e0c258fe_128_95849130@N00.xml_3.jpg']
# #
# # individualModel = joblib.load("/Volumes/Extend/20170212Extend/git/GREP_RESTful/models/individual_5netsRRDE_SVR.pkl")
# #
# # prediction1 = individualModel.predict(feat1)
# # print(prediction1)
# # prediction2 = individualModel.predict(feat2)
# # print(prediction2)
# # prediction3 = individualModel.predict(feat3)
# # print(prediction3)
#
# data6 = np.load('features/train_5nets_LSTM_features_to_validate3.npy')
# data6 = data6.item()
# feat6 = data6['389757423_76e0c258fe_128_95849130@N00.xml_2.jpg']
#
# data7 = np.load('features/train_5nets_LSTM_features_to_validate4.npy')
# data7 = data7.item()
# feat7 = data7['389757423_76e0c258fe_128_95849130@N00.xml_2.jpg']
#
individualModel = joblib.load("/Volumes/Extend/20170212Extend/git/GREP_RESTful/models/individual_5netsRRDE_SVR.pkl")

prediction1 = individualModel.predict(feat1)
print(prediction1)
prediction2 = individualModel.predict(feat2)
print(prediction2)

# prediction6 = individualModel.predict(feat6)
# print(prediction6)
# prediction7 = individualModel.predict(feat7)
# print(prediction7)


# # look at CNN features
# cnn_correct = np.load("features/train_5nets_feature_correct.npy")
# cnn_mine = np.load("features/train_5nets_feature2.npy")
# cnn_mine2 = np.load("features/train_5nets_feature3.npy")
# cnn_mine3 = np.load("/Users/PAN/Downloads/group-emotion-analysis-master/features/train_5nets_feature.npy")
# cnn_mine4 = np.load("/Users/PAN/Downloads/group-emotion-analysis-master/features/train_5nets_feature2.npy")
# x=cnn_correct[0]['feature'][0]
# y=cnn_mine[0]['feature'][0]
# z=cnn_mine2[0]['feature'][0]
# l=cnn_mine3[0]['feature'][0]
# m=cnn_mine4[0]['feature'][0]
# for i in range(0, len(cnn_correct)):
#     print("###")
#     for j in range(0, 5):
#         print(cnn_correct[i]['feature'][j]==cnn_mine[i]['feature'][j])
#     print("###")