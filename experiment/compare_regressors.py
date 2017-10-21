from __future__ import division
from GEMs import *
from sklearn.cross_decomposition import PLSRegression

modelPath = '/Users/PAN/PycharmProjects/untitled/models/'
class_names = [0, 1, 2, 3, 4, 5]

# 5-networks
networks5TrainPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/LSTM_features2/train_5nets_LSTM_features.npy'
networks5ValidationPath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/LSTM_features2/val_5nets_LSTM_features.npy'

trainFeat = np.load(networks5TrainPath)
trainFeatInDic = trainFeat.item()
trainFeatList, trainLabelList, trainFeatListWithoutUnlabeled, trainLabelListWithoutUnlabeled = getFeatureAndLabel(trainMainDic, trainFeatInDic)

validationFeat = np.load(networks5ValidationPath)
validationFeatInDic = validationFeat.item()
validationFeatList, validationLabelList, validationFeatListWithoutUnlabeled, validationLabelListWithoutUnlabeled = getFeatureAndLabel(validationMainDic, validationFeatInDic)

# # individual SVR
# svrModel = SVR(C=0.5, cache_size=200, coef0=0.0, degree=3, epsilon=0.13, gamma='auto', kernel='linear')
# svrModel.fit(trainFeatListWithoutUnlabeled, trainLabelListWithoutUnlabeled)
# predictionSVR = svrModel.predict(validationFeatListWithoutUnlabeled)
# # rmse
# rmseSVR = np.sqrt(mean_squared_error(predictionSVR, validationLabelListWithoutUnlabeled)) # 0.86695545403728402
# # accuracy
# roundedPredictionSVR = roundPrediction(predictionSVR)
# comparison =  np.array(roundedPredictionSVR) == validationLabelListWithoutUnlabeled
# accuracySVR = (np.count_nonzero(comparison==True)) / len(comparison) # 0.5146135527170383
# # confusion matrix
# cmSVR = confusion_matrix(validationLabelListWithoutUnlabeled, roundedPredictionSVR)
# print(classification_report(validationLabelListWithoutUnlabeled, roundedPredictionSVR))
# np.set_printoptions(precision=2)
# # plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cmSVR, classes=class_names, title='Confusion matrix for indivisual happiness estimation using SVR')
# plt.show()

# # individual linear regression
# linearRModel = linear_model.LinearRegression()
# linearRModel.fit(trainFeatListWithoutUnlabeled, trainLabelListWithoutUnlabeled)
# predictionlinearR = linearRModel.predict(validationFeatListWithoutUnlabeled)
# # rmse
# rmselinearR = np.sqrt(mean_squared_error(predictionlinearR, validationLabelListWithoutUnlabeled)) # 0.86977538795635401
# # accuracy
# roundedPredictionlinearR  = roundPrediction(predictionlinearR)
# comparison =  np.array(roundedPredictionlinearR) == validationLabelListWithoutUnlabeled
# accuracylinearR = (np.count_nonzero(comparison==True)) / len(comparison)
# # confusion matrix
# cmlinearR = confusion_matrix(validationLabelListWithoutUnlabeled, roundedPredictionlinearR)
# np.set_printoptions(precision=2)
# # plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cmlinearR, classes=class_names, title='Confusion matrix for indivisual happiness estimation using Linear Regression')
# print(classification_report(validationLabelListWithoutUnlabeled, roundedPredictionlinearR))
# plt.show()

# # individual logistic regression
# params = {"C": [x/10.0 for x in range(5,15,1)], 'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag']}
# # logisticRModel = GridSearchCV(linear_model.LogisticRegression(), params, cv=5, n_jobs=-1, verbose = 10)
# logisticRModel = linear_model.LogisticRegression(C=1.0, solver='liblinear') # optimal parameters
# logisticRModel.fit(trainFeatListWithoutUnlabeled, trainLabelListWithoutUnlabeled)
# predictionlogisticR = logisticRModel.predict(validationFeatListWithoutUnlabeled)
# # rmse
# rmselogisticR = np.sqrt(mean_squared_error(predictionlogisticR, validationLabelListWithoutUnlabeled)) # 1.0023786388887026
# # accuracy
# roundedPredictionlogisticR  = roundPrediction(predictionlogisticR)
# comparison =  np.array(roundedPredictionlogisticR) == validationLabelListWithoutUnlabeled
# accuracylogisticR = (np.count_nonzero(comparison==True)) / len(comparison) # 0.5092011257848019
# # confusion matrix
# cmlogisticR = confusion_matrix(validationLabelListWithoutUnlabeled, roundedPredictionlogisticR)
# np.set_printoptions(precision=2)
# print(classification_report(validationLabelListWithoutUnlabeled, roundedPredictionlogisticR))
# # plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cmlogisticR, classes=class_names, title='Confusion matrix for indivisual happiness estimation using Logistic Regression')
# plt.show()

# individual PLS regression
params = {"n_components": [x for x in range(1, 10)]}
# PLSModel = GridSearchCV(PLSRegression(), params, cv=5, n_jobs=-1, verbose = 10)
PLSModel = PLSRegression(n_components=2) # optimal parameters
PLSModel.fit(trainFeatListWithoutUnlabeled, trainLabelListWithoutUnlabeled)
predictionPLS = PLSModel.predict(validationFeatListWithoutUnlabeled)
# normalize
min_max_scaler = preprocessing.MinMaxScaler((0, 5))
predictionPLSNormalized = min_max_scaler.fit_transform(predictionPLS)
# rmse
rmsePLS = np.sqrt(mean_squared_error(predictionPLSNormalized, validationLabelListWithoutUnlabeled)) # 1.2313264246706654
# accuracy
roundedPredictionPLS = roundPrediction(predictionPLSNormalized)
comparison =  np.array(roundedPredictionPLS) == validationLabelListWithoutUnlabeled
accuracyPLS = (np.count_nonzero(comparison==True)) / len(comparison) # 0.25005412426932240
# confusion matrix
cmPLS = confusion_matrix(validationLabelListWithoutUnlabeled, roundedPredictionPLS)
print(classification_report(validationLabelListWithoutUnlabeled, roundedPredictionPLS))
np.set_printoptions(precision=2)
# plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cmPLS, classes=class_names, title='Confusion matrix for indivisual happiness estimation using PLS Regression')
plt.show()