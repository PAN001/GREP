import scipy.io as sio
import numpy as np

# train
trainTextFilePath = "/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/HappeiDetectedFaces/train.txt"
trainFeaturePath = '/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/features/train.mat'
trainFeature = sio.loadmat(trainFeaturePath)


DATA_TR = np.zeros((1, 4), dtype=np.object)
DATA_TR[0][0] = 'Filename'
DATA_TR[0][1] = 'GT_intensty'
DATA_TR[0][2] = 'TR/VA/TE'
DATA_TR[0][3] = 'Faces'

file = open(trainTextFilePath)
line = file.readline()
numOfNoLabel = 0
while line:
    lineSplitted = line.split(" ")

    if lineSplitted[0]:
        thisImage = np.zeros((1,4), dtype=np.object)
        groupImageName = lineSplitted[0]
        groupIntensity = lineSplitted[1]

        faces = np.zeros((1, 3), dtype=np.object)
        faces[0][0] = 'FileName'
        faces[0][1] = 'GT_face_intensity'
        faces[0][2] = 'Feature'
        i = 2
        while i < (len(lineSplitted) - 1):
            thisFace = faces = np.zeros((1,3), dtype=np.object)
            faceImageName = lineSplitted[i]
            faceIntensity = lineSplitted[i + 1]

            if faceImageName in trainFeature:
                faceImageFeature = trainFeature[faceImageName]['feature'][0][0][0]

                thisFace[0][0] = faceImageName
                thisFace[0][1] = faceIntensity
                thisFace[0][2] = faceImageFeature

                faces = np.concatenate((faces, thisFace))
            else:
                print('file: ' + faceImageName + ' does not exist')
                numOfNoLabel = numOfNoLabel + 1
                break

            i = i + 2

        thisImage[0][0] = groupImageName
        thisImage[0][1] = groupIntensity
        thisImage[0][2] = 'TR'
        thisImage[0][3] = faces

        DATA_TR = np.concatenate((DATA_TR, thisImage))

    line = file.readline()





# oneSample = data['389870598_f400878dc5_177_49207252@N00.xml.jpg_7.jpg']
# imageFeature = oneSample['feature'][0][0][0]
# imageFeature = imageFeature.tolist()
# label = oneSample['label'][0][0][0][0]