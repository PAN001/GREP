from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import shutil

def dataAugmentation(imagePath, savePrefix, saveToDir):
	datagen = ImageDataGenerator(
		zca_whitening = True,
		rotation_range = 0.1,
		width_shift_range = 0.1,
		height_shift_range = 0.1,
		shear_range = 0.1,
		zoom_range = 0.2,
		horizontal_flip = False,
		fill_mode = 'nearest')

	img = load_img(imagePath)  # this is a PIL image, please replace to your own file path
	x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
	x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

	# the .flow() command below generates batches of randomly transformed images
	# and saves the results to the `preview/` directory

	i = 0
	for batch in datagen.flow(x, batch_size = 1):
		newImage = array_to_img(batch[0])
		newImageName = savePrefix + "_" + str(i) + ".jpg"
		newImage.save(saveToDir + newImageName)
		i += 1
		if i >= 2:
			break  # otherwise the generator would loop indefinitely


originAddress = "/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/HappeiDetectedFaces/"
trainOriginAddress = originAddress + "train/"
validationOriginAddress = originAddress + "validation/"

targetAddress = "/Volumes/Extend/20170212Extend/Document/UNNC/Current/Year_4_Autumn/AE3IDS/Project/FYP/HappeiDetectedFaces_augmented/"
os.mkdir(targetAddress)

trainTargetAddress = targetAddress + "train/"
validationTargetAddress = targetAddress + "validation/"


# for train
os.mkdir(trainTargetAddress)
file = open(originAddress + "train.txt")
line = file.readline()

trainLabels = ""

while line:
	lineSplitted = line.split(" ")
	print(line)

	if lineSplitted[0]:
		folderName = lineSplitted[0]
		groupIntensity = lineSplitted[1]
		folderOriginAddress = trainOriginAddress + folderName + "/"
		folderTargetAddress = trainTargetAddress + folderName + "/"
		os.mkdir(folderTargetAddress)
		trainLabels += folderName + " " + groupIntensity + " "

		i = 2
		while i < (len(lineSplitted) - 1):
			image = lineSplitted[i]
			label = lineSplitted[i+1]
			trainLabels += image + " " + label + " "

			imageOriginAddress = folderOriginAddress + image
			shutil.copyfile(imageOriginAddress, folderTargetAddress + image)

			dataAugmentation(imageOriginAddress, image, folderTargetAddress)

			for j in range (0, 2):
				trainLabels += image + "_" + str(j) + ".jpg" + " " + label + " "

			i = i + 2

	trainLabels += "\n"
	line = file.readline()

fileToWrite = open(targetAddress + "train.txt", "w")
fileToWrite.write(trainLabels)

file.close()

# for validation
os.mkdir(validationTargetAddress)
file = open(originAddress + "validation.txt")
line = file.readline()

validationLabels = ""

while line:
	lineSplitted = line.split(" ")
	print(line)

	if lineSplitted[0]:
		folderName = lineSplitted[0]
		groupIntensity = lineSplitted[1]
		folderOriginAddress = validationOriginAddress + folderName + "/"
		folderTargetAddress = validationTargetAddress + folderName + "/"
		os.mkdir(folderTargetAddress)
		validationLabels += folderName + " " + groupIntensity + " "

		i = 2
		while i < (len(lineSplitted) - 1):
			image = lineSplitted[i]
			label = lineSplitted[i+1]
			validationLabels += image + " " + label + " "

			imageOriginAddress = folderOriginAddress + image
			shutil.copyfile(imageOriginAddress, folderTargetAddress + image)

			dataAugmentation(imageOriginAddress, image, folderTargetAddress)

			for j in range (0, 2):
				validationLabels += image + "_" + str(j) + ".jpg" + " " + label + " "

			i = i + 2

		validationLabels += "\n"
	line = file.readline()

fileToWrite = open(targetAddress + "validation.txt", "w")
fileToWrite.write(validationLabels)

file.close()


