<img src="" alt="alt text" align="middle"/>

<p align="center"><i>Feature Extraction via Random Recurrent Deep Ensembles and its Application in Group-level Happiness Estimation</i></p>


## Table of Contents
1. [Abstract](#1-motivation)
2. [The Database](#2-the-database)
3. [The Model](#3-the-model)
	* [3.1 Input Layer](#31-input-layer)
	* [3.2 Convolutional Layers](#32-convolutional-layers)  
	* [3.3 Dense Layers](#33-dense-layers)
	* [3.4 Output Layer](#34-output-layer)
	* [3.5 Deep Learning](#35-deep-learning)
4. [Model Validation](#4-model-validation)
	* [4.1 Performance](#41-performance)
	* [4.2 Analysis](#42-analysis)
	* [4.3 Computer Vision](#43-computer-vision)
5. [The Apps](#5-the-apps)
	* [5.1 RESTful API](#51-restful-api)
	* [5.2 Interactive Web App](#52-interactive-web-app)
	* [5.3 Real-Time Prediction via Webcam](#53-real-time-prediction-via-webcam)
6. [About the Author](#7-about-the-author)
7. [References](#8-references)

## 1 Abstract
In this project, we present a novel ensemble framework to extract highly discriminative feature representation of image and its application for group-level happiness intensity prediction in wild. In order to generate enough diversity of decisions, we train n convolutional neural networks by bootstrapping the training set and extract n features for each image from them. A recurrent network is then used to remember which network extracts better feature and generate the final feature representation for one individual image. We use several group emotion models (GEM) to aggregate face features in a group and use fine-tuned support vector regression (SVR) to get the final results. Through extensive experiments, the great effectiveness of our Random Recurrent Deep Ensembles (RRDE) are demonstrated in both structural and decisional ways. Our best result yields a 0.55  root-mean-square error (RMSE) on validation set of HAPPEI dataset, significantly better than the baseline of 0.78. Meanwhile, I also build an online RESTful API called GREP (GRoup Emotion Parser) and a website for demonstration purpose. The whole project is based on the undergraduate thesis of ___Yichen PAN___.

## 2 Introduction
### 2.1 Motivation
Groups are a rich source of emotions. Automatic Group-level Emotion Analysis (GEA) in images is of great importance as it has a wide variety of applications. In e-learning applications, group-level affective computing can be used to adjust the presentation style of a computerized tutor when a learner is bored, interested, frustrated, or pleased. In terms of social monitoring, for example, a car can monitor the emotion of all occupants and engage in additional safety measures, such as alerting other vehicles if it detects the driver to be angry. With regard to management of images, dividing images into albums based on facial emotion is a good solution to searching, browsing and managing images in multi-media systems \cite{dhall2010facial}. It also has practical meaning in key-frame detection and event detection \cite{vandal2015event}.

### 2.2 Description of Proposed Approach
In this project, we demonstrate the efficacy of our proposed random recurrent deep ensembles on Group level emotion recognition sub-challenge based on HAPPEI database. The task is to infer the happiness intensity of the group as a whole on a scale from 0 to 5 from a bottom-up perspective. We first introduce our CNN ensembles to extract several efficiently representative features of each face. Then we conduct feature aggregation on each face using LSTM. The proposed method will selectively memorize (or forget) the components of features which are important (or less important). Then we conduct face-level estimation using our trained SVR on the compact feature representation of each face. Various group emotion models are explored, including mean encoding and weighted fusion framework based on top-down features, such as sizes of faces and the distances between them. Note that the proposed method caters to aggregating information from multiple sources including different number of faces in one image. Our best result exhibits a RMSE of 0.55 on the validation set of HAPPEI dataset, which compares favorably to the RMSE of 0.78 of the baseline.

### 2.3 Database: HAPPEI
The database used in the experiment is the HAPpy PEople Images (HAPPEI) database \cite{dhall2015automatic}, which is a fully annotated database (by human labelers) for the task of group happiness level prediction. Images in this database are collected from Flicker by using a Matlab based program that automatically searches particular images associated with groups of people and predefined events. For those downloaded images, a Viola-Jones object detector trained on different data is executed on the images. Only images containing more than one subject were kept. Figure 2.1 shows a collage of images from the HAPPEI database.
<p align="center">
<img src="img/HAPPEI_sample.jpg" width="600" align="middle"/>
<h4 align="center"> Figure 2.1 A collage of images from the HAPPEI database.</h4>
</p>

Every image in the data set is given a label between [0, 5]. These six discrete numbers correspond to six stages of happiness: Neutral, Small Smile, Large Smile, Small Laugh, Large Laugh and Thrilled, as shown in Figure 2.2. During the manual labelling stage, if the teeth of a member of a group were visible, the face was often labelled as a \emph{Laugh} (happiness intensity ranges from 3 to 4). If the mouth was open wide, the face was labelled as \emph{Thrilled} (of happiness intensity 5). A face with a closed mouth was assigned the label \emph{Smile} (happiness intensity ranges from 1 to 2). \cite{dhall2015automatic} The LabelMe \cite{russell2008labelme} based annotation tool was also used for labelling.
<p align="center">
<img src="img/HAPPEI.png" width="600" align="middle"/>
<h4 align="center"> Figure 2.2 Samples in HAPPEI database.</h4>
</p>
	
## 3. Background Knowledge
### 3.1 Feed-forward Neural Network
It is necessary to describe the basic feed-forward neural network before investigating into the more complex convolutional neural network. A detailed discussion of feed-forward neural network could be found in \cite{bishop2006pattern}.  Given a supervised learning scenario and a set of labeled data a neural network offers a way of representing a nonlinear function $h_W (x)$ of input vector variable x.  The function $h_W (x)$ is parameterized by a weights matrix $W$ that can be tuned to fit our data. Figure x shows a simple neural network architecture which consists of two input units or neurons.
<p align="center">
<img src="img/feedforward_NN.png" width="500" align="middle"/>
<h4 align="center"> Figure 3.1 Typical feed-forward neural network architecture.</h4>
</p>

### 3.2 Convolutional Neural Network
A convolutional neural network is a type of feed-forward neural network where the connectivity between neurons conforms to the organization of the animal visual cortex. There are three principal factors that distinguish the CNN from the simple feed-forward neural networks: local receptive fields, weight sharing, and spatial pooling or subsampling layers. Figure 3.2 shows a typical example of max pooling.
<p align="center">
<img src="img/CNN_pooling.png" width="500" align="middle"/>
<h4 align="center"> Figure 3.2 Typical max pooling layer in CNN.</h4>
</p>
		
In a typical CNN, there are multiple layers, alternating between convolution and pooling. From an intuitive perspective, the low-level convolutional filters, such as those in the first convolutional layer, can be thought of low-level enco ding of the input data. In terms of the image as input, these low-level filters may consist of simple edge filters. As for those higher layers in the neural network, the CNN begins to learn more and more complicated structures. With multiple layers and a large number of filters, the CNN architecture is capable of extracting powerful representation of input data.

### 3.3 Recurrent Neural Network: Long-Short Term Memory
Long short-term memory (LSTM) is a special kind of RNN architecture proposed in 1997 \cite{hochreiter1997long}. LSTMs are explicitly designed to deal with the long-term dependency problem. All RNNS have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module has a very simple structure, such as a single tanh layer.
<p align="center">
<img src="img/LSTM-SimpleRNN.png" width="300" align="middle"/>
<h4 align="center"> Figure 3.3 The single-layer repeating module in a standard RNN.</h4>
</p>

## 3 The Model


Deep learning is a popular technique used in computer vision. I chose convolutional neural network (CNN) layers as building blocks to create my model architecture. CNNs are known to imitate how the human brain works when analyzing visuals. I will use a picture of Mr. Bean as an example to explain how images are fed into the model, because who doesn’t love Mr. Bean?

A typical architecture of a convolutional neural network will contain an input layer, some convolutional layers, some dense layers (aka. fully-connected layers), and an output layer (Figure 4). These are linearly stacked layers ordered in sequence. In [Keras](https://keras.io/models/sequential/), the model is created as `Sequential()` and more layers are added to build architecture.

<p align="center">
<img src="https://github.com/JostineHo/mememoji/blob/master/figures/netarch.png" width="650" align="middle"/>
<h4 align="center">Figure 4. Facial Emotion Recognition CNN Architecture (modification from Eindhoven University of Technology-PARsE).</h4>
</p>

###3.1 Input Layer
+ The input layer has pre-determined, fixed dimensions, so the image must be __pre-processed__ before it can be fed into the layer. I used [OpenCV](http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0), a computer vision library, for face detection in the image. The `haar-cascade_frontalface_default.xml` in OpenCV contains pre-trained filters and uses `Adaboost` to quickly find and crop the face.
+ The cropped face is then converted into grayscale using `cv2.cvtColor` and resized to 48-by-48 pixels with `cv2.resize`. This step greatly reduces the dimensions compared to the original RGB format with three color dimensions (3, 48, 48).  The pipeline ensures every image can be fed into the input layer as a (1, 48, 48) numpy array.

###3.2 Convolutional Layers
+ The numpy array gets passed into the `Convolution2D` layer where I specify the number of filters as one of the hyperparameters. The __set of filters__(aka. kernel) are unique with randomly generated weights. Each filter, (3, 3) receptive field, slides across the original image with __shared weights__ to create a __feature map__.
+  __Convolution__ generates feature maps that represent how pixel values are enhanced, for example, edge and pattern detection. In Figure 5, a feature map is created by applying filter 1 across the entire image. Other filters are applied one after another creating a set of feature maps.

<p align="center">
<img src="https://github.com/JostineHo/mememoji/blob/master/figures/conv_maxpool.png" width="600" align="middle"/>
<h4 align="center">Figure 5. Convolution and 1st max-pooling used in the network</h4>
</p>

+ __Pooling__ is a dimension reduction technique usually applied after one or several convolutional layers. It is an important step when building CNNs as adding more convolutional layers can greatly affect computational time. I used a popular pooling method called `MaxPooling2D` that uses (2, 2) windows across the feature map only keeping the maximum pixel value. The pooled pixels form an image 
with dimentions reduced by 4.

###3.3 Dense Layers
+ The dense layer (aka fully connected layers), is inspired by the way neurons transmit signals through the brain. It takes a large number of input features and transform features through layers connected with trainable weights.

<p align="center">
<img src="https://github.com/JostineHo/mememoji/blob/master/figures/forward_back_prop.png" width="750" align="middle"/>
<h4 align="center">Figure 6. Neural network during training: Forward propagation (left) to Backward propagation (right).</h4>
</p>

+ These weights are trained by forward propagation of training data then backward propagation of its errors. __Back propagation__ starts from evaluating the difference between prediction and true value, and back calculates the weight adjustment needed to every layer before. We can control the training speed and the complexity of the architecture by tuning the hyper-parameters, such as __learning rate__ and __network density__. As we feed in more data, the network is able to gradually make adjustments until errors are minimized. 
+ Essentially, the more layers/nodes we add to the network the better it can pick up signals. As good as it may sound, the model also becomes increasingly prone to overfitting the training data. One method to prevent overfitting and generalize on unseen data is to apply __dropout__. Dropout randomly selects a portion (usually less than 50%) of nodes to set their weights to zero during training. This method can effectively control the model's sensitivity to noise during training while maintaining the necessary complexity of the architecture.

###3.4 Output Layer
+ Instead of using sigmoid activation function, I used **softmax** at the output layer. This output presents itself as a probability for each emotion class.
+ Therefore, the model is able to show the detail probability composition of the emotions in the face. As later on, you will see that it is not efficient to classify human facial expression as only a single emotion. Our expressions are usually much complex and contain a mix of emotions that could be used to accurately describe a particular expression.

> It is important to note that there is no specific formula to building a neural network that would guarantee to work well. Different problems would require different network architecture and a lot of trail and errors to produce desirable validation accuracy. __This is the reason why neural nets are often perceived as "black box algorithms."__ But don't be discouraged. Time is not wasted when experimenting to find the best model and you will gain valuable experience. 

###3.5 Deep Learning
I built a simple CNN with an input, three convolution layers, one dense layer, and an output layer to start with. As it turned out, the simple model preformed poorly. The low accuracy of 0.1500 showed that it was merely random guessing one of the six emotions. The simple net architecture failed to pick up the subtle details in facial expressions. This could only mean one thing...

<p align="center">
<img src="https://github.com/JostineHo/mememoji/blob/master/figures/inception.png" width="500" align="middle"/>
</p>

This is where deep learning comes in. Given the pattern complexity of facial expressions, it is necessary to build with a deeper architecture in order to identify subtle signals. So I fiddled combinations of three components to increase model complexity: 
+ __number and configuraton of convolutional layers__
+ __number and configuration of dense layers__ 
+ __dropout percentage in dense layers__

Models with various combinations were trained and evaluated using GPU computing `g2.2xlarge` on Amazon Web Services (AWS). This greatly reduced training time and increased efficiency in tuning the model (Pro tip: use _automation script_ and _tmux detach_ to train on AWS EC2 instance over night). In the end, my final net architecture was 9 layers deep in convolution with one max-pooling after every three convolution layers as seen in Figure 7.

<p align="center">
<img src="https://github.com/JostineHo/mememoji/blob/master/figures/mynetarch.png" width="850" align="middle"/>
<h4 align="center">Figure 7. Final model CNN architecture.</h4>
</p>

## 4 Model Validation
<p align="center">
<img src="https://github.com/JostineHo/mememoji/blob/master/figures/works_every_time.png" width="500" align="middle"/>
</p>

###4.1 Performance
As it turns out, the final CNN had a __validation accuracy of 58%__. This actually makes a lot of sense. Because our expressions usually consist a combination of emotions, and _only_ using one label to represent an expression can be hard. In this case, when the model predicts incorrectly, the correct label is often the __second most likely emotion__ as seen in Figure 8 (examples with light blue labels).

<p align="center">
<img src="https://github.com/JostineHo/mememoji/blob/master/figures/predictions.png" width="850" align="middle"/>
<h4 align="center">Figure 8. Prediction of 24 example faces randomly selected from test set.</h4>
</p>

###4.2 Analysis
 
<p align="center">
<img src="https://github.com/JostineHo/mememoji/blob/master/figures/confusion_matrix.png" width="400" align="middle"/>
<h4 align="center">Figure 9. Confusion matrix for true and prediction emotion counts.</h4>
</p>

Let's take a closer look at predictions for individual emotions. Figure 9 is the confusion matrix for the model predictions on the test set. The matrix gives the counts of emotion predictions and some insights to the performance of the multi-class classification model:
+ The model performs really well on classifying **positive emotions** resulting in relatively high precision scores for happy and surprised. **Happy** has a precision of 76.7% which could be explained by having the most examples (~7000) in the training set. Interestingly, **surprise** has a precision of 69.3% having the least examples in the training set. There must be very strong signals in the suprise expressions. 
+ Model performance seems weaker across **negative emotions** on average. In particularly, the emotion **sad** has a low precision of only 39.7%. The model frequently misclassified angry, fear and neutral as sad. In addition, it is most confused when predicting sad and neutral faces because these two emotions are probably the least expressive (excluding crying faces).  
+ Frequency of prediction that misclassified by less than 3 ranks.

<p align="center">
<img src="https://github.com/JostineHo/mememoji/blob/master/figures/correct_emotion.png" width="250" align="middle"/>
<h4 align="center">Figure 10. Correct predictions on 2nd and 3rd highest probable emotion.</h4>
</p>

###4.3 Computer Vision
As a result, the feature maps become increasingly abstract down the pipeline when more pooling layers are added. Figure 11 and 12 gives an idea of what the machine sees in feature maps after 2nd and 3rd max-pooling. [__Deep nets are beautiful!__](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html). 

**Code for analysis and visualiation of the inter-layer outputs in the convolutional neural net:** https://github.com/JostineHo/mememoji/blob/master/data_visualization.ipynb

<p align="center">
<img src="https://github.com/JostineHo/mememoji/blob/master/figures/L4_compviz.png" width="800"align="middle"/>
<h4 align="center">Figure 11. CNN (64-filter) feature maps after 2nd layer of max-pooling.</h4>
</p>

<p align="center">
<img src="https://github.com/JostineHo/mememoji/blob/master/figures/conv128pool3.png" width="600" align="middle"/>
<h4 align="center">Figure 12. CNN (128-filter) feature maps after 3nd layer of max-pooling.</h4>
</p>

## 5 The Apps
<p align="center">
<img src="https://github.com/JostineHo/mememoji/blob/master/figures/system.png" width="500" align="middle"/>
<h4 align="center">Figure 13. Web application and REST API.</h4>
</p>

###5.1 REST API
I built a REST API that finds human faces within images and make prediction about each facial emotion in `POST /v1.0.0/predict`. You can paste the url of an image in `image_url` or drag-and-drop an image file to `image_buf `. In addition, you have the option to have the API return the image with annotated faces and cropped thumbnail of each face in base64 by using the dropdown menu in `annotate_image` and `crop_image`. The API returns the probabilities of emotions for each face (indexed) and an unique ID for each image in json format. MongoDB is installed to store input into facial expression database on EC2 for future training. 

`POST /v1.0.0/feedback` can be used to collect user feedback from the web app for incorrect predictions. Developers have to option to send back user feedback (true emotion) by providing the unique ID and face index. The built-in MongoDB will use unique ID `image_id` to find the document and `face_index` to append the true emotion as `feedback` in the database.

**Source Code:** https://github.com/JostineHo/mememoji_api

**Demo:** [mememoji.rhobota.com](mememoji.rhobota.com)

###5.2 Interactive Web App
**Mememoji** is an interactive emotion recognition system that detects emotions based on facial expressions. This app uses the REST API to predict the compositions of the emotions expressed by users. Users have the option to paste image url, upload your own image, or simply turn on your webcam to interact with the app. Users can also provide feedback by selecting the correct emotion from a dropdown menu should the convolutional neural network predicts incorrectly. This will serve as a training sample and help improve the algorithm in the future.

_Special thanks to Chris Impicciche, Web Development Fellow at Galvanize, who made it possible for online demo of the technology._

**Source Code:** [FaceX](https://github.com/Peechiz/FaceX)

**Demo:** [mememoji.me](https://mememoji.me/)

###5.3 Real-Time Prediction via Webcam
In addition, I built a real-time facial emotion analyzer that can be accessed through a webcam. `real-time.py` overlays a meme face matching the emotion expressed in real-time. `live-plotting` outputs a live-recording graph that responds to the changes in facial expressions. The program uses OpenCV for face detection and the trained neural network for live prediction.

**Source Code**: [https://github.com/JostineHo/real-time_emotion_analyzer](https://github.com/JostineHo/real-time_emotion_analyzer)

## 6 About the Author

**Jostine Ho** is a data scientist who loves building intelligent applications and exploring the exciting possibilities using deep learning. She is interested in computer vision and automation that creates innovative solutions to real-world problems. She holds a masters degree in Petroleum & Geosystems Engineering at The University of Texas at Austin. You can reach her on [LinkedIn](https://www.linkedin.com/in/jostinefho).

## 7 References

1. [*"Dataset: Facial Emotion Recognition (FER2013)"*](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) ICML 2013 Workshop in Challenges in Representation Learning, June 21 in Atlanta, GA.

2. [*"Andrej Karpathy's Convolutional Neural Networks (CNNs / ConvNets)"*](http://cs231n.github.io/convolutional-networks/) Convolutional Neural Networks for Visual Recognition (CS231n), Stanford University.

3. Srivastava et al., 2014. *"Dropout: A Simple Way to Prevent Neural Networks from Overfitting"*, Journal of Machine Learning Research, 15:1929-1958.

4. Duncan, D., Shine, G., English, C., 2016. [*"Report: Facial Emotion Recognition in Real-time"*](http://cs231n.stanford.edu/reports2016/022_Report.pdf) Convolutional Neural Networks for Visual Recognition (CS231n), Stanford University.
