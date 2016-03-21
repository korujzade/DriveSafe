#DRIVE SAFE

##Problem

In today's world, passing vehicles in traffic, especially for a cyclist to pass a double-decker bus, has still being dangerous. Not paying proper attention could cause very disappointed results. While warnings on roads may help in this situation, a software written using Computer Vision and Machine Learning theories comes up to promising better results.

##Goal

The project “Drive Safe” is an application which recognises pre-defined object (bicycle in my case) from a dashboard camera of a vehicle and instantly informs a driver about this object. Main goal behind this project is to find a solution for making driving more secure by understanding and developing Computer Vision and Machine learning algorithms. Researching principles and algorithms to make the software as accurate and as fast as possible is another important challenge in the project.

##Solution

“Drive Safe” uses computer vision algorithms, and artificial intelligence techniques for classification data provided by a camera. Some external libraries like OpenCV are used during the development.

##How to Install and Use it?

- Prerequisites
 - OpenCV 3.0 (http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html)

- Use the system
  - cmake .
  - make
  - ./main

##How It Works?

- Briefly, I am using Histogram of Oriented Gradients to extract features from positve and negative images and used extracted values to train SVM model. Then testing data are used to evaluate and demonstrate the system.

  - When the program starts, it asks for paths to requested folders.

  - For the first time, it is necessary to extract features from positive and negative images and train descriptor values using Support Vectors Machine algorithm. So that you should provide paths to directory of negative images and positive images being used for training, and directory to store HOG descriptors and SVM model (path to xml files). 
  If you do not want to do it anyway, I have provided svm model trained with images and videos which I have taken on roads and made them suitable for training, and trained with images from http://drivesafe.orujzade.com. I have built that website so that users can upload images and test whether it is bicycle or not by using my system online. I have stored images which have been detected false and used for training this model mentioned above.

  - Then you can test and evaluate system using images, video or real time from a camera. For each of the choices, you should provide the paths to necessary directories.

  - You do not have to extract features and train new SVM model each time, if you are happy with training result.

- Notes
  - Annotations of the positive images basically describe the location of each object on each image with ground truth data. They are binary images, which white pixels show background and black pixels show an object. Most of the official datasets provide ground truth data of images. It makes your work easy as if you do not have annotations of positive images (ground truth data), you should provide bicycle images cropped from original images having monotone background on them.
  
  The example image and it's ground truth data are downloaded from the dataset collection provided by Institute of Electrical Measurement
  and Measurement Signal Processing, Graz University of Technology (http://www.emt.tugraz.at/~pinz/data/GRAZ_02/).


  - After extracting features and train SVM model with them, you can test system with three different ways: images, video and real time from a camera.
    If you choose images, you should provide paths to directories of positive and negative images being used for testing, and a path to a directory where extracted features and svm model are located. You can also see accuracy, specificity and sensitivity of the system on the result.
    For video and real time from a camera options, requested paths are similar to those mentioned above. In these options, detected parts from video/camera frames are saved to "false-negatives" directory (you should provide a path for that directory as well) so that they have being used for "hard negative mining" to improve system continuously, but you need to do it manually by choosing false negatives and false positives and add them relevant training directories.

 - You can also see svmligt library among source files. This library git cloned from https://github.com/LihO/SVMLightClassifier and I have made slight changes on it to fit my system. Even though results from using this classifier are not as good as my current system, I have kept this library for future experiments and comparisons.  

