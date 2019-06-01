# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 16:42:51 2019

@author: Mehdi
"""
"""
Create a function that choose 10 images among test images and show the prediction into the image. 
See the final result with a list of examples images 
"""
import cv2
import os 
from imutils import paths
import pickle
from lbph import LocalBinaryPatterns
import random

if __name__ == "__main__":
    
    desc = LocalBinaryPatterns(24, 8)   #the descripter use to calculate the 
                                        # local binary pattern histogram (LBPH)
    test_path_images =  [i for i in paths.list_images("images/test")] #list of the path to test images   
    train_path_images = [i for i in paths.list_images("images/train")] #list of the path to train images
    
    test_images = random.choices(test_path_images, k=10)
    
    model = pickle.load(open("finalized_model.sav", 'rb'))   # We load our classifier
    
    labels = []
    data= []
    for image_path in train_path_images: 
    	image = cv2.imread(image_path)              # Variable to store the data of the image 
    	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # We convert the image in gray 
    	hist = desc.describe(gray)                     # We get the histograme of gray image 
    	labels.append(image_path.split(os.path.sep)[-2])  #We append the label corresponding to the histogramme
    	data.append(hist)
    
    model.fit(data,labels)
    
    for imagePath in test_images:
        # load the image, convert it to grayscale, describe it,
    	# and classify it
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        prediction = model.predict(hist.reshape(1, -1))
    	# display the image and the prediction
        cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
    		1.0, (0, 0, 255), lineType=cv2.LINE_AA)
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
        # cv2.resizeWindow("output", 400, 300)              # Resize window to specified dimensions
        cv2.imshow("output", image)
        cv2.waitKey(0)