# IIT_Project
This github is the building of a binary classifier for apple's image and tomato's image. We use a SVM classifier on a dataset 
containing 100 training images (50 of apple and 50 of tomato) and 60 test images (30 of apple and 30 of tomato).

# lbph.py

Local Binary Patter Histogram classes, which permit to determine the LBPH descriptors for an image. 


# classifier.py 
Main python code that create the SVM model and print the different figures 
for the accuracy of the model and the  boundaries decision. 


# visualize_prediction_image.py 

python script which choose 10 randomly images in the test dataset and print the prediction of the model onto the image.
