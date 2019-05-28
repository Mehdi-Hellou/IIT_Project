import cv2
import os 
import glob 
from lbph import LocalBinaryPatterns
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import train_test_split

desc = LocalBinaryPatterns(24, 8)

data = []
labels = []

X = np.empty((0,26))
for image_path in glob.glob('images/*/*'): 
	image = cv2.imread(image_path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	X = np.vstack([X,hist])
	labels.append(image_path.split(os.path.sep)[-2])
	data.append(hist)
print(X.shape)
# labels = [1 if i=="apple" else 0 for i in labels]
# print(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)

# train a Linear SVM on the data
model = LinearSVC(C=1.0, random_state=42)

model.fit(X_train,y_train)

y_predict = model.predict(X_test)

accuracy = [1 if y_predict[i]==y_test[i] else 0 for i in range(0,len(X_test)) ]

print(accuracy)
print(sum(accuracy)/len(accuracy))