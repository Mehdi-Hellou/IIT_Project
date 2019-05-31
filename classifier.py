import cv2
import os 
from lbph import LocalBinaryPatterns
from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.feature_selection import SelectKBest, RFE, chi2, mutual_info_classif
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy
from imutils import paths

if __name__ == "__main__":
    desc = LocalBinaryPatterns(24, 8)   #the descripter use to calculate the 
                                        # local binary pattern histogram (LBPH)
 
    data = []            # List to store the data of each image 
    labels = []          # Lis that permit to store the label of each image corresponding
    
    X_train = np.empty((0,26)) # a numpy array that permit to store the features of each train image in a way
                               # that we can use the library scikit learn on it and that we are able to visualize 
                               # the data on figures
    train_path_images = [i for i in paths.list_images("images/train")] #Path to train images
    test_path_images =  [i for i in paths.list_images("images/test")] #Path to test images                      
    for image_path in train_path_images: 
    	image = cv2.imread(image_path)              # Variable to store the data of the image 
    	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # We convert the image in gray 
    	hist = desc.describe(gray)                     # We get the histograme of gray image 
    	X_train = np.vstack([X_train,hist])         # We add the the values of histogramme in the numpy array
    	labels.append(image_path.split(os.path.sep)[-2])  #We append the label corresponding to the histogramme
    	data.append(hist)
    
    print(X_train.shape)
    y_train = [1 if i=="apple" else 0 for i in labels]   # The label of each image convert in int 
    # print(labels)
        
    # We create a list of classifier to compare after and see which one of them has the best performance 
    clf_list = [AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=50),  
                LinearSVC(C=1.0, random_state=42), 
                SVC(gamma ='auto'), 
                MLPClassifier(int(1e2),activation='tanh',max_iter=100)] 
    
    y_test = []   #List where we're going to store the true label of test data
    X_test = np.empty((0,26))   #numpy array to store the LBPH informations of test image 
    
    labels = []
    # loop over the testing images
    for imagePath in test_path_images:
    	# load the image, convert it to grayscale, describe it,
    	# and classify it
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
#        prediction = model.predict(hist.reshape(1, -1))
        X_test = np.vstack([X_test,hist.reshape(1, -1)])         # We add the the values of histogramme in the numpy array
        labels.append(imagePath.split(os.path.sep)[-2])  #We append the string label corresponding to the histogramme
#    	# display the image and the prediction
#        cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#    		1.0, (0, 0, 255), lineType=cv2.LINE_AA)
#        cv2.imshow("Image", image)
#        cv2.waitKey(0)
   
    err_clf = 1.0
    
    y_test = [1 if i=="apple" else 0 for i in labels]   # The label of each image convert in int 
                                                        # 1 for apple and 0 for tomatoes
    
    model = None
    for clf in clf_list: 
        clf.fit(X_train,y_train)
        
        y_pred, y_true = clf.predict(X_test),y_test           # y_pred is the prediction of our output 
        tp = [1 for k in range(len(y_pred)) if y_true[k]==y_pred[k]]
        tp = numpy.sum(tp)
        acc = tp/float(len(y_pred))
        err = 1-acc
        
        print('The current accuracy of : ',clf.__class__.__name__,' is : ',acc)
        if err_clf > err:
            err_clf = err
            model = clf 
            
    
    
    color = numpy.array([x for x in "cmkr"])
     
    #Analyse en composante principale 
    pca = PCA(n_components = 2) 
    pca.fit(X_train)
    X_pca  =pca.transform(X_train)
    pyplot.title(" Projection des données en 2 dimensions à l'aide de l'Analyse en composante principale ")
    pyplot.scatter(X_pca[:, 0], X_pca[:, 1], c=color[y_train].tolist(), cmap=pyplot.cm.nipy_spectral,
           edgecolor='k')
    pyplot.show()
    
    #positionnement multidimensionnel (MDS)
    mds = MDS(n_components=2, n_init=1)
    X_mds = mds.fit_transform(X_train)
    pyplot.title(" Projection of data in 2 dimensions using multidimensional positioning (MDS) ")
    pyplot.scatter(X_mds[:, 0], X_mds[:, 1], c=color[y_train].tolist(), cmap=pyplot.cm.nipy_spectral,
           edgecolor='k')
    
    pyplot.show()
    
    #t-SNE
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X_train)
    pyplot.title(" Projection des données en 2 dimensions à l'aide du t-SNE")
    pyplot.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color[y_train].tolist(), cmap=pyplot.cm.nipy_spectral,
           edgecolor='k')
    pyplot.show()
       
    X_Train_embedded = X_tsne
    model.fit(X_train,y_train)
    y_predicted = model.predict(X_train)
    # create meshgrid
    resolution = 100 # 100x100 background pixels
    X2d_xmin, X2d_xmax = np.min(X_Train_embedded[:,0]), np.max(X_Train_embedded[:,0])
    X2d_ymin, X2d_ymax = np.min(X_Train_embedded[:,1]), np.max(X_Train_embedded[:,1])
    xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))
    
    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    background_model = KNeighborsClassifier(n_neighbors=1).fit(X_Train_embedded, y_predicted) 
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((resolution, resolution))
    
    fig = pyplot.figure()
    fig.set_size_inches(10.5, 8.5)

    ax = fig.add_subplot(211) #small subplot to show how the legend has moved. 
    #plot
    ax.contourf(xx, yy, voronoiBackground)
    ax.set_title(" Boundaries decision in using the dimensionality reduction of Multidimensional scaling")
    ax.scatter(X_Train_embedded[:,0], X_Train_embedded[:,1], c=color[y_train].tolist())
    
    label =numpy.array([x for x in ["Apple","Tomatoes"]])
    # Legend
    for ind, s in enumerate(label):
        ax.scatter([], [], label=s, color=color[ind])
        
    pyplot.legend(scatterpoints=1, frameon=True, labelspacing=0.5
               , bbox_to_anchor=(1.2, .4) , loc='center right')

    pyplot.tight_layout()
    pyplot.savefig("BoundariesDecision.PNG")
    pyplot.show()
    