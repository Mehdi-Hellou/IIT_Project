import cv2
import os 
from imutils import paths
import pickle
from lbph import LocalBinaryPatterns

import numpy
import numpy as np
from matplotlib import pyplot,cm
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import MDS
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def plot_boundaries_decision(X,y,clf,namefile): 
    """
    Method to plot the boundaries decision of our data 
    X : A numpy array of the data we want to plot 
    y : A numpy array of the  label corresponding to our data
    clf : the model use to predict the label of our data
    namefile : the name of the file in which we want to save the figure  
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)
    #    #The plot of boundary decision in the 2D space of representation of data
    model.fit(X_train,y_train)
    
    # create meshgrid
    resolution = 100 # 100x100 background pixels
    X2d_xmin, X2d_xmax = np.min(X[:,0]), np.max(X[:,0])
    X2d_ymin, X2d_ymax = np.min(X[:,1]), np.max(X[:,1])
    xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))
    
    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    background_model = KNeighborsClassifier(n_neighbors=1).fit(X, y) 
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((resolution, resolution))
    
    fig = pyplot.figure()
    fig.set_size_inches(10.5, 8.5)

    ax = fig.add_subplot(211) #small subplot to show how the legend has moved. 
    #plot
    ax.contourf(xx, yy, voronoiBackground)
    ax.set_title(" Boundaries decision in using the dimensionality reduction of Multidimensional scaling")
    ax.scatter(X[:,0], X[:,1], c=color[y].tolist())
    
    label =numpy.array([x for x in ["Apple","Tomatoes"]])
    # Legend
    for ind, s in enumerate(label):
        ax.scatter([], [], label=s, color=color[ind])
        
    pyplot.legend(scatterpoints=1, frameon=True, labelspacing=0.5
               , bbox_to_anchor=(1.2, .4) , loc='center right')
    
    pyplot.tight_layout()
    pyplot.savefig(namefile)
    pyplot.show()

def optimal_SVC_parameters(X,y,params_grid): 
    """
    A method that permit to determine the optimal parameters of a SVM
    the parameters gamma, C and the kernel use linear or or radial basis functions
    X: the data where the model is going to learn 
    y: labels of data 
    params_grid: dictionnary of each parameter that we want to optimize
    """
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=params_grid, cv=cv)
    grid.fit(X, y)
    
    best_param = grid.best_params_
    best_score = grid.best_score_
    print("The best parameters are %s with a score of %0.2f"
          % (best_param, best_score))
    
    return best_param.get('C'), best_param.get('gamma'), best_param.get('kernel') 
   
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
    
    y_train = [1 if i=="apple" else 0 for i in labels]   # The label of each image convert in int 
    color = numpy.array([x for x in "cmkr"])
       
    # We create a list of classifier to compare after and see which one of them has the best performance 
    clf_list = [AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=50),  
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
        X_test = np.vstack([X_test,hist.reshape(1, -1)])         # We add the the values of histogramme in the numpy array
        labels.append(imagePath.split(os.path.sep)[-2])  #We append the string label corresponding to the histogramme

    err_clf = 1.0
    
    y_test = [1 if i=="apple" else 0 for i in labels]   # The label of each image convert in int 
                                                        # 1 for apple and 0 for tomatoes
                                                        
    #We concantenate the datat train and test together for the plot of decision boundaries 
    X,y = numpy.concatenate((X_train,X_test)), numpy.concatenate((y_train,y_test)) 
    
    ### Step to decrease the dimension of the data so as to visualize 
    ### the ptojection of data into a 2D space 
    #Multidimensional scaling
    mds = MDS(n_components=2, n_init=1)
    X_2d = mds.fit_transform(X)
 
    fig = pyplot.figure()
    fig.set_size_inches(10.5, 8.5)

    ax = fig.add_subplot(211) #small subplot to show how the legend has moved. 
    #plot
    ax.set_title("Projection of data in 2 dimensions using multidimensional positioning (MDS)")
    ax.scatter(X_2d[:,0], X_2d[:,1], c=color[y].tolist())
    
    label =numpy.array([x for x in ["Apple","Tomatoes"]])
    # Legend
    for ind, s in enumerate(label):
        ax.scatter([], [], label=s, color=color[ind])
        
    pyplot.legend(scatterpoints=1, frameon=True, labelspacing=0.5
               , bbox_to_anchor=(1.2, .4) , loc='center right')

    pyplot.tight_layout()
    pyplot.savefig("MDS_2D.PNG")
    pyplot.show()
    
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
    
    ### We are going to determine the optimal parameters since we know at this moment
    ### That the best classifier we can use for this problem is a Support Vectore Machine (SVM)
    
    #Dictionary of possible parameters
    params_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.0001, 0.001, 0.01, 0.1],
              'kernel':['linear','rbf'] }
    
    C_opt, gamma_opt, kernel_opt = optimal_SVC_parameters(X_test,y_test,params_grid)
    
    model = SVC(C=C_opt,kernel = kernel_opt, gamma = gamma_opt)  # The model we choose is a SVM with the best parameters
    
    # save the optimal SVM model to disk
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    
    ####We are going to visualize the effect of differents parameters for the boundaries decisions 
    #List of SVM classifier
    classifiers = []
    for C in params_grid.get('C'):
        for gamma in params_grid.get('gamma'):
            clf = SVC(C=C, gamma=gamma, kernel = 'linear')
            clf.fit(X_train, y_train)
            classifiers.append((C, gamma, clf))
    
    scores = []
    list_C = []
    list_gamma = []
    # Adding the score of of SVM classifier considering the differents parameters 
    for (k, (C, gamma, clf)) in enumerate(classifiers):
        scores.append(clf.score(X_test,y_test))
        list_C.append(C)
        list_gamma.append(gamma)
    # Plot the 3D surface of the score considering the different values of gamma and C
    plotx,ploty, = np.meshgrid(np.linspace(np.min(list_C),np.max(list_C),10),\
                           np.linspace(np.min(list_gamma),np.max(list_gamma),10))
    plotz = interp.griddata((list_C,list_gamma),scores,(plotx,ploty),method='linear')
    
    fig = pyplot.figure()
    fig.set_size_inches(10.5, 8.5)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(plotx,ploty,plotz,cstride=1,rstride=1,cmap='viridis')  # or 'hot'
    ax.set_xlabel('C', labelpad=20, fontsize = 14)
    ax.set_ylabel('gamma',labelpad=20, fontsize = 14)
    ax.set_zlabel('scores',labelpad=20, fontsize = 14)
    ax.set_title('Validation accuracy')    
    ax.tick_params(labelsize=14)
    pyplot.tight_layout()
    pyplot.savefig('Score_Parameters.PNG')
    pyplot.show()

    plot_boundaries_decision(X_2d,y,model,"BoundariesDecision.PNG")
