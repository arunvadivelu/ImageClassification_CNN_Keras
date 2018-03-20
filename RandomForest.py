import os
import numpy as np
import cPickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

class RandomForestTrial():
    X = []
    Y = []
    n_files = 2
    f_path="/home/yvaska/Downloads/Python_Images_Data/cifar-10-batches-py/"
    input_file1= "data_batch_1"
    input_file2= "data_batch_2"
    PCA_model_name= "finalized_PCAmodel"
    i_file1 = os.path.join(f_path, input_file1)
    i_file2 = os.path.join(f_path, input_file2)
    myfile = [i_file1,i_file2]
    PCA_fname= os.path.join(f_path, PCA_model_name)

# use of fiel import and data structure
    def unpickle(file):  
        with open(file, 'rb') as fo:
            dict = cPickle.load(fo)
            return dict
            
    def showImg(X,n): 
        XR = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
        fig, axes1 = plt.subplots(n, n, figsize=(8, 8))
        for j in range(n):
            for k in range(n):
                i = np.random.choice(range(len(XR)))
                axes1[j][k].set_axis_off()
                axes1[j][k].imshow(XR[i:i + 1][0])  
                
    def build_PCA(X,n,fname):
        my_model = PCA(n_components=n)
        mypca = my_model.fit_transform(X)
        print ("sum of the variance explained:",my_model.explained_variance_ratio_.sum())
        cPickle.dump(my_model, open(fname, 'wb'))
        return mypca
    try: 
   
        for i in range(n_files):
            d = unpickle(myfile[i])
            X = d["data"]
            Y = d['labels']

        X = np.array(X)
        Y = np.array(Y)
        
        if np.amax(X)==0:
            print("Values are Zero. please check input file")
            raise ValueError
        
        # Show Image
        showImg(X,2)
        

        # conver to Gray scale
        X_Gray = 0.21*X[:,0:1024] + 0.72*X[:,1024:2048] + 0.07*X[:,2048:3072]

        pca_x= build_PCA(X_Gray,20,PCA_fname)
  
        # trining and testing 
        training, test = pca_x[:8000,:], pca_x[8000:,:]
        tr_label, tst_label = Y[:8000,], Y[8000:,]
        
        # Randomforest model and predict the results on Test data       
        clf = RandomForestClassifier(n_estimators=10)
        clf = clf.fit(pca_x, Y)
        y_predicted =clf.predict(test)
        
        #Analyze the results
        myTuple = zip(tst_label,y_predicted)
        notsame=list(filter(lambda xy: xy[0] !=xy[1], myTuple))
        print ("Predicted not same as the True Values", notsame)
        accuracy = (1-len(notsame)/len(tst_label))*100
        #print accuracy
        
        #Check number of class lables in the test data
        unique, counts = np.unique(tst_label, return_counts=True)
        print dict(zip(unique, counts))
        
        p_unique, p_counts = np.unique(y_predicted, return_counts=True)
        print dict(zip(p_unique, p_counts))
        
        print confusion_matrix(tst_label, y_predicted)

    except Exception as e:
        print("Error in Model" % str(e))