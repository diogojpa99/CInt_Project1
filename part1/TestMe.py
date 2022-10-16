#!/usr/bin/python

import sys
from hamcrest import none
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, recall_score,f1_score, balanced_accuracy_score, accuracy_score
import pickle
from sklearn.metrics import confusion_matrix


"""*********************************  Initializations() ************************************"""

####### Read .csv from arguments #########

if len(sys.argv) < 2:
    print(" ----- Error: arguments should follow the following setup ----- ")
    print("(python) TestMe.py xxx.csv")
    exit(0)
elif sys.argv[1][int(len(sys.argv[1])-4):int(len(sys.argv[1]))] != '.csv':
        print(" ----- Error: arguments should follow the following setup ----- ")
        print("(python) TestMe.py xxx.csv")
else:
    df = pd.read_csv(sys.argv[1])


####### Import Max-Min values #########

x_train_max_min = np.load("X_Train_MaxMin.npy")


######## Init Variables #######
 
n_features = len(df.columns) ## Number of features
missing_data = [(-1,'Feature')] #We will not count the first position
outliers =  [(-1,'Feature')] 

sm = True #SMOTE

"""*********************************  Functions() ************************************"""

#Returns an array with a vector of the shape:(missing_data_index, Feature)
def missing_data_detector(data_column, missing_data,feature): 
    
    i = 0
    for d in data_column:
        if (d == True):
            #print('Missing df index::',i)
            missing_data.append((i,str(feature)))
        i += 1
    
    return

# Z-score for outlier detection - Change Name
def outlier_detector(df_column,feature,outliers): 
    
    df_column = np.array(df_column)
    stdev = df_column.std()
    avg = df_column.mean()
    column = [-1]
    threshold = 7.5 #4 outliers
    #threshold = 7.4 #6 outliers
    
    for d in df_column:
        column.append(abs(d-avg)/stdev)
    
    column.pop(0) #Remove first element of list
    column = np.array(column)
    
    for i in range(len(column)):
        if (column[i] > threshold*column.mean()):
            outliers.append((i,str(feature)))

    return 

# Filling data with interpolation between two points
def Filling_data(df, data):
    
    for d in data:
        df.loc[d[0],d[1]] = (df.loc[d[0]-1,d[1]] + df.loc[d[0]+1,d[1]])/2
        
    return df

#Clean noise
def Clean_Noise(df):
    
    for i in range (2,n_features-3):
        df[df.columns[i]]= df[df.columns[i]].rolling(15).mean()

    print(df.describe())

    return df 

#Normalize data with min-max of the training set
def Normalize_test_set(x_test, max_min):
    
    x_test_norm = x_test
       
    for i in range(7-2):
        x_test_norm[:,i] = (x_test[:,i] - max_min[0][i])/(max_min[1][i] - max_min[0][i])
    
    return x_test_norm

#Feature Selection
# If two features have a pearson correlation coefficient bigger than 85 remove one of them
def Feature_Selection(df):
    
    df = df.drop(['S3Temp','CO2'], axis=1)
                
    return df

#Print scores
def Print_Scores_Multiclass(Y_pred,Y_test):
    
    print('Multiclass - Labeles:\t["0"\t"1"\t\t"2"\t\t"3"]')
    print('Multiclass - Precision:', precision_score(Y_pred,Y_test, labels=[0,1,2,3], average=None))
    print('Multiclass - Recall:', recall_score(Y_pred,Y_test, labels=[0,1,2,3], average=None))
    print('Multiclass - f1-score:', f1_score(Y_pred,Y_test, labels=[0,1,2,3], average=None))
    
    print('Multiclass - Macro Precision:', precision_score(Y_pred,Y_test, labels=[0,1,2,3], average='macro'))
    print('Multiclass - Macro Recall:', recall_score(Y_pred,Y_test, labels=[0,1,2,3], average='macro'))
    print('Multiclass - Macro f1-score:', f1_score(Y_pred,Y_test, labels=[0,1,2,3], average='macro'))
    
    return

#Plotonfusion Matrix
def Plot_ConfusionMatrix_multiclass(test_y, pred_y):
    
    
    cf_matrix = confusion_matrix(test_y, pred_y)
    
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['0','1','2','3'])
    ax.yaxis.set_ticklabels(['0','1','2','3'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()
    
    return

    
"""*********************************  main() ************************************"""

"""********************* Missing data Detection **********************"""

for i in range (2,n_features):
    missing_data_detector(df.isnull()[df.columns[i]], missing_data, df.columns[i])
    
missing_data.pop(0)
#print("Missing data Index", missing_data)

"""********************* Filling Missing data **********************"""

df = Filling_data(df, missing_data)
    
"""********************* Outlier Detection **********************"""

for i in range (2,n_features):
    outlier_detector(df[df.columns[i]],df.columns[i],outliers)

outliers.pop(0)
#print('Outliers',outliers)

"""********************* Removing Outliers **********************"""

df = Filling_data(df, outliers)

"""********************* Droping 'DATA' and 'Time' Columns **********************"""

df = df.drop(['Date', 'Time'], axis=1)

"""************************* Feature Selection ******************************"""

df = Feature_Selection(df)

"""************************** X-Y Split *********************************"""

y = df['Persons']
df = df.drop(['Persons'], axis = 1)

x = df

y_real = y.to_numpy()
x_real = x.to_numpy()

"""********************* Normalize the Test Set *************************"""

x_test_norm = Normalize_test_set(x_real, x_train_max_min)

"""**************************** Upload Multiclass Model ***********************************"""

loaded_mlp = pickle.load(open('Best_Multiclass_Model.sav', 'rb'))

"""**************************** predict ***********************************"""

print("------------- Multiclass Model Output ----------------")
y_pred =loaded_mlp.predict(x_test_norm)

"""**************************** Output ***********************************"""

Print_Scores_Multiclass(y_pred,y_real)
Plot_ConfusionMatrix_multiclass(y_real, y_pred)