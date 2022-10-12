import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as met
from sklearn.metrics import precision_score, recall_score,f1_score, balanced_accuracy_score, accuracy_score



"""*********************************  Initializations() ************************************"""

df = pd.read_csv('Proj1_Dataset.csv') 
print('Features: ', df.keys())

sns.set_theme(style="darkgrid")

n_features = len(df.columns) ## Number of features
missing_data = [(-1,'Feature')] #We will not count the first position
outliers =  [(-1,'Feature')] 


"""*********************************  Functions() ************************************"""

# Plot - Observe the data
def plot_data(df):
        
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2)
        
    for i in range(2, len(df.columns)):
        
        fig.suptitle(df.columns[i])
        ax1.plot(df[df.columns[i]])
        ax2.hist(df[df.columns[i]]) # Histograms give more information about outliers  
        ax2.set_title('Histogram')      
        plt.show()
        plt.pause(4)
        ax1.cla()
        ax2.cla()
    
    fig.clf()
    plt.close()
    return

# Observe if there is correlation between different features 
def plot_correlation(df):
    
    #Using Pearson Correlation
    plt.figure(figsize=(n_features,n_features))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()
    plt.close()

    print('Please insert the index of the first feature:')
    feature_1 = int(input())
    print('Please insert the index of the second feature:')
    feature_2 = int(input())
    
    sns.relplot(data = df, x = df.columns[feature_1],y = df.columns[feature_2],hue="Persons")
    plt.title("Correlation between two Features")
    plt.show()

  
    return

#Important plots for the report
def plot_information(df):
    
    sns.set_theme(style="ticks")
    sns.pairplot(df, hue="Persons")
    plt.show()

    return

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
def z_score(df_column,feature,outliers): 
    
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
    plot_correlation(df)
    plot_data(df)
    
    return df 

#Normalize data with min-max of the training set
def Normalize_data(data, train_set):
    
    scaler = MinMaxScaler()
    scaler.fit(train_set) #Normalize to the min-max of the training set

    return scaler.transform(data)

#Balance Data with SMOTE method
def Balance_data(x_train, y_train):
    
    sm = SMOTE(random_state = 11)
    return sm.fit_resample(x_train,y_train)

#Feature Selection

#Model Hyper-parameter tunning
#Without feature selection
#Multiclass
#Everything changes if it is binary + The number of features are not the same
def model(X_train, Y_train, sm ):
    
    alpha = 0.0001
    alphas = [alpha]
    learn_rate = 0.01
    l_rates =[learn_rate]
    
    for i in range(2):
        alpha = alpha/0.1
        learn_rate = learn_rate/0.35
        alphas.append(alpha)
        l_rates.append(learn_rate)
        
    """print()        
    print('alphas',alphas)
    print('l_rates',l_rates)"""
    
    if sm == True:
        pipeline = imbpipeline(steps = [['smote', SMOTE()],
                                        ['norm', MinMaxScaler()],
                                        ['MLP', MLPClassifier()]])
    else:
        pipeline = Pipeline(steps = [['norm', MinMaxScaler()],
                                     ['MLP', MLPClassifier()]])
        
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True)
         
    param_grid = {'MLP__hidden_layer_sizes':[(12,), (12,12), (10,10),(14,)], 
                  'MLP__activation':['logistic', 'tanh', 'relu'], 
                  'MLP__solver':['sgd', 'adam'],
                  'MLP__alpha':alphas, 
                  'MLP__learning_rate':['constant','adaptive'],
                  'MLP__max_iter':[600],
                  'MLP__learning_rate_init':l_rates,
                  'MLP__n_iter_no_change':[10,18,27]}
    
    scoring = {'accuracy':met.make_scorer(met.accuracy_score)}
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scoring, refit="accuracy",
                               cv=stratified_kfold, n_jobs=-1)
    
    grid_search.fit(X_train, Y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    return

def simple_multiclass_model(X_train, Y_train, X_test, Y_test, sm):
    
    if sm == True:
        X_train,Y_train = Balance_data(X_train, Y_train)
    
    X_train = Normalize_data(X_train, X_train)
    
    mlp = MLPClassifier(hidden_layer_sizes=(12,12), activation='logistic',
                        solver = 'sgd', alpha = 0.05, batch_size=32, learning_rate='adaptive', 
                        max_iter = 400, learning_rate_init = 0.03, n_iter_no_change = 18).fit(X_train,Y_train)
    
    Y_pred = mlp.predict(X_test)
    print('Multiclass - Accuracy:', accuracy_score(Y_pred,Y_test))
    print('Multiclass - Balanced Accuracy:', balanced_accuracy_score(Y_pred,Y_test))
    
    print('Multiclass - micro Precision:', precision_score(Y_pred,Y_test, labels=[0,1,2,3], average='micro'))
    print('Multiclass - micro Recall:', recall_score(Y_pred,Y_test, labels=[0,1,2,3], average='micro'))
    print('Multiclass - micro f1-score:', f1_score(Y_pred,Y_test, labels=[0,1,2,3], average='micro'))
    
    print('Multiclass - macro Precision:', precision_score(Y_pred,Y_test, labels=[0,1,2,3], average='macro'))
    print('Multiclass - macro Recall:', recall_score(Y_pred,Y_test, labels=[0,1,2,3], average='macro'))
    print('Multiclass - macro f1-score:', f1_score(Y_pred,Y_test, labels=[0,1,2,3], average='macro'))
    
    return

"""*********************************  main() ************************************"""

"""********************* Missing data Detection **********************"""

for i in range (2,n_features):
    missing_data_detector(df.isnull()[df.columns[i]], missing_data, df.columns[i])
    
missing_data.pop(0)
print("Missing data Index", missing_data)


"""********************* Filling Missing data **********************"""

df = Filling_data(df, missing_data)
    
"""********************* Outlier Detection **********************"""

for i in range (2,n_features):
    z_score(df[df.columns[i]],df.columns[i],outliers)

outliers.pop(0)
print('Outliers',outliers)

"""********************* Removing Outliers **********************"""

df = Filling_data(df, outliers)

"""*********************  Vizualize Data  *************************"""

#plot_correlation(df)

"""********************* Droping 'DATA' and 'Time' Columns **********************"""

df = df.drop(['Date', 'Time'], axis=1)

"""************************** Removing Test Set *********************************"""

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,:(len(df.columns)-1)].values, 
                                                    df.iloc[:,(len(df.columns)-1)].values,
                                                    test_size=0.1,shuffle=True)

"""*********** [Training data] Dealing with noise - Moving Average ***************"""

# train_set = Clean_Noise(train_set)

"""********************* Normalize the Training Set *************************"""

#x_train = Normalize_data(x_train, x_train)

"""********************* Normalize the Test Set *************************"""

x_test = Normalize_data(x_test, x_train)

"""********************* Balance the Training Set ***************************"""

#x_train,y_train = Balance_data(x_train, y_train)

"""************************* Feature Selection ******************************"""

"""************************* Model Fine-Tunning ******************************"""

#Todo Grid_Seacrh CV with SMOTE
model(x_train, y_train, True)

"""**************************** Model Training ***********************************"""

"""**************************** Model Predicting ***********************************"""

#Simple model to test Feature Selection, Balance techniques, etc.
simple_multiclass_model(x_train, y_train, x_test, y_test, sm = True)
