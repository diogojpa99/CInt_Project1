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
from scipy.stats import pearsonr
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix




"""*********************************  Initializations() ************************************"""

df = pd.read_csv('Proj1_Dataset.csv') 
print('Features: ', df.keys())

sns.set_theme(style="darkgrid")

n_features = len(df.columns) ## Number of features
missing_data = [(-1,'Feature')] #We will not count the first position
outliers =  [(-1,'Feature')] 

sm = True #SMOTE

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
# If two features have a pearson correlation coefficient bigger than 85 remove one of them
def Feature_Selection(df):
    
    selected_features = ['Features']
    
    for f in df:
        for d in df:
            corr, _ = pearsonr(df[f],df[d])
            if ( (corr > 0.85) and (d != f) and (d != df.columns[0])): #Do not remove first feature
                if d not in selected_features:
                    selected_features.append(d)
                
    selected_features.pop(0)
    
    for s in selected_features:
        df.pop(s)
        
    #print('Data after feature selection:', df.columns)
                
    return df

#Print scores
def Print_Scores_binary(Y_pred,Y_test):
    
    print('binary - Accuracy:', accuracy_score(Y_pred,Y_test))
    print('binary - Balanced Accuracy:', balanced_accuracy_score(Y_pred,Y_test))
    
    print('binary - Precision:', precision_score(Y_pred,Y_test, average='binary'        ))
    print('binary - Recall:', recall_score(Y_pred,Y_test, average='binary'))
    print('binary - f1-score:', f1_score(Y_pred,Y_test, average='binary'))
    
    return

#Model Hyper-parameter tunning
#Without feature selection
#Multiclass
#Everything changes if it is binary + The number of features are not the same
def model_FineTuning(X_train, Y_train, sm ):

    alphas = [0.0005,  0.0001, 0.000175]
    l_rates =[0.01, 0.0175, 0.02]
              
    if sm == True:
        pipeline = imbpipeline(steps = [['smote', SMOTE(random_state=11)],
                                        ['norm', MinMaxScaler()],
                                        ['MLP', MLPClassifier()]])
    else:
        pipeline = Pipeline(steps = [['norm', MinMaxScaler()],
                                     ['MLP', MLPClassifier()]])
        
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
         
    param_grid = {'MLP__hidden_layer_sizes':[(12,),(10,)], 
                  'MLP__activation':['tanh','relu'], 
                  'MLP__solver':['adam'],
                  'MLP__alpha':alphas, 
                  'MLP__learning_rate':['constant','adaptive'],
                  'MLP__max_iter':[500],
                  'MLP__learning_rate_init':l_rates}
    
    #scoring = {'accuracy':met.make_scorer(met.accuracy_score)}
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='roc_auc',
                               cv=stratified_kfold)
    
    grid_search.fit(X_train, Y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    return

#Simple NLP use for testing
def simple_binary_model(X_train, Y_train, X_test, Y_test):
    
    mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic',
                        solver = 'sgd', alpha = 0.0001, learning_rate='constant', 
                        max_iter = 400, learning_rate_init = 0.01).fit(X_train,Y_train)
    
    Y_pred = mlp.predict(X_test)
    Print_Scores_binary(Y_pred,Y_test)
    
    return

# Best multiclass model if number of features = 9 
# According to model()
def best_binary_model(X_train, Y_train, X_test, Y_test):
    
    mlp = MLPClassifier(hidden_layer_sizes=(12,), activation='tanh',
                        solver = 'adam', alpha = 0.0001, learning_rate='adaptive', 
                        max_iter = 400, learning_rate_init = 0.01).fit(X_train,Y_train)
    
    Y_pred = mlp.predict(X_test)
    
    Plot_LossCurve(mlp)
    Print_Scores_binary(Y_pred,Y_test)
    Plot_ConfusionMatrix(mlp, X_test, Y_test, Y_pred)
    Plot_ConfusionMatrix_multiclass(Y_test, Y_pred)
    
    return

#Other binary model
def binary_model_3(X_train, Y_train, X_test, Y_test):
    
    mlp = MLPClassifier(hidden_layer_sizes=(10,10), activation='tanh',
                        solver = 'adam', alpha = 0.00015, learning_rate='adaptive', 
                        max_iter = 400, learning_rate_init = 0.01).fit(X_train,Y_train)
    
    Y_pred = mlp.predict(X_test)
    
    Plot_LossCurve(mlp)
    Print_Scores_binary(Y_pred,Y_test)
    Plot_ConfusionMatrix(mlp, X_test, Y_test, Y_pred)
    
    return
# binary model number 4
def binary_model_4(X_train, Y_train, X_test, Y_test):
    
    mlp = MLPClassifier(hidden_layer_sizes=(12,), activation='tanh',
                        solver = 'adam', alpha = 0.000175, learning_rate='adaptive', 
                        max_iter = 400, learning_rate_init = 0.0175).fit(X_train,Y_train)
    
    Y_pred = mlp.predict(X_test)
    
    Plot_LossCurve(mlp)
    Print_Scores_binary(Y_pred,Y_test)
    Plot_ConfusionMatrix(mlp, X_test, Y_test, Y_pred)

    return

# binary model number 5
def binary_model_5(X_train, Y_train, X_test, Y_test):
    
    mlp = MLPClassifier(hidden_layer_sizes=(12,), activation='tanh',
                        solver = 'adam', alpha = 0.000175, learning_rate='adaptive', 
                        max_iter = 400, learning_rate_init = 0.024).fit(X_train,Y_train)
    
    Y_pred = mlp.predict(X_test)
    
    Plot_LossCurve(mlp)
    Print_Scores_binary(Y_pred,Y_test)
    Plot_ConfusionMatrix(mlp, X_test, Y_test, Y_pred)

    return

#Plotonfusion Matrix
def Plot_ConfusionMatrix_multiclass(test_y, pred_y):
    
    
    cf_matrix = confusion_matrix(test_y, pred_y)
    
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['0','1'])
    ax.yaxis.set_ticklabels(['0','1'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()
    
    return

#Plot Confusion Matrix
def Plot_ConfusionMatrix(mlp, test_x, test_y, pred_y):
    
    fig = plot_confusion_matrix(mlp, test_x, test_y, display_labels=mlp.classes_)
    fig.figure_.suptitle("Confusion Matrix")
    plt.show()
    
    print(classification_report(test_y, pred_y))
    
    return

#Plot Loss Curve
def Plot_LossCurve(mlp):
    
    plt.plot(mlp.loss_curve_)
    plt.title("Loss Curve", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
    plt.close()
    
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

"""*********************  Vizualize Data  *************************"""

#plot_correlation(df)

"""********************* Droping 'DATA' and 'Time' Columns **********************"""

df = df.drop(['Date', 'Time'], axis=1)

"""************************* Transforming to Binary Variables ******************************"""

df['Persons'].replace({1:0, 2:0 ,3:1}, inplace=True)

"""************************* Feature Selection ******************************"""

df = Feature_Selection(df)

"""************************** Removing Test Set *********************************"""

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,:(len(df.columns)-1)].values, 
                                                    df.iloc[:,(len(df.columns)-1)].values,
                                                    test_size=0.1,shuffle=True)

"""*********** [Training data] Dealing with noise - Moving Average ***************"""

#x_train = Clean_Noise(x_train)

"""************************* Model Fine-Tunning ******************************"""

#model_FineTuning(x_train, y_train, True) #To do Grid_Seacrh CV with SMOTE

"""********************* Normalize the Training Set *************************"""

x_train_norm = Normalize_data(x_train, x_train)

"""********************* Normalize the Test Set *************************"""

x_test_norm = Normalize_data(x_test, x_train)

"""********************* Balance the Training Set ***************************"""
if sm == True:
    x_train_norm,y_train = Balance_data(x_train_norm, y_train)

"""**************************** Model Training ***********************************"""

"""**************************** Model Predicting ***********************************"""

#Simple model to test Feature Selection, Balance techniques, etc.
'''print("--------- Simple_binary_model ----------")
simple_binary_model(x_train_norm, y_train, x_test_norm, y_test)'''

print("--------- best_binary_model ----------")
best_binary_model(x_train_norm, y_train, x_test_norm, y_test)

'''print("--------- binary_model_3 ----------")
binary_model_3(x_train_norm, y_train, x_test_norm, y_test)
print("--------- binary_model_4 ----------")
binary_model_4(x_train_norm, y_train, x_test_norm, y_test)
print("--------- binary_model_5 ----------")
binary_model_5(x_train_norm, y_train, x_test_norm, y_test)'''