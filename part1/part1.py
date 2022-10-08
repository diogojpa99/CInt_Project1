import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import MinMaxScaler



"""*********************************  Initializations() ************************************"""

df = pd.read_csv('Proj1_Dataset.csv') 
print('Features: ', df.keys())

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
    
    fig, ax = plt.subplots()

    scatter = ax.scatter(df[df.columns[feature_1]], df[df.columns[feature_2]], c = df['Persons'], s = 10)

    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper left", title="Persons")
    ax.add_artist(legend1)
    ax.set_xlabel(df.columns[feature_1])
    ax.set_ylabel(df.columns[feature_2])
    fig.suptitle("Correlation between two Features")
    plt.show()
  
    return

#Important plots for the report
def plot_information(df):
    
    sns.set_theme(style="ticks")
    sns.pairplot(df, hue="Persons")
    plt.show()
    
    print(df.columns)
    print("Please input the idex of the feature that you want:")
    feature = int(input())
    
    sns.set_theme(style="darkgrid")
    sns.displot(df, x="Persons", col=df[df.columns[feature]],
                    binwidth=3, height=3, facet_kws=dict(margin_titles=True))
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
    
    sm = SMOTE()
    return sm.fit_resample(x_train,y_train)

#Feature Selection

# Model Hyper-parameter tunning

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

"""********************* Droping 'DATA' and 'Time' Columns **********************"""

df = df.drop(['Date', 'Time'], axis=1)

"""************************** Removing Test Set *********************************"""

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,:(len(df.columns)-1)].values, 
                                                    df.iloc[:,(len(df.columns)-1)].values,test_size=0.1,shuffle=True)

"""*********** [Training data] Dealing with noise - Moving Average ***************"""

# train_set = Clean_Noise(train_set)

"""********************* Normalize the Training Set *************************"""

x_train = Normalize_data(x_train, x_train)

"""********************* Normalize the Test Set *************************"""

x_test = Normalize_data(x_test, x_train)

"""********************* Balance the Training Set ***************************"""

x_train,y_train = Balance_data(x_train, y_train)

plt.hist(y_train)
plt.show()

"""************************* Feature Selection ******************************"""

"""************************* Model Fine-Tunning ******************************"""

"""**************************** Model Training ***********************************"""

"""**************************** Model Predicting ***********************************"""

