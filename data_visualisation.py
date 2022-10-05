from cv2 import threshold
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


''' Read the .csv file '''

df = pd.read_csv('Proj1_Dataset.csv') 
print('Features: ', df.keys())


''' Initializations '''

n_features = len(df.columns) ## Number of features
missing_data = [(-1,'Feature')] #We will not count the first position
outliers =  [(-1,'Feature')] 



''' Functions '''

# missing_df_detector():
# Returns an array with a vector og the shape:
# (missing_df_index, Feature)
def missing_data_detector(data_column, missing_data,feature): 
    
    i = 0
    for d in data_column:
        if (d == True):
            #print('Missing df index::',i)
            missing_data.append((i,str(feature)))
        i += 1
    
    return


# Plot - Observe the df 
def plot_df(df):
    
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
    return


# Observe if there is correlation between different features 
def plot_correlation(df):

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


''' main() '''

###### Missing df Detection #######

for i in range (2,n_features):
    #print('---------------------------')
    #print('Feature: ', df.columns[i])
    missing_data_detector(df.isnull()[df.columns[i]], missing_data, df.columns[i])
    
missing_data.pop(0)
print('-----------------------------------------------------------')
print("Missing data Index", missing_data)
print('-----------------------------------------------------------')

###### Filling Missing df ######

print('-----------------------------------------------------------')
print('Select the mode you want for filling missing df:')
print('-----------------------------------------------------------')
print('1. Interpolation of the previous and the next value')
print('-----------------------------------------------------------')
interpolation_flag = int(input())

if interpolation_flag == 1:   
    for m in missing_data:
        df.loc[m[0],m[1]] = (df.loc[m[0]-1,m[1]] + df.loc[m[0]+1,m[1]])/2
    


###### Outlier Detection #######

##### Z-score ######

def z_score(df_column,feature,outliers): 
    
    df_column = np.array(df_column)
    stdev = df_column.std()
    avg = df_column.mean()
    column = [-1]
    threshold = 7.5

    for d in df_column:
        column.append(abs(d-avg)/stdev)
    
    column.pop(0) #Remove first element of list
    column = np.array(column)
    
    for i in range(len(column)):
        if (column[i] > threshold*column.mean()):
            outliers.append((i,str(feature)))

    return 

for i in range (2,n_features):
    z_score(df[df.columns[i]],df.columns[i],outliers)

outliers.pop(0)
print('-----------------------------------------------------------')
print('Outliers',outliers)
print('-----------------------------------------------------------')

#plot_df(df)

if interpolation_flag == 1:  
    for o in outliers:
        df.loc[o[0],o[1]] = (df.loc[o[0]-1,o[1]] + df.loc[o[0]+1,o[1]])/2
        
#plot_df(df)
plot_correlation(df)