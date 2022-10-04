import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


''' Read the .csv file '''

data = pd.read_csv('Proj1_Dataset.csv') 
print(data.keys())


''' Initializations '''

n_features = len(data.columns) ## Number of features
missing_data = [(-1,'Feature')] #We will not count the first position


''' Functions '''

# missing_data_detector():
# Returns an array with a vector og the shape:
# (missing_data_index, Feature)
def missig_data_detector(data_column, missing_data,feature): 
    
    i = 0
    for d in data_column:
        if (d == True):
            print('Missing Data index::',i)
            missing_data.append((i,str(feature)))
        i += 1
    
    return


# Plot - Observe the data 
def plot_data(data):
        
    for i in range(2, len(data.columns)):
        plt.plot(data[data.columns[i]])
        plt.title(data.columns[i])
        plt.show()
        print(i,'Data:',data.columns[i])
        i +=1
        
    return


# Histogram - Observe the data 
def plot_histogram(data):
    
    for i in range(2, len(data.columns)):
        plt.hist(data[data.columns[i]]) # Histograms give more information about outliers
        plt.title(data.columns[i])
        plt.show()
        i +=1
        
    return


# Observe if there is correlation between different features 
def plot_correlation(data):

    print('Please insert the index of the first feature:')
    feature_1 = int(input())
    print('Please insert the index of the second feature:')
    feature_2 = int(input())

    plt.scatter(data[data.columns[feature_1]], data[data.columns[feature_2]])
    plt.title("Correlation between two variables")
    plt.show()
    
    return


''' main() '''

###### Missing Data Detection #######

for i in range (2,n_features):
    print('---------------------------')
    print('Feature: ', data.columns[i])
    missig_data_detector(data.isnull()[data.columns[i]], missing_data, data.columns[i])
    
print("Missing Data Index", missing_data)


###### Filling Missing Data ######

print('Select the mode you want for filling missing data:')
print('Interpolation of the previous and the next one: 1')
missing_data_flag = int(input())

if missing_data_flag == 1:   
    for m in missing_data:
        if m[0] != -1:
            data[m[1]][m[0]] = (data[m[1]][m[0]-1]+ data[m[1]][m[0]+1])/2
    


###### Outlier Detection #######

outliers = [-1] #We will not count the first position

def outlier_detector_stdev(feature, stdev, outliers): 
   
    stdev = feature.std()
    avg = feature.mean()
    k = 4
    i = 0
    
    for f in feature:
        if (f > avg + k*stdev) or (f < avg - k*stdev):
            print(i, 'Outlier')
            outliers.append(i)
        i += 1
        
    return

''' Z-Score '''

def z_score(feature): 
   
    stdev = feature.std()
    avg = feature.mean()
    column = [-1]

    for f in feature:
        column.append(abs(f-avg)/stdev)
    
    column.pop(0) #remove first element of list
    column = np.array(column)
    
    print(column)
    print(column.mean())
    print(column.max())
    
    return column

import seaborn as sns

st_column = z_score(data[data.columns[2]])

exit(0)

    
data_values = data.drop(['Date','Time'], axis = 1)
print(data_values.apply(stats.zscore))
print(type(data_values.apply(stats.zscore)))


for i in range(2,n_features):
    sns.histplot(data.apply(stats.zscore))
    plt.show()
    
"""for i in range (2,n_features):
    #outlier_detector_stdev(data[data.columns[i]],stdev, outliers)
    outlier_detector_quantiles(data[data.columns[i]], quantiles, outliers)"""

print('Insert Feature:')
outlier_flag = int(input())
outlier_detector_stdev(data[data.columns[outlier_flag]],stdev, outliers)
print(outliers)

exit(0)




print("New")
print(data[missing_data[1][1]][missing_data[1][0]-1])   
print(data[missing_data[1][1]][missing_data[1][0]])    
print(data[missing_data[1][1]][missing_data[1][0]+1])        

f= 0
for i in data['S1Temp']:
    if i <15:
        print(f,i) 
    f +=1 
          
plt.plot(data['S1Temp'])
plt.title('S1Temp')
plt.show()
                
plt.hist(data['S1Temp']) # Histograms give more information about outliers
plt.title('S1Temp')
plt.show()
