import pandas as pd
import matplotlib.pyplot as plt

''' Read the .csv file '''

data = pd.read_csv('Proj1_Dataset.csv') 
print(data.keys())


''' Plot - Observe the data '''

for i in range(2, len(data.columns)):
    plt.plot(data[data.columns[i]])
    plt.title(data.columns[i])
    plt.show()
    print(i,'Data:',data.columns[i])
    i +=1
    

''' Histogram - Observe the data '''

for i in range(2, len(data.columns)):
    plt.hist(data[data.columns[i]]) # Histograms give more information about outliers
    plt.title(data.columns[i])
    plt.show()
    i +=1
    

''' Observe if there is correlation between different features '''

print('Please insert the index of the first feature:')
feature_1 = int(input())
print('Please insert the index of the second feature:')
feature_2 = int(input())

plt.scatter(data[data.columns[feature_1]], data[data.columns[feature_2]])
plt.title("Correlation between two variables")
plt.show()

# What will the behaviour be if we remove outliers and handle missing data ?

