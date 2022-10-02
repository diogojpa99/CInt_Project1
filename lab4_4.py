import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_haberman = pd.read_csv('haberman\haberman.data')
data_haberman.columns = ['Age', 'Operation Year', 'Nodes Detected', 'Survival Status']

#print(data_haberman)

data_iris = pd.read_csv('iris\iris.data')
data_iris.columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class']

#Data Visualization
#data.describe()
print(data_iris['Class'].value_counts())
import seaborn as sns
sns.set_palette('husl')

g = sns.pairplot(data_iris, hue='Class', markers='+')
plt.show()

g = sns.violinplot(y='Class', x='Sepal Length', data=data_iris, inner='quartile')
plt.show()
g = sns.violinplot(y='Class', x='Sepal Width', data=data_iris, inner='quartile')
plt.show()
g = sns.violinplot(y='Class', x='Petal Length', data=data_iris, inner='quartile')
plt.show()
g = sns.violinplot(y='Class', x='Petal Width', data=data_iris, inner='quartile')
plt.show()

#Selecting x and y train
x = data_iris.iloc[:, :4].values
#print(x)
y = data_iris.iloc[:, 4].values
#print(y)

#Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#Scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

#from sklearn.preprocessing import MinMaxScaler
#ss = MinMaxScaler()
#X_train = ss.fit_transform(X_train)
#X_test = ss.fit_transform(X_test)



from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression (random_state=0)
#Train the model
classifier1.fit(X_train, y_train)
#Make preditions
y_pred1 = classifier1.predict(X_test)



from sklearn.naive_bayes import GaussianNB
# GAUSSIAN NAIVE BAYES
classifier2 = GaussianNB()
# train the model
classifier2.fit(X_train, y_train)
# make predictions
y_pred2 = classifier2.predict(X_test)
 
# DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
classifier3 = DecisionTreeClassifier(random_state=0)
# train the model
classifier3.fit(X_train, y_train)
# make predictions
y_pred3 = classifier3.predict(X_test)



from sklearn import svm # SUPPORT VECTOR MACHINE
classifier4 = svm.SVC(kernel='linear')  # Linear Kernel
# train the model
classifier4.fit(X_train, y_train)
# make predictions
y_pred4 = classifier4.predict(X_test)

classifier5 = svm.SVC(kernel='poly', degree=1) # Polynominal kernel
classifier5.fit(X_train, y_train)
y_pred5 = classifier5.predict(X_test)

classifier6 = svm.SVC(kernel='rbf') # Gaussian kernel gaussian (Outro exempol é: 'sigmoid': melhor para casos binarios)
classifier6.fit(X_train, y_train)
y_pred6 = classifier6.predict(X_test)

#Evaluation metrics
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)
cm2 = confusion_matrix(y_test, y_pred2)
cm3 = confusion_matrix(y_test, y_pred3)
cm4 = confusion_matrix(y_test, y_pred4)
cm5 = confusion_matrix(y_test, y_pred5)
cm6 = confusion_matrix(y_test, y_pred6)


print ("Confusion Matrix : \n", cm1)
print ("Confusion Matrix2 : \n", cm2)
print ("Confusion Matrix3 : \n", cm3)
print ("Confusion Matrix4 : \n", cm4)
print ("Confusion Matrix5 : \n", cm5)
print ("Confusion Matrix6 : \n", cm6)

from sklearn.metrics import accuracy_score
print("Accuracy Logistic: ", accuracy_score(y_test, y_pred1))
print("Accuracy Naive Bayes: ", accuracy_score(y_test, y_pred2))
print("Accuracy of Decision Tree Classifier: ", accuracy_score(y_test, y_pred3))
print("Accuracy of linear SVM: ", accuracy_score(y_test, y_pred4))
print("Accuracy of Poly SVM: ", accuracy_score(y_test, y_pred5))
print("Accuracy of Gaussaim SVM: ", accuracy_score(y_test, y_pred6))

from sklearn.metrics import precision_score
print("Precision: ", precision_score(y_test, y_pred1, average='macro'))
print("Precision2: ", precision_score(y_test, y_pred2, average='macro'))
print("Precision3: ", precision_score(y_test, y_pred3, average='macro'))
print("Precision4: ", precision_score(y_test, y_pred4, average='macro'))
print("Precision5: ", precision_score(y_test, y_pred5, average='macro'))
print("Precision6: ", precision_score(y_test, y_pred6, average='macro'))


# KNN by looking at the best k
from sklearn.neighbors import KNeighborsClassifier
k_range = list(range(1,26))
scores = []
for k in k_range:
    classifier7 = KNeighborsClassifier(n_neighbors=k)
    classifier7.fit(X_train, y_train)
    y_pred7 = classifier7.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred7))

plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.show()

# Take the better accuracy from scores
idx_knn = np.argmax(scores)
print('KNN with hight accuracy: ', scores[idx_knn])



# SVM's with grid search
from sklearn.model_selection import GridSearchCV
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}

grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose=3)

#fitting the model for grid search
grid.fit(X_train, y_train)

# print best parameter after tuning
print('Best parameters: ', grid.best_params_)
  
# print how our model looks after hyper-parameter tuning
print('Confirm that our model choose the best parameters: ', grid.best_estimator_)

grid_predictions = grid.predict(X_test)
print("Accuracy of SVM with grid ", accuracy_score(y_test, grid_predictions))

# A conclusao a que cheguei com o grid search é que os parametros que aí estao nao sao os melhores kkkkkk
# Sem a grid serach e com o C de default, todas as SVM que fiz antes deram melhor accuracy
