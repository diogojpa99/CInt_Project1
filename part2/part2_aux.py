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
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


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
# New Feature: time[dia noite]
#def fuzzy_time()

df = pd.DataFrame(df)
df_fuzzy = df
#Condiçao para dia = 1, noite = -1
for k in range (0,len(df['Time'])):#Criar uma column e depois dar insert -> ja nao deve tar wearning
    if ((df['Time'][k] <= '20:00:00') & (df['Time'][k] >= '08:00:00')):
        
        df['Time'][k] = 1
    else:
        df['Time'][k] = -1

'''
#Condicoes para de noite
Noite1 = df[((df['Time'] >= '20:00:00') & (df['Time'] <= '23:59:59'))] #& ((df['Time'] >= '00:00:00') & (df['Time'] < '08:00:00'))]
#print('Noite1:', Noite1)
Noite2 = df[(df['Time'] >= '00:00:00') & (df['Time'] < '08:00:00')]
#print('Noite2:', Noite2)

Noite = pd.concat([Noite1, Noite2])
print('Noite:', Noite)
'''
# New Feature: mean(S1Light,S2Light,S3Light)
MeanLight = np.mean(pd.concat([df['S1Light'], df['S2Light'], df['S3Light']], axis=1), axis=1)
df.insert(5, 'MeanLight', MeanLight, True)#df['PIR_max'] = PIR_max

# New Feature: PIR_maxx = max(PIR1, PIR2)
PIR_max = np.maximum(df['PIR1'], df['PIR2'])
df.insert(11, 'PIR_max', PIR_max, True)#df['PIR_max'] = PIR_max

# New Feature: S1Temp, S2Temp, S3Temp)
MeanTemp= np.mean(pd.concat([df['S1Temp'], df['S2Temp'], df['S3Temp']], axis=1), axis=1)
df.insert(2, 'MeanTemp', MeanTemp, True)#df['PIR_max'] = PIR_max

# New Feature: CO2
co2_eval = np.zeros((len(df['CO2']), 1))
ac = np.zeros((len(df['CO2'])-1, 1))

co2_der = np.ones((len(df['CO2']),1))

a1 = np.transpose(np.diff([df['CO2']], n=1, axis=1))# DIFF

ac = np.zeros((len(df['CO2']), 1))

#Esta a crescer, constante ou a decrescer
num_idx = 0
for num in a1:
    if num > 0:
        ac[num_idx] = 1 # Crescer
        #pos_count += 1
    elif num == 0:
        ac[num_idx] = 0 # Mantenho
    else:
        ac[num_idx] = -1 # Decrescer
        #neg_count += 1
    num_idx += 1


# Isto é o diff: Contabilizamos a varição
for b in range(0, len(df['CO2'])-1):
    if b <= len(df['CO2'])-2:
        co2_der[b] = a1[b]
    else:
        co2_der[b-1] = co2_der[b-2]


#co2_der = a1
#print(np.shape(co2_der))
#co2_der[len(df['CO2'])-1] = co2_der[len(df['CO2'])-2]
#print(np.shape(co2_der))
#print(np.shape(df['CO2']))

df.insert(7, 'CO2_eval', (co2_der), True)#df['PIR_max'] = PIR_max #ac!!!!!!!!!!!


'''#Problema binario de mais de 2 pessoas ou menos
for p in range (0,len(df['Persons'])):
    if df['Persons'][p] <= 2:
        #df.loc[:,'Time'] = 1
        df['Persons'][p] = 0
    else:
        df['Persons'][p] = 1
'''

"""********************* Droping 'DATA' and 'Time' Columns **********************"""
print('Passei1')
df['Time'] = pd.to_numeric(pd.to_datetime(df['Time']))

print(df['Time'])
#df = df.drop(['Date', 'Time'], axis=1)
df = df.drop(['Date', 'PIR1', 'PIR2', 'S1Light', 'S2Light', 'S3Light', 'S1Temp', 'S2Temp', 'S3Temp', 'PIR_max', 'MeanTemp'], axis=1)
#print(df)
df.to_csv('CheckTime.csv')
"""*********************  Vizualize Data  *************************"""
print(df.columns)
print('Passei 2')

#plot_correlation(df)


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
#model(x_train, y_train, True)

"""**************************** Model Training ***********************************"""

"""**************************** Model Predicting ***********************************"""

#Simple model to test Feature Selection, Balance techniques, etc.
#simple_multiclass_model(x_train, y_train, x_test, y_test, sm = True)

"""**************************** Fuzzy  ***********************************"""

df_fuzzy = df_fuzzy.drop(['Date', 'PIR1', 'PIR2', 'S1Light', 'S2Light', 'S3Light', 'S1Temp', 'S2Temp', 'S3Temp', 'PIR_max', 'MeanTemp', 'CO2'], axis=1)
#plot_correlation(df_fuzzy)

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

#Inputs
time_in_day = ctrl.Antecedent(np.arange(0, 24+1, 1), 'Time')#Ver aqui
mean_lights = ctrl.Antecedent(np.arange(min(df['MeanLight']), max(df['MeanLight'])+1, 1), 'Lights')
C02_growth = ctrl.Antecedent(np.arange(min(df['CO2_eval']), max(df['CO2_eval'])+1, 1), 'CO2')

"""print('minimo de luz::', min(df['MeanLight']))
print('maximo de luz:', max(df['MeanLight']))
print('minimo de CO2::', min(df['CO2']))
print('maximo de CO2:', max(df['CO2']))
print('minimo de CO2_eval::', min(df['CO2_eval']))
print('maximo de CO2_eval:', max(df['CO2_eval]))"""


#Outputs
Persons = ctrl.Consequent(np.arange(0, 3+1, 1), 'Persons')#Ver aqui


#time_in_day['day'] = fuzz.trimf(time_in_day.universe,[0, 5, 12])#8 da manha
#time_in_day['night'] = fuzz.trimf(time_in_day.universe,[11, 16, 24])#10/11
#'''
mean_lights['low'] = fuzz.trimf(mean_lights.universe,[min(df['MeanLight']), min(df['MeanLight']), 125])
mean_lights['medium'] = fuzz.trimf(mean_lights.universe,[104, 162, 200])
mean_lights['high'] = fuzz.trimf(mean_lights.universe,[180, max(df['MeanLight']), max(df['MeanLight'])])#,,1270

#C02_growth['decrease'] = fuzz.trimf(C02_growth.universe, [min(df['CO2_eval']), min(df['CO2_eval']), -5]) #345,,
#C02_growth['unchanged'] = fuzz.trimf(C02_growth.universe, [-10, 0, 10])
#C02_growth['increase'] = fuzz.trimf(C02_growth.universe, [5, max(df['CO2_eval']), max(df['CO2_eval'])])#,,1270
#'''
#time_in_day.view()
#mean_lights.view()
#C02_growth.view()
#plt.show()

#C02_growth['decrese'] = fuzz.trimf(C02_growth.universe, [-1, -1, 0]) #345,,
#C02_growth['mantain'] = fuzz.trimf(C02_growth.universe, [0, 0.5, 1])
#C02_growth['high'] = fuzz.trimf(C02_growth.universe, [0, 1, 1])#,,1270



Persons['LowerThanThree'] = fuzz.trapmf(Persons.universe, [0,0, 2,3]) #ou 0,0,2?
Persons['EqualToThree'] = fuzz.trimf(Persons.universe, [2, 3, 3])

#Persons.view()
#plt.show()


#core_temp.view()
#clock_speed.view()
#fan_speed.view()
#plt.show()

'''
rule1 = ctrl.Rule(time_in_day['day'] & mean_lights['low'] & C02_growth['decrease'], Persons['LowerThanThree'])
rule2 = ctrl.Rule(time_in_day['day'] & mean_lights['low'] & C02_growth['unchanged'], Persons['EqualToThree'])
rule3 = ctrl.Rule(time_in_day['day'] & mean_lights['low'] & C02_growth['increase'], Persons['EqualToThree'])

rule4 = ctrl.Rule(time_in_day['day'] & mean_lights['medium'] & C02_growth['decrease'], Persons['EqualToThree'])
rule5 = ctrl.Rule(time_in_day['day'] & mean_lights['medium'] & C02_growth['unchanged'], Persons['EqualToThree'])
rule6 = ctrl.Rule(time_in_day['day'] & mean_lights['medium'] & C02_growth['increase'], Persons['EqualToThree'])

rule7 = ctrl.Rule(time_in_day['day'] & mean_lights['high'] & C02_growth['decrease'], Persons['EqualToThree'])
rule8 = ctrl.Rule(time_in_day['day'] & mean_lights['high'] & C02_growth['unchanged'], Persons['EqualToThree'])
rule9 = ctrl.Rule(time_in_day['day'] & mean_lights['high'] & C02_growth['increase'], Persons['EqualToThree'])

rule10 = ctrl.Rule(time_in_day['night'] & mean_lights['low'] & C02_growth['decrease'], Persons['LowerThanThree'])
rule11 = ctrl.Rule(time_in_day['night'] & mean_lights['low'] & C02_growth['unchanged'], Persons['LowerThanThree'])
rule12 = ctrl.Rule(time_in_day['night'] & mean_lights['low'] & C02_growth['increase'], Persons['LowerThanThree'])

rule13 = ctrl.Rule(time_in_day['night'] & mean_lights['medium'] & C02_growth['decrease'], Persons['LowerThanThree'])
rule14 = ctrl.Rule(time_in_day['night'] & mean_lights['medium'] & C02_growth['unchanged'], Persons['LowerThanThree'])
rule15 = ctrl.Rule(time_in_day['night'] & mean_lights['medium'] & C02_growth['increase'], Persons['LowerThanThree'])

rule16 = ctrl.Rule(time_in_day['night'] & mean_lights['high'] & C02_growth['decrease'], Persons['LowerThanThree'])
rule17 = ctrl.Rule(time_in_day['night'] & mean_lights['high'] & C02_growth['unchanged'], Persons['LowerThanThree'])
rule18 = ctrl.Rule(time_in_day['night'] & mean_lights['high'] & C02_growth['increase'], Persons['LowerThanThree'])


rule1.view()
plt.show()

# Control System
persons_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18])
'''
rule1 = ctrl.Rule(mean_lights['low'], Persons['LowerThanThree'])
rule2 = ctrl.Rule(mean_lights['low'], Persons['EqualToThree'])

# Control System
persons_ctrl = ctrl.ControlSystem([rule1, rule2])

# Control System Simulation
nr_persons_ctrl = ctrl.ControlSystemSimulation(persons_ctrl)

print(df_fuzzy['Time'])

# Adicionat aqui as nossas variaveis

#nr_persons_ctrl.input['Time'] = df_fuzzy['Time']
nr_persons_ctrl.input['Lights'] = df_fuzzy['MeanLight']
#nr_persons_ctrl.input['CO2'] = df_fuzzy['CO2_eval']

# Crunch the numbers
nr_persons_ctrl.compute()

print(nr_persons_ctrl.output['Persons'])
Persons.view(sim=nr_persons_ctrl)

plt.show()

