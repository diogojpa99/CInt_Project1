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
from sklearn.metrics import precision_score, recall_score,f1_score, balanced_accuracy_score, accuracy_score
from scipy.stats import pearsonr
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
import sklearn.metrics as met
import math
import skfuzzy as fuzz
from skfuzzy import control as ctrl


"""*********************************  Initializations() ************************************"""

df = pd.read_csv('Proj1_Dataset.csv') 
print('Features: ', df.keys())

sns.set_theme(style="darkgrid")

n_features = len(df.columns) ## Number of features
missing_data = [(-1,'Feature')] #We will not count the first position
outliers =  [(-1,'Feature')] 

sm = False #SMOTE

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
    scaler = scaler.fit(train_set) #Normalize to the min-max of the training set
    
    return scaler.transform(data)

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
        
    print('Binary - Precision:', precision_score(Y_pred,Y_test, average='binary'))
    print('Binary - Recall:', recall_score(Y_pred,Y_test, average='binary'))
    print('Binary - f1-score:', f1_score(Y_pred,Y_test, average='binary'))
    
    return

#Model Hyper-parameter tunning
def model_FineTuning(X_train, Y_train, sm ):

    alphas = [0.00005,  0.0001, 0.000125]
    l_rates =[0.01, 0.01115, 0.01225]
              
    if sm == True:
        pipeline = imbpipeline(steps = [['smote', SMOTE(random_state=11)],
                                        ['norm', MinMaxScaler()],
                                        ['MLP', MLPClassifier()]])
    else:
        pipeline = Pipeline(steps = [['norm', MinMaxScaler()],
                                     ['MLP', MLPClassifier()]])
        
    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=11)
         
    param_grid = {'MLP__hidden_layer_sizes':[(6,),(5,5)], 
                  'MLP__activation':['tanh','relu'], 
                  'MLP__solver':['adam'],
                  'MLP__alpha':alphas, 
                  'MLP__learning_rate':['constant','adaptive'],
                  'MLP__max_iter':[1000],
                  'MLP__learning_rate_init':l_rates}
    
    scoring = {'accuracy':met.make_scorer(met.accuracy_score)}
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='roc_auc',
                               cv=stratified_kfold)
    
    grid_search.fit(X_train, Y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    return

# According to model()
def MLP_binary_model(X_train, Y_train, X_test, Y_test):
    
    mlp = MLPClassifier(hidden_layer_sizes=(6,), activation='tanh',
                        solver = 'adam', alpha = 0.0001, learning_rate='constant', 
                        max_iter = 1000, learning_rate_init = 0.01225).fit(X_train,Y_train)
    
    Y_pred = mlp.predict(X_test)
    
    Print_Scores_binary(Y_pred,Y_test)

    return


#Plotonfusion Matrix
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
    
    
def Tune_fuzzy_classifier(x_train, y_train, nr_persons_ctrl):
    
    y_pred_train = np.array([])
    
    for i in range(len(x_train)):
        nr_persons_ctrl.input['Temp'] = x_train[i,0]
        nr_persons_ctrl.input['Lights_Avg'] = x_train[i,1]
        nr_persons_ctrl.input['CO2_Var'] = x_train[i,2]
        nr_persons_ctrl.compute()
        y_pred_train = np.append(y_pred_train, math.ceil(nr_persons_ctrl.output['Persons']))
        
    for i in range (len(y_pred_train)):
        if y_pred_train[i] == 3:
            y_pred_train[i] = 1
        else:
            y_pred_train[i] = 0

    Print_Scores_binary(y_pred_train,y_train)
    
    return
    
    
def Fuzzy_classifier(x_test, y_test, nr_persons_ctrl):
    
    y_pred = np.array([])


    for i in range(len(x_test)):
        nr_persons_ctrl.input['Temp'] = x_test[i,0]
        nr_persons_ctrl.input['Lights_Avg'] = x_test[i,1]
        nr_persons_ctrl.input['CO2_Var'] = x_test[i,2]
        nr_persons_ctrl.compute()
        y_pred = np.append(y_pred, math.ceil(nr_persons_ctrl.output['Persons']))
        
        
    for i in range (len(y_pred)): #Pass for binary
        if y_pred[i] == 3:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
            
    Print_Scores_binary(y_pred,y_test)

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

df_fuzzy = Filling_data(df, outliers)

"""**************************** Fuzzy - Define Variables ***********************************"""

####### S1Temp #######

''' S1Temp Will Stay the same '''

'''sns.relplot(data = df, x = df['Time'],y = df['S1Temp'],hue="Persons")
plt.show()'''

min_temp = min(df_fuzzy['S1Temp'])
max_temp = max(df_fuzzy['S1Temp'])

####### CO2 Var #######

''' Variation between CO2 instances '''

df_fuzzy['CO2'] = np.append(np.diff(df_fuzzy['CO2']),0)

max_CO2 = max(df_fuzzy['CO2'])
min_CO2 = min(df_fuzzy['CO2'])

####### Light Avg #######

''' Mean of all the three SLight Sensors '''

df_fuzzy['S1Light'] = (df_fuzzy['S1Light']+df_fuzzy['S2Light']+df_fuzzy['S3Light'])/3

max_light_avg = max(df_fuzzy['S1Light'])
min_light_avg = min(df_fuzzy['S1Light'])

####### Data ########

df_fuzzy.rename({'CO2': 'CO2_dev'}, axis=1, inplace=True)
df_fuzzy.rename({'S1Light': 'light_avg'}, axis=1, inplace=True)
df_fuzzy = df_fuzzy.drop(['Date','Time','S2Temp','S3Temp','S2Light','S3Light',
                          'PIR1', 'PIR2'], axis=1)


"""************************** Removing Test Set *********************************"""

x_train, x_test, y_train, y_test = train_test_split(df_fuzzy.iloc[:,:(len(df_fuzzy.columns)-1)].values, 
                                                    df_fuzzy.iloc[:,(len(df_fuzzy.columns)-1)].values,
                                                    test_size=0.1,shuffle=True)

"""*********** [Training data] Dealing with noise - Moving Average ***************"""

# train_set = Clean_Noise(train_set)

"""************************* Model Fine-Tunning ******************************"""

for i in range (len(y_train)):
    if y_train[i] == 3:
        y_train[i] = 1
    else:
        y_train[i] = 0
        
#model_FineTuning(x_train, y_train, False) #To do Grid_Seacrh CV without SMOTE

"""********************* Normalize the Training Set *************************"""

x_train_norm = Normalize_data(x_train, x_train)

"""********************* Normalize the Test Set *************************"""

x_test_norm = Normalize_data(x_test, x_train)

"""********************* Balance the Training Set ***************************"""

#if sm == True:
    #x_train_norm,y_train = Balance_data(x_train_norm, y_train)

"""**************************** Fuzzy System Inputs ***********************************"""

temp = ctrl.Antecedent(np.arange(min_temp, max_temp+1, 0.001), 'Temp') 
mean_lights = ctrl.Antecedent(np.arange(min_light_avg, max_light_avg+1, 1), 'Lights_Avg')
CO2_deriv = ctrl.Antecedent(np.arange(min_CO2, max_CO2+1, 1), 'CO2_Var')

"""**************************** Fuzzy System Output ***********************************"""

Persons = ctrl.Consequent(np.arange(0, 3+1, 1), 'Persons')

"""*********************************** Fuzzifier ***************************************"""


###### Input ########

temp['cold'] = fuzz.trimf(temp.universe,[min_temp, 20.1 ,20.3])
temp['hot'] = fuzz.trapmf(temp.universe,[20.0, 20.8 , max_temp-0.1, max_temp])

mean_lights['low'] = fuzz.trimf(mean_lights.universe,[min_light_avg, min_light_avg, 180])
mean_lights['medium'] = fuzz.trimf(mean_lights.universe,[140, 190, 375])
mean_lights['high'] = fuzz.trimf(mean_lights.universe,[330, max_light_avg, max_light_avg]) 

CO2_deriv['decrease'] = fuzz.trimf(CO2_deriv.universe, [min_CO2, min_CO2, -5]) 
CO2_deriv['constant'] = fuzz.trimf(CO2_deriv.universe, [-15, 0, 15])
CO2_deriv['increase'] = fuzz.trapmf(CO2_deriv.universe, [5, 30, max_CO2, max_CO2])


#Plots
temp.view()
mean_lights.view()
CO2_deriv.view()
plt.show()


###### Output  #######

Persons['LowerThanThree'] = fuzz.trimf(Persons.universe, [0,0,2])
Persons['EqualToThree'] = fuzz.trimf(Persons.universe, [1, 3, 3])

Persons.view()
plt.show()

"""*********************************** Rules ***************************************"""

rule1 = ctrl.Rule(mean_lights['low'] & CO2_deriv['decrease'], Persons['LowerThanThree'])
rule2 = ctrl.Rule(mean_lights['low'] & CO2_deriv['constant'], Persons['LowerThanThree'])
rule3 = ctrl.Rule(mean_lights['low'] & CO2_deriv['increase'], Persons['EqualToThree'])

rule4 = ctrl.Rule(mean_lights['medium'] & CO2_deriv['decrease'], Persons['LowerThanThree'])
rule5 = ctrl.Rule(mean_lights['medium'] & CO2_deriv['constant'], Persons['LowerThanThree'])
rule6 = ctrl.Rule(mean_lights['medium'] & CO2_deriv['increase'], Persons['EqualToThree'])

rule7 = ctrl.Rule(mean_lights['high'] & CO2_deriv['decrease'], Persons['LowerThanThree'])
rule8 = ctrl.Rule(mean_lights['high'] & CO2_deriv['constant'], Persons['EqualToThree'])
rule9 = ctrl.Rule(mean_lights['high'] & CO2_deriv['increase'], Persons['EqualToThree'])

rule10 = ctrl.Rule(temp['cold'] & CO2_deriv['decrease'], Persons['LowerThanThree'])
rule11 = ctrl.Rule(temp['cold'] & CO2_deriv['constant'], Persons['LowerThanThree'])
rule12 = ctrl.Rule(temp['cold'] & CO2_deriv['increase'], Persons['LowerThanThree'])

rule13 = ctrl.Rule(temp['hot'] & CO2_deriv['decrease'], Persons['LowerThanThree'])
rule14 = ctrl.Rule(temp['hot'] & CO2_deriv['constant'], Persons['EqualToThree'])
rule15 = ctrl.Rule(temp['hot'] & CO2_deriv['increase'], Persons['EqualToThree'])

rule16 = ctrl.Rule(temp['cold'] & mean_lights['low'], Persons['LowerThanThree'])
rule17 = ctrl.Rule(temp['cold'] & mean_lights['medium'], Persons['LowerThanThree'])
rule18 = ctrl.Rule(temp['cold'] & mean_lights['high'], Persons['LowerThanThree'])

rule19 = ctrl.Rule(temp['hot'] & mean_lights['low'], Persons['LowerThanThree'])
rule20 = ctrl.Rule(temp['hot'] & mean_lights['medium'], Persons['LowerThanThree'])
rule21 = ctrl.Rule(temp['hot'] & mean_lights['high'], Persons['EqualToThree'])

"""*********************************** Inference Engine ***************************************"""

# Control System
"""persons_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, 
                                   rule9,rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,rule18,rule20,rule21])"""

persons_ctrl = ctrl.ControlSystem([rule1, rule2, rule6, rule8, rule9,rule10,rule11,
                                   rule12,rule14, rule15,rule17,rule18,rule20,rule21])

"""*********************************** Defuzzifier ***************************************"""

# Control System Simulation
nr_persons_ctrl = ctrl.ControlSystemSimulation(persons_ctrl)
       
"""*********************************** Output ***************************************"""

#Tune_fuzzy_classifier(x_train, y_train,nr_persons_ctrl)

for i in range (len(y_test)): #Pass for binary        
    if y_test[i] == 3:
        y_test[i] = 1
    else:
        y_test[i] = 0

print("------------- Fuzzy Output -------------------")
Fuzzy_classifier(x_test, y_test, nr_persons_ctrl)
        
print("--------------- MLP Output -------------------")
MLP_binary_model(x_train_norm, y_train, x_test_norm, y_test)