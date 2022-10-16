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
import pickle
import math
import skfuzzy as fuzz
from skfuzzy import control as ctrl


"""*********************************  Initializations() ************************************"""

df = pd.read_csv('Proj1_Dataset.csv') 
print('Features: ', df.keys())

#sns.set_theme(style="darkgrid")

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

def Save_Max_Min(df_train):
    
    df_norm = {'S1Temp':[1,0],'S2Temp':[1,0],'S3Temp':[1,0],
               'S1Light':[1,0],'S2Light':[1,0],'S3Light':[1,0],
               'CO2':[1,0],'PIR1':[1,0],'PIR2':[1,0]}
    
    
    return

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
    
    print('binary - Precision:', precision_score(Y_pred,Y_test, average='binary'))
    print('binary - Recall:', recall_score(Y_pred,Y_test, average='binary'))
    print('binary - f1-score:', f1_score(Y_pred,Y_test, average='binary'))
    
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
         
    param_grid = {'MLP__hidden_layer_sizes':[(12,12),(14,)], 
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

df_fuzzy = Filling_data(df, outliers)

"""**************************** Fuzzy - Define Variables ***********************************"""

####### S1Temp #######

# IT will stay the same

""" Através da análise do plot """

'''sns.relplot(data = df, x = df['Time'],y = df['S1Temp'],hue="Persons")
plt.show()'''

# Cold : min till 20.40º
# hot : 20.30 till Max

min_temp = min(df_fuzzy['S1Temp'])
max_temp = max(df_fuzzy['S1Temp'])
cold_max = 20.5
hot_min = 20.0

####### SLigth_avg #######
    
df_fuzzy['S1Light'] = (df_fuzzy['S1Light']+df_fuzzy['S2Light']+df_fuzzy['S3Light'])/3
df_fuzzy.rename({'S1Light': 'SLight_Avg'}, axis=1, inplace=True)

""" Através da análise do plot """

'''sns.relplot(data = df, x = df['Time'],y = df['S1Light'],hue="Persons")
plt.show()'''

# dark : min till 350
# bright : 300 till Max

max_light = max(df_fuzzy['SLight_Avg'])
min_light = min(df_fuzzy['SLight_Avg'])
dark_max = 350
bright_min = 300

####### CO2 #######

#df_fuzzy['CO2'] = df_fuzzy['CO2'].rolling(8).mean() #8 indices correspondem a 4 minutos 
df_fuzzy['CO2'] = np.append(np.diff(df_fuzzy['CO2']),0)
df_fuzzy.rename({'CO2': 'CO2_Dif'}, axis=1, inplace=True)


""" Através da análise do plot """

'''sns.relplot(data = df, x = df['Time'],y = df['CO2_Dif'],hue="Persons")
plt.show()'''

# Decreasing: min till -2
# Constant: -4.5 till 5
# Increasing: 4 till max 

'''plt.plot(df_fuzzy['CO2_Dif'])
plt.show()'''

max_CO2_var = max(df_fuzzy['CO2_Dif'])
min_CO2_var = min(df_fuzzy['CO2_Dif'])
CO2_drecr_max = -2
CO2_const_min = -6
CO2_comst_max = 5
CO2_incr_min = 4

####### Data ########

df_fuzzy.rename({'CO2': 'CO2_dev'}, axis=1, inplace=True)
df_fuzzy.rename({'S1Light': 'light_avg'}, axis=1, inplace=True)
df_fuzzy = df_fuzzy.drop(['Date','Time','S2Temp','S3Temp','S2Light','S3Light',
                          'PIR1', 'PIR2'], axis=1)

print(df_fuzzy.columns)


"""************************** Removing Test Set *********************************"""

x_train, x_test, y_train, y_test = train_test_split(df_fuzzy.iloc[:,:(len(df_fuzzy.columns)-1)].values, 
                                                    df_fuzzy.iloc[:,(len(df_fuzzy.columns)-1)].values,
                                                    test_size=0.1,shuffle=True)



"""*********** [Training data] Dealing with noise - Moving Average ***************"""

# train_set = Clean_Noise(train_set)

"""************************* Model Fine-Tunning ******************************"""

#model_FineTuning(x_train, y_train, True) #To do Grid_Seacrh CV with SMOTE

"""********************* Normalize the Training Set *************************"""

#x_train_norm = Normalize_data(x_train, x_train)

"""********************* Normalize the Test Set *************************"""

#x_test_norm = Normalize_data(x_test, x_train)

"""********************* Balance the Training Set ***************************"""

#if sm == True:
    #x_train_norm,y_train = Balance_data(x_train_norm, y_train)

"""**************************** Fuzzy System Inputs ***********************************"""

temp = ctrl.Antecedent(np.arange(min_temp, max_temp+1, 0.02), 'Temp') 
mean_lights = ctrl.Antecedent(np.arange(min_light, max_light+1 , 1), 'Mean_Lights')
CO2_dif = ctrl.Antecedent(np.arange(min_CO2_var, max_CO2_var+1 , 5), 'CO2_Dif')

"""**************************** Fuzzy System Output ***********************************"""

Persons = ctrl.Consequent(np.arange(0, 3+1, 1), 'Persons')

"""*********************************** Fuzzifier ***************************************"""


###### Input ########

temp['cold'] = fuzz.trimf(temp.universe,[min_temp, min_temp ,cold_max])
temp['hot'] = fuzz.trimf(temp.universe,[hot_min, (hot_min+ max_temp)/2 ,max_temp])

mean_lights['dark'] = fuzz.trimf(mean_lights.universe,[min_light, (min_light+dark_max)/2 , dark_max])
mean_lights['bright'] = fuzz.trimf(mean_lights.universe,[bright_min, (bright_min+max_light)/2 , max_light])

CO2_dif['decrease'] = fuzz.trimf(CO2_dif.universe, [min_CO2_var, (min_CO2_var +CO2_drecr_max)/2, CO2_drecr_max]) 
CO2_dif['constant'] = fuzz.trimf(CO2_dif.universe, [CO2_const_min, 0, CO2_comst_max])
CO2_dif['increase'] = fuzz.trimf(CO2_dif.universe, [CO2_incr_min, (CO2_incr_min+max_CO2_var)/2, max_CO2_var])


#Plots
temp.view()
mean_lights.view()
CO2_dif.view()
plt.show()


###### Output  #######

Persons['LowerThanThree'] = fuzz.trimf(Persons.universe, [0,0,2])
Persons['EqualToThree'] = fuzz.trimf(Persons.universe, [1, 3, 3])

Persons.view()
plt.show()

"""*********************************** Rules ***************************************"""


rule1 = ctrl.Rule(temp['cold'] & CO2_dif['decrease'], Persons['LowerThanThree'])
rule2 = ctrl.Rule(temp['cold'] & CO2_dif['constant'], Persons['LowerThanThree'])
rule3 = ctrl.Rule(temp['cold'] & CO2_dif['increase'], Persons['LowerThanThree'])

rule4 = ctrl.Rule(temp['cold'] & mean_lights['dark'], Persons['LowerThanThree'])
rule5 = ctrl.Rule(temp['cold'] & mean_lights['bright'], Persons['LowerThanThree'])

rule6 = ctrl.Rule(temp['hot'] & CO2_dif['decrease'], Persons['LowerThanThree'])
rule7 = ctrl.Rule(temp['hot'] & CO2_dif['constant'], Persons['LowerThanThree'])
rule8 = ctrl.Rule(temp['hot'] & CO2_dif['increase'], Persons['EqualToThree'])

rule9 = ctrl.Rule(temp['hot'] & mean_lights['dark'], Persons['LowerThanThree'])
rule10 = ctrl.Rule(temp['hot'] & mean_lights['bright'], Persons['EqualToThree'])

rule11 = ctrl.Rule(mean_lights['dark'] & CO2_dif['decrease'], Persons['LowerThanThree'])
rule12 = ctrl.Rule(mean_lights['dark'] & CO2_dif['constant'], Persons['LowerThanThree'])
rule13 = ctrl.Rule(mean_lights['dark'] & CO2_dif['increase'], Persons['EqualToThree'])

rule14 = ctrl.Rule(mean_lights['bright'] & CO2_dif['decrease'], Persons['LowerThanThree'])
rule15 = ctrl.Rule(mean_lights['bright'] & CO2_dif['constant'], Persons['EqualToThree'])
rule16 = ctrl.Rule(mean_lights['bright'] &  CO2_dif['increase'], Persons['EqualToThree'])





"""*********************************** Inference Engine ***************************************"""

# Control System
persons_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, 
                                   rule9,rule10,rule11,rule12,rule13,rule14,rule15,rule16])

# Control System Simulation
nr_persons_ctrl = ctrl.ControlSystemSimulation(persons_ctrl)

"""*********************************** Defuzzifier ***************************************"""

y_pred = np.array([])

print(x_test)

for i in range(len(x_test)):
    nr_persons_ctrl.input['Temp'] = x_test[i,0]
    nr_persons_ctrl.input['Mean_Lights'] = x_test[i,1]
    nr_persons_ctrl.input['CO2_Dif'] = x_test[i,2]
    nr_persons_ctrl.compute()
    y_pred = np.append(y_pred, math.ceil(nr_persons_ctrl.output['Persons']))
    
    
        
"""*********************************** Output ***************************************"""

for i in range (len(y_pred)):
    if y_pred[i] == 3:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
        
    if y_test[i] == 3:
        y_test[i] = 1
    else:
        y_test[i] = 0


Print_Scores_binary(y_pred,y_test)