
import pandas as pd # for data handling
import numpy as np # for data manipulation 
import sklearn as sk
from matplotlib import pyplot as plt # for plotting
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder # For encoding class variables
from sklearn.model_selection import train_test_split # for train and test split
from sklearn.svm import SVC # to built svm model
from sklearn import svm # inherits other SVM objects
from sklearn import metrics # to calculate classifiers accuracy
from sklearn.model_selection import cross_val_score # to perform cross validation
from sklearn.preprocessing import StandardScaler # to perform standardization
from sklearn.model_selection import GridSearchCV # to perform grid search for all classifiers
from sklearn import tree # to perform decision tree classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors # to perform knn
from sklearn import naive_bayes # to perform Naive Bayes
from sklearn.metrics import classification_report # produce classifier reports
from sklearn.ensemble import RandomForestClassifier # to perform ensemble bagging - random forest
from sklearn.ensemble import AdaBoostClassifier # to perform ensemble boosting
from sklearn.metrics import roc_curve, auc # to plot ROC Curve
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic(u'matplotlib inline')


# Data Manipulation

df_train = pd.read_table('echocardiogram.txt',sep=',',na_values=['?'])
df_train.columns = ["survival", "still-alive", "age-at-heart-attack", "pericardial-effusion","fractional-shortening","epss","lvdd","wall-motion-score","wall-motion-index","mult","name","group","alive-at-1"]
df_train.columns

df_train.shape

df_train.head(5)

df_train = df_train.sort_values('survival')

df_train = df_train.reset_index()
df_train.head()

# Removal of unnessary Data

del df_train["group"]
del df_train["name"] 
del df_train["wall-motion-score"]
del df_train["mult"]
del df_train["index"]

df_train.head()

lable = df_train.columns
print lable

df_train.dtypes

# Seperating Independent and Target Variables: 

data_x = df_train[lable[0:8]].copy()
data_x = data_x.fillna(value=-1.0)
data_y = df_train[lable[-1]].copy()
data_y = data_y.fillna(value=-1.0)
print('Independent var: \n',data_x.head(10),'\n')
print('Dependent var: \n',data_y.head(10))

# Target Variable Encoding

encode_obj = LabelEncoder()
data_y = encode_obj.fit_transform(data_y)
print('sample values of target values:\n',data_y[0:10])

# Setting a benchmark accuracy for classifiers using Raw Data & Naive Bayes

test_x_train,test_x_test,test_y_train,test_y_test = train_test_split(data_x,data_y,train_size=0.75,test_size=0.25,random_state=1)

nbclf = naive_bayes.GaussianNB()
nbclf = nbclf.fit(test_x_train, test_y_train)
nbpreds_test = nbclf.predict(test_x_test)
print('Accuracy obtained from train-test split on training data is:',nbclf.score(test_x_train, test_y_train))
print('Accuracy obtained from train-test split on testing data is:',nbclf.score(test_x_test, test_y_test))

test_eval_result = cross_val_score(nbclf, data_x, data_y, cv=10, scoring='accuracy')
print('Accuracy obtained from 10-fold cross validation on actual raw data is:',test_eval_result.mean())

data_y = df_train[lable[-1]].copy()
data_y = data_y.fillna(value=-1.0)

plt.subplot(221)
plt.hist(data_x['survival'])
plt.subplot(222)
plt.hist(data_x['still-alive'])
plt.subplot(223)
plt.hist(data_x['age-at-heart-attack'])
plt.subplot(224)
plt.hist(data_x['pericardial-effusion'])

# Variable survival is normally distributed and variables still-alive, age-at-heart-attack,pericardial-effusion are skewed to the right 

plt.subplot(221)
plt.hist(data_x['fractional-shortening'])
plt.subplot(222)
plt.hist(data_x['epss'])
plt.subplot(223)
plt.hist(data_x['lvdd'])
plt.subplot(224)
plt.hist(data_x['wall-motion-index'])


# Variable epss is normally distributed and variables fractional-shortening,vdd,wall-motion-index are skewed to the right 

# Data Mungling and Data Reduction

data_y = data_y.to_frame(name='alive-at-1')
data_x1 = data_x[((data_x['survival'] > -0.1) & (data_x['still-alive'] > -0.1)) & (data_x['age-at-heart-attack'] > -0.1) & (data_x['pericardial-effusion'] > -0.1) & (data_x['fractional-shortening'] > -0.1) & (data_x['epss'] > -0.1) & (data_x['lvdd'] > -0.1) & (data_x['wall-motion-index'] > -0.1)]
data_y1 = data_y[data_y['alive-at-1'] > -0.1]
data_y1.head()

data_y1 = data_y1.reset_index()
del data_y1["index"]
del data_y1["level_0"]

data_y1.head()

data_y1.shape

data_x1.shape

data_x1 = data_x1.reset_index()
del data_x1["index"]
data_x1.head(5)
plt.figure() 
data_y1.plot.hist(alpha=0.5)

# Validating the cleaned dataset with benchmark accuracy obtained

nbclf = naive_bayes.GaussianNB()
data_x1 = data_x1.drop(data_x1.index[73:])
data_x1_train,data_x1_test,data_y1_train,data_y1_test = train_test_split(data_x1,data_y1,train_size=0.8,test_size=0.2,random_state=1)

data_x1_train.shape

data_x1_test.shape

data_y1_train.shape

data_y1_test.shape

nbclf = nbclf.fit(data_x1_train, data_y1_train)
nbpreds_test = nbclf.predict(data_x1_test)
nb_eval_result1 = cross_val_score(nbclf, data_x1, data_y1, cv=10, scoring='accuracy')
print('Mean accuracy with 10 fold cross validation on Naive Bayes with treated data: ',nb_eval_result1.mean())

# Core Model Building with SVM Classifier


def funct_svm(kernal_type,xTrain,yTrain,xTest,yTest):
    svm_obj=SVC(kernel=kernal_type)
    svm_obj.fit(xTrain,yTrain)
    yPredicted=svm_obj.predict(xTest)
    print('Accuracy Score of',kernal_type,'Kernal SVM is:',metrics.accuracy_score(yTest,yPredicted))
    return metrics.accuracy_score(yTest,yPredicted)

get_ipython().magic(u'timeit 10')
PN_linear_result = funct_svm('linear',data_x1_train,data_y1_train,data_x1_test,data_y1_test)

get_ipython().magic(u'timeit 10')
PN_rbf_result = funct_svm('rbf',data_x1_train,data_y1_train,data_x1_test,data_y1_test)

get_ipython().magic(u'timeit 10')
PN_poly_result = funct_svm('poly',data_x1_train,data_y1_train,data_x1_test,data_y1_test)

get_ipython().magic(u'timeit 10')
PN_sigmoid_result = funct_svm('sigmoid',data_x1_train,data_y1_train,data_x1_test,data_y1_test)


# Parameter tuning on the best Kernal for SVM with 5-fold cross validation

def funct_tune_svm(kernal_type,margin_val,xData,yData,k,eval_param):
    if(kernal_type=='poly'):
        svm_obj=SVC(kernel=kernal_type,degree=margin_val) 
    eval_result = cross_val_score(svm_obj, xData, yData, cv=k, scoring=eval_param)
    return eval_result.mean()

accu_list = list()
for c in np.arange(0.1,10,1):
    result = funct_tune_svm('poly',c,data_x1,data_y1,5,'accuracy')
    accu_list.append(result)


np.arange(0.1,10,1)
C_values=list(np.arange(0.1,10,1))

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

plt.plot(C_values,accu_list)
plt.xticks(np.arange(0.1,10,1))
plt.xlabel('Value of C for SVC')
plt.ylabel('Cross-Validated Accuracy')

tuning_poly_svm = pd.DataFrame(columns=['Parameter Degree', 'Accuracy'])
tuning_poly_svm['Parameter Degree'] = np.arange(0.1,10,1)
tuning_poly_svm['Accuracy'] = accu_list
tuning_poly_svm


# Visualization of kernal Margin and boundries

plt.scatter(data_x1['survival'],data_x1['still-alive'])

plt.scatter(data_x1['fractional-shortening'],data_x1['still-alive'])


# Visualizing the margin modeled 

X = data_x1[['survival','still-alive']].copy()
X = np.array(X)
y = np.array(data_y1)

clf = SVC(kernel='poly', degree=1.1, gamma = 0.05,C=1.6)
clf.fit(X, y)

title = ('SVC with poly kernel(with degree=1.1 & gamma=0.05 & C=1.6)')

plt.scatter(X[:,0], X[:,1], c='y', s=30, cmap=plt.cm.Paired)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors

ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none')
ax.set_xlabel('survival')
ax.set_ylabel('still-alive')
ax.set_title(title)
plt.show()

final_model = SVC(kernel='poly', C=1.6, gamma=0.005, degree=1)
print('Final Model Detail:\n',final_model)
final_model_score = final_model.fit(data_x1_train, data_y1_train).decision_function(data_x1_test)



final_eval_result = cross_val_score(final_model, data_x1, data_y1, cv=10, scoring='accuracy')

print('\nAccuracy obtained from final model with 10 fold CV:\n',final_eval_result.mean())

fpr, tpr, _ = roc_curve(data_y1_test,final_model_score)
roc_auc= auc(fpr, tpr)
print('\nROC Computed Area Under Curve:\n',roc_auc)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Best SVM model')
plt.legend(loc="lower right")
plt.show()


# Personal idea

col_name =data_y1.columns[0]
data_y1=data_y1.rename(columns = {col_name:'DoA'})

Person = []
for row in data_y1['DoA']:
    if row == 1.0:
        Person.append('Alive')
    else:
        Person.append('Dead')
        
        
data_y1['Person'] = Person   

data_y1.head(30)

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(data_x1_train, data_y1_train)

knn.score(data_x1_test,data_y1_test)

Result =  dict(zip(data_y1.DoA.unique() , data_y1.Person.unique()))
Result


# Let's test the Prediction Model

# Test 1

Result_after_one_year = knn.predict([[0.50,1.0,59.0,0.0,0.130,16.400,4.96,1.37]])
Result_after_one_year


Result_after_one_year = int(Result_after_one_year)
Result[Result_after_one_year]


# Test 2

Result_after_one_year = knn.predict([[19.00,0.0,46.000,0.0,0.340,0.000,5.090,1.140]])
Result_after_one_year

Result_after_one_year = int(Result_after_one_year)
Result[Result_after_one_year]

# Final Result Graph 

pd.Series(data_y1.DoA).value_counts().sort_values().plot(kind='bar',color='gr')
plt.title("The Bar Graph of people diagnose with Heart Attack")
plt.ylabel('No of People')
plt.xlabel('Number of People Alive/dead after one year')

NA = mpatches.Patch(color='green', label='Alive')
EU = mpatches.Patch(color='red', label='Dead')
plt.legend(handles=[NA,EU,], loc=2)

