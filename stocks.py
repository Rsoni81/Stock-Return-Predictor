## author: Sahil Sulekhiya
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm
training_dataset = pd.read_csv('train.csv')
x = []
for _ in training_dataset['portfolio_id']:
   x.append(float(( _[2:])))
training_dataset['corrected_id'] = x
y = []
for _ in training_dataset['office_id']:
   y.append(float( _[3:]))
training_dataset['corrected_office'] = y
swiss_rate = 1005560/10**6
pound_rate = 1346840/10**6
euro = 1175450/10**6
yuan = 8880/10**6
training_dataset['currency'].replace({'USD':1,'GBP':pound_rate,'CHF':swiss_rate,'EUR':euro,'JPY':yuan},inplace = True)
training_dataset['pf_category'].replace({'A':0,'B':1,'C':2,'D':3,'E':4},inplace = True)
training_dataset['country_code'].replace({'T':0,'N':1,'Z':2,'U':3,'M':4},inplace = True)
training_dataset['type'].replace({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7},inplace = True)
training_dataset['indicator_code'].replace({True:1, False:0})
training_dataset['hedge_value'].replace({True:1, False:0})
training_dataset['status'].replace({True:1, False:0})
training_dataset.fillna(-99999,inplace = True)
dat = []
mon = []
yea = []
for _ in training_dataset['start_date']:
    d = _ % 100
    m = ((_ - d)//100)%100
    y = _//10000 
    dat.append(d)
    mon.append(m)
    yea.append(y)

training_dataset['start_y'] = yea
training_dataset['start_m'] = mon
training_dataset['start_d'] = dat
dat = []
mon = []
yea = []
for _ in training_dataset['creation_date']:
    d = _ % 100
    m = ((_ - d)//100)%100
    y = _//10000 
    dat.append(d)
    mon.append(m)
    yea.append(y)

training_dataset['creation_y'] = yea
training_dataset['creation_m'] = mon
training_dataset['creation_d'] = dat
dat = []
mon = []
yea = []
for _ in training_dataset['sell_date']:
    d = _ % 100
    m = ((_ - d)//100)%100
    y = _//10000 
    dat.append(d)
    mon.append(m)
    yea.append(y)

training_dataset['sell_y'] = yea
training_dataset['sell_m'] = mon
training_dataset['sell_d'] = dat
training_dataset['sold'] = training_dataset['sold'] * training_dataset['currency']
training_dataset['bought'] = training_dataset['bought'] * training_dataset['currency']
X = training_dataset[['corrected_id','pf_category','start_y','start_m','start_d','creation_y','creation_m','creation_d','sold','country_code','euribor_rate','currency','libor_rate','bought','indicator_code','type','hedge_value','status']]
y = training_dataset['return']
X_train = np.array(X, dtype = float)
y_train = np.array(y, dtype = float)
#X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
clf=LinearRegression(n_jobs=-1)

clf.fit(X_train,y_train)
accuracy_train = clf.score(X_train,y_train)
 
print('Training Accuracy is: ', accuracy_train)

test_dataset = pd.read_csv('test.csv')
x = []
for _ in test_dataset['portfolio_id']:
   x.append(float(( _[2:])))
test_dataset['corrected_id'] = x
y = []
for _ in test_dataset['office_id']:
   y.append(float( _[3:]))
test_dataset['corrected_office'] = y
test_dataset['currency'].replace({'USD':1,'GBP':pound_rate,'CHF':swiss_rate,'EUR':euro,'JPY':yuan},inplace = True)
test_dataset['pf_category'].replace({'A':0,'B':1,'C':2,'D':3,'E':4},inplace = True)
test_dataset['country_code'].replace({'T':0,'N':1,'Z':2,'U':3,'M':4},inplace = True)
test_dataset['type'].replace({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7},inplace = True)
test_dataset['indicator_code'].replace({True:1, False:0})
test_dataset['hedge_value'].replace({True:1, False:0})
test_dataset['status'].replace({True:1, False:0})
test_dataset.fillna(-99999,inplace = True)
dat = []
mon = []
yea = []
for _ in test_dataset['start_date']:
    d = _ % 100
    m = ((_ - d)//100)%100
    y = _//10000 
    dat.append(d)
    mon.append(m)
    yea.append(y)

test_dataset['start_y'] = yea
test_dataset['start_m'] = mon
test_dataset['start_d'] = dat
dat = []
mon = []
yea = []
for _ in test_dataset['creation_date']:
    d = _ % 100
    m = ((_ - d)//100)%100
    y = _//10000 
    dat.append(d)
    mon.append(m)
    yea.append(y)

test_dataset['creation_y'] = yea
test_dataset['creation_m'] = mon
test_dataset['creation_d'] = dat
dat = []
mon = []
yea = []
for _ in test_dataset['sell_date']:
    d = _ % 100
    m = ((_ - d)//100)%100
    y = _//10000 
    dat.append(d)
    mon.append(m)
    yea.append(y)

test_dataset['sell_y'] = yea
test_dataset['sell_m'] = mon
test_dataset['sell_d'] = dat
test_dataset['sold'] = test_dataset['sold'] * test_dataset['currency']
test_dataset['bought'] = test_dataset['bought'] * test_dataset['currency']
X = test_dataset[['corrected_id','pf_category','start_y','start_m','start_d','creation_y','creation_m','creation_d','sold','country_code','euribor_rate','currency','libor_rate','bought','indicator_code','type','hedge_value','status']]

X_test = np.array(X, dtype = float)
pred = clf.predict(X_test)
result = "portfolio_id, return\n"
print(X_test.shape)
for i in range(X_test.shape[0]):
    result = result + str(test_dataset['portfolio_id'][i]) +  ", " + str(pred[i]) + "\n"
obj = open("mysubmissions.csv","w")
obj.write(result)
obj.close()
print(result)
