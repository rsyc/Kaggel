import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

#from sklearn import preprocessing
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#import seaborn as sns
#sns.set(style="white")
#sns.set(style="whitegrid", color_codes=True)
#-----------------------------------------------------------------------------------------------------------------------------
def func(x):
  if x[u'age'] >=20 and (x[u'age'] <30): return 20
  elif x[u'age'] >=30 and (x[u'age'] <40): return 30
  elif x[u'age'] >=40 and (x[u'age'] <50): return 40
  elif x[u'age'] >=50 and (x[u'age'] <60): return 50
  elif x[u'age'] >=60 and (x[u'age'] <70): return 60
  elif x[u'age'] >=70 and (x[u'age'] <80): return 70
  elif x[u'age'] >=80 and (x[u'age'] <90): return 80
  elif x[u'age'] >=90 and (x[u'age'] <100): return 90
  elif x[u'age'] >=100 : return 100
  else: return 0
#------------------------------------------------------------------------------------------------------------------------------




train_df = pd.read_csv('..\Data\cs-training.csv')
test_df = pd.read_csv('..\Data\cs-test.csv')


#Finding Nulls in the training and test set
print ('Sum of nulls in each column (train0:', pd.isnull(train_df).sum(axis=0)) #--> They are Nulls in "MonthlyIncome" and "NumberOfDependents" in training set
print ('Sum of nulls in each column (test):', pd.isnull(test_df).sum(axis=0)) #--> They are Nulls in "MonthlyIncome" and "NumberOfDependents" in test set
# see the effect of nulls in "MonthlyIncome" and "NumberOfDependents" on "SeriousDlqin2yrs"
train_df['MonthlyIncome_Null'] = pd.isnull(train_df['MonthlyIncome'])
test_df['MonthlyIncome_Null'] = pd.isnull(test_df['MonthlyIncome'])
incomeGrouped = train_df.groupby('MonthlyIncome_Null')
Dlqin = incomeGrouped['SeriousDlqin2yrs'].aggregate(np.mean).reset_index()
print (Dlqin)
train_df['Dependent_Null'] = pd.isnull(train_df['NumberOfDependents'])
test_df['Dependent_Null'] = pd.isnull(test_df['NumberOfDependents'])
DependentGrouped = train_df.groupby('Dependent_Null')
Dlqin = DependentGrouped['SeriousDlqin2yrs'].aggregate(np.mean).reset_index()
print (Dlqin)
#Drop rows with nulls in column "NumberOfDependents" as these nulls seem to have low deliquency rate
print (train_df.shape, len(train_df))
train_df=train_df.dropna(subset=['NumberOfDependents'])
train_df.reset_index()
print (train_df.shape, len(train_df))
print ('Sum of nulls in each column after dropping nulls in NumberOfDependents:', pd.isnull(train_df).sum(axis=0))


# Exploring  Data and plotting some results
Ycount_values=train_df[u'SeriousDlqin2yrs'].value_counts()
print ('', Ycount_values)
count_OK = len(train_df[train_df[u'SeriousDlqin2yrs']==0])
count_Dlqin = len(train_df[train_df[u'SeriousDlqin2yrs']==1])
pct_of_OK = count_OK/(count_OK+count_Dlqin)
print ("percentage of no delinquency=", count_OK*100.0/(count_OK+count_Dlqin))
print ("percentage of delinquency=", count_Dlqin*100.0/(count_OK+count_Dlqin))
DlqinGrouped=train_df.groupby(u'SeriousDlqin2yrs').mean()
fig_num=1
plt.figure(fig_num)
plt.xlabel('SeriousDlqin2yrs', fontsize=16)
plt.ylabel('Count', fontsize=16) 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
y_pos = [0,1]
label = ['0','1']
plt.xticks(y_pos, label, fontsize=16, rotation=0)
plt.bar(y_pos,Ycount_values,align='center')

fig_num+=1
plt.figure(fig_num)
plt.xlabel('SeriousDlqin2yrs', fontsize=16)
plt.ylabel('debt_to_Limit', fontsize=16) 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
y_pos = [0,1]
label = ['0','1']
plt.xticks(y_pos, label, fontsize=16, rotation=0)
plt.bar(y_pos,DlqinGrouped[u'RevolvingUtilizationOfUnsecuredLines'],align='center')

fig_num+=1
plt.figure(fig_num)
plt.xlabel('SeriousDlqin2yrs', fontsize=16)
plt.ylabel('Age', fontsize=16) 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
y_pos = [0,1]
label = ['0','1']
plt.xticks(y_pos, label, fontsize=16, rotation=0)
plt.bar(y_pos,DlqinGrouped[u'age'],align='center')

fig_num+=1
plt.figure(fig_num)
plt.xlabel('SeriousDlqin2yrs', fontsize=16)
plt.ylabel('debt_to_income', fontsize=16) 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
y_pos = [0,1]
label = ['0','1']
plt.xticks(y_pos, label, fontsize=16, rotation=0)
plt.bar(y_pos,DlqinGrouped[u'DebtRatio'],align='center')

fig_num+=1
plt.figure(fig_num)
plt.xlabel('SeriousDlqin2yrs', fontsize=16)
plt.ylabel('Income', fontsize=16) 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
y_pos = [0,1]
label = ['0','1']
plt.xticks(y_pos, label, fontsize=16, rotation=0)
plt.bar(y_pos,DlqinGrouped[u'MonthlyIncome'],align='center')

fig_num+=1
plt.figure(fig_num)
plt.xlabel('SeriousDlqin2yrs', fontsize=16)
plt.ylabel('Number Of Dependents', fontsize=16) 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
y_pos = [0,1]
label = ['0','1']
plt.xticks(y_pos, label, fontsize=16, rotation=0)
plt.bar(y_pos,DlqinGrouped[u'NumberOfDependents'],align='center')

fig_num+=1
plt.figure(fig_num)
plt.xlabel('SeriousDlqin2yrs', fontsize=16)
plt.ylabel('Number Of Open Credit Lines', fontsize=16) 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
y_pos = [0,1]
label = ['0','1']
plt.xticks(y_pos, label, fontsize=16, rotation=0)
plt.bar(y_pos,DlqinGrouped[u'NumberOfOpenCreditLinesAndLoans'],align='center')

fig_num+=1
plt.figure(fig_num)
plt.xlabel('SeriousDlqin2yrs', fontsize=16)
plt.ylabel('Number of Real Estate Loans', fontsize=16) 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
y_pos = [0,1]
label = ['0','1']
plt.xticks(y_pos, label, fontsize=16, rotation=0)
plt.bar(y_pos,DlqinGrouped[u'NumberRealEstateLoansOrLines'],align='center')

## plots of delay (any range) of credit payment on whether on average there is any difference between Dlqin and not-Dlqin groups
fig_num+=1
plt.figure(fig_num)
plt.xlabel('SeriousDlqin2yrs', fontsize=16)
plt.ylabel('Number Of Time 30-59 Days Past Due Not Worse', fontsize=16) 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
y_pos = [0,1]
label = ['0','1']
plt.xticks(y_pos, label, fontsize=16, rotation=0)
plt.bar(y_pos,DlqinGrouped[u'NumberOfTime30-59DaysPastDueNotWorse'],align='center')

fig_num+=1
plt.figure(fig_num)
plt.xlabel('SeriousDlqin2yrs', fontsize=16)
plt.ylabel('Number Of Time 60-89 Days Past Due Not Worse', fontsize=16) 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
y_pos = [0,1]
label = ['0','1']
plt.xticks(y_pos, label, fontsize=16, rotation=0)
plt.bar(y_pos,DlqinGrouped[u'NumberOfTime60-89DaysPastDueNotWorse'],align='center')

fig_num+=1
plt.figure(fig_num)
plt.xlabel('SeriousDlqin2yrs', fontsize=16)
plt.ylabel('Number Of Times 90 Days Late', fontsize=16) 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
y_pos = [0,1]
label = ['0','1']
plt.xticks(y_pos, label, fontsize=16, rotation=0)
plt.bar(y_pos,DlqinGrouped[u'NumberOfTimes90DaysLate'],align='center')

#it seems that delays (either short=30-59 or long >90 days) have a great impact on defining a delinquent person
#so I define a new parameter that adds these three together:
Delay_days=train_df[u'NumberOfTime30-59DaysPastDueNotWorse']+train_df[u'NumberOfTime60-89DaysPastDueNotWorse']+train_df[u'NumberOfTimes90DaysLate']
Delay_days_test=test_df[u'NumberOfTime30-59DaysPastDueNotWorse']+test_df[u'NumberOfTime60-89DaysPastDueNotWorse']+test_df[u'NumberOfTimes90DaysLate']


# let's see how age distribution play a role in delinquency 
#print (train_df[u'age'].unique())
AgeGrouped_df = train_df.groupby('age')
train_df[u'age']=train_df.apply(func,axis=1)  #--> ages are aggregated together e.g. >=20 till <30 years old are replaced by 20 and so on
pd.crosstab(train_df[u'age'],train_df[u'SeriousDlqin2yrs']).plot(kind='bar')
label=['<20','20-30','30-40','40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '>100']
plt.xticks(range(0,len(label)), label, fontsize=12, rotation=0)
print ('length of data after grouping by "Age"=', len(AgeGrouped_df))


#______________________Logestic regression model_______________________________
#create list of features
train_X=train_df.drop(['SeriousDlqin2yrs','Unnamed: 0'],axis=1,inplace=False)#, 'NumberOfTime30-59DaysPastDueNotWorse','NumberOfTimes90DaysLate','NumberOfTime60-89DaysPastDueNotWorse'],axis=1,inplace=False)
test_X=test_df.drop(['SeriousDlqin2yrs','Unnamed: 0'],axis=1,inplace=False)#, 'NumberOfTime30-59DaysPastDueNotWorse','NumberOfTimes90DaysLate','NumberOfTime60-89DaysPastDueNotWorse'],axis=1,inplace=False)

train_X['NumberOfTimeDelays']=Delay_days
test_X['NumberOfTimeDelays']=Delay_days_test
train_X=train_X.dropna()
print (pd.isnull(train_X).sum(axis=0))
train_Y=train_df.dropna()[u'SeriousDlqin2yrs']
#because the test csv file did not have y_test I split the training data into 
#training and test to see how the model works and then used th model on test csv data for prediction
x_train, x_test, y_train, y_test = train_test_split(train_X, train_Y, test_size = 0.3, random_state = 0)


clf = LogisticRegression(random_state=0).fit(x_train, y_train)
#make predictions
predictions_train = clf.predict(x_train)
y_predict = clf.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(clf.score(x_test, y_test)))

tn, fp, fn, tp = confusion_matrix(y_test,y_predict).ravel()
print (tn, fp, fn, tp)
#create confusion matrix and plot
skplt.metrics.plot_confusion_matrix(y_test,y_predict,figsize=(8,8))

print(classification_report(y_test, y_predict))
print('f1_score:', sklearn.metrics.f1_score(y_test, y_predict) )
print('accuracy_score:', sklearn.metrics.accuracy_score(y_test, y_predict))
print('precision_score:', sklearn.metrics.precision_score(y_test, y_predict))
print('Recall:', sklearn.metrics.recall_score(y_test, y_predict))


test_X=test_X.dropna()
yPredict = clf.predict(test_X)


yout = pd.DataFrame({'Id':test_df.dropna(subset=['MonthlyIncome'])['Unnamed: 0'],'Binary result':yPredict})
yout.to_csv('Logestic_results.csv',index=False)

#___________________________________XGBooster Model____________________________
model = XGBClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
tn, fp, fn, tp = confusion_matrix(y_test,y_predict).ravel()
print (tn, fp, fn, tp)
#create confusion matrix and plot
skplt.metrics.plot_confusion_matrix(y_test,y_predict,figsize=(8,8))
print(classification_report(y_test, y_predict))
print('f1_score:', sklearn.metrics.f1_score(y_test, y_predict) )
print('accuracy_score:', sklearn.metrics.accuracy_score(y_test, y_predict))
print('precision_score:', sklearn.metrics.precision_score(y_test, y_predict))
print('Recall:', sklearn.metrics.recall_score(y_test, y_predict))

yPredict_XGBoost = model.predict(test_X)


yout = pd.DataFrame({'Id':test_df.dropna(subset=['MonthlyIncome'])['Unnamed: 0'],'Binary result':yPredict_XGBoost})
yout.to_csv('XGBoost_results.csv',index=False)



#plt.show()
'''debt_to_Limit=train_df[u'RevolvingUtilizationOfUnsecuredLines']
debt_to_income=train_df[u'DebtRatio']
age=train_df[u'age']
income=train_df[u'MonthlyIncome']
No_dependents=train_df[u'NumberOfDependents']
No_OpenCredit=train_df[u'NumberOfOpenCreditLinesAndLoans']
No_HomeMortgage=train_df[ u'NumberRealEstateLoansOrLines']
days_30_59=train_df[u'NumberOfTime30-59DaysPastDueNotWorse']
days_60_89=train_df[u'NumberOfTime60-89DaysPastDueNotWorse']
days_90more=train_df[u'NumberOfTimes90DaysLate']'''


'''fig_num=1
plt.figure(fig_num)
plt.xlabel('Customer ID', fontsize=26)
plt.ylabel('Revolving Utilization Of Unsecured Lines (%)', fontsize=26) 
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
y_pos = np.arange(len(train_df))
plt.bar(y_pos, debt_to_Limit, width=0.8)'''

'''grouped_df = train_df.groupby('NumberOfTime30-59DaysPastDueNotWorse')
dlinq_delay1 = grouped_df['SeriousDlqin2yrs'].aggregate([np.mean,'count']).reset_index()
print(dlinq_delay1)
dlinq_delay1.columns =['Delay30-59Days','DlqinFreq','count']
sns.regplot(x='Delay 30-59 Days',y='DlqinFreq',data=dlinq_delay1)
plt.show()

grouped_df = train_df.groupby('NumberOfTime60-89DaysPastDueNotWorse')
dlinq_delay2 = grouped_df['SeriousDlqin2yrs'].aggregate([np.mean,'count']).reset_index()
print(dlinq_delay2)
dlinq_delay2.columns =['Delay60-89Days','DlqinFreq','count']
sns.regplot(x='Delay 30-59 Days',y='DlqinFreq',data=dlinq_delay2)
plt.show()

grouped_df = train_df.groupby('Number Of Times 90 Days Late)
dlinq_delay3 = grouped_df['SeriousDlqin2yrs'].aggregate([np.mean,'count']).reset_index()
print(dlinq_delay3)
dlinq_delay3.columns =['Delay90Days','DlqinFreq','count']
sns.regplot(x='Delay 30-59 Days',y='DlqinFreq',data=dlinq_delay3)'''

