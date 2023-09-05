
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler


# > # **2. Data Preprocessing** 
#Print the dataframe
#Dataset link : "https://www.kaggle.com/datasets/ron2112/mental-health-data"
url = "https://www.kaggle.com/datasets/ron2112/mental-health-data"
data=pd.read_csv(url)
data.head(10)

# Information of dataframe
data.info()

#Check the Shape of dataset
print(data.shape)


#Make the list of columns
a=list(data.columns)
print(a)
# New name of the all columns
b=['self_employed',
   'no_of_employees',
   'tech_company','role_IT',
   'mental_healthcare_coverage',
   'knowledge_about_mental_healthcare_options_workplace',
   'employer_discussed_mental_health ',
   'employer_offer_resources_to_learn_about_mental_health',
   'medical_leave_from_work ',
  'comfortable_discussing_with_coworkers',
   'employer_take_mental_health_seriously',
   'knowledge_of_local_online_resources ',
   'productivity_affected_by_mental_health ',
   'percentage_work_time_affected_mental_health',
   'openess_of_family_friends',
  'family_history_mental_illness',
   'mental_health_disorder_past',
   'currently_mental_health_disorder',
   'diagnosed_mental_health_condition',
   'type_of_disorder',
   'treatment_from_professional',
   'while_effective_treatment_mental_health_issue_interferes_work',
   'while_not_effective_treatment_interferes_work ',
   'age',
   'gender',
   'country',
   'US state',
  'country work ',
   'US state work',
   'role_in_company',
   'work_remotely','']
for i,j in zip(a,b):
    data.rename(columns={i:j},inplace=True)
# Information of dataframe after the rename
data.info()


## Now We Find the Missing values in different Columns
columns=data.columns
pd.DataFrame({'no of missing values':data.isnull().sum()})


# Now we copy the dataset in data1
data1=data.copy()
data1


# Now there are sum columns which has so many tuple are not have any value so it is unnecessary columns for us so we can remove it using  drop. 
remove_columns = ['role_IT',
                  'knowledge_of_local_online_resources ',
        'productivity_affected_by_mental_health ',
        'percentage_work_time_affected_mental_health']
data2=data1.drop(remove_columns,axis=1)
data2.shape


# > # **Cleaning Different Columns**

# No of employee column
print(data2.no_of_employees.unique())
data2.no_of_employees.unique()

# change the value format
data2.no_of_employees.replace(to_replace=['1 to 5', '6 to 25','More than 1000','26-99'],
                                value=['1-5','6-25','>1000','26-100'],inplace=True)

print(data2.no_of_employees.value_counts())




# Cleaning Mental Health Care coverage column
data2.mental_healthcare_coverage.unique()

data2.mental_healthcare_coverage.replace(to_replace=['Not eligible for coverage / N/A'],
                                value='No',inplace=True)
print(data2.mental_healthcare_coverage.unique())
print(data2.mental_healthcare_coverage.value_counts())


# openess_of_family_friends column
data2.openess_of_family_friends.unique()


data2.openess_of_family_friends.replace(to_replace=['Not applicable to me (I do not have a mental illness)'],
                                        value="I don't know",inplace=True)
data2.openess_of_family_friends.unique()


print(data2.openess_of_family_friends.value_counts())


# Cleaning the age column remove outliers
med_age = data2[(data2['age'] >= 18) | (data2['age'] <= 75)]['age'].median()
print(med_age)
data2['age'].replace(to_replace = data2[(data2['age'] < 18) | (data2['age'] > 75)]['age'].tolist(),
                          value = med_age, inplace = True)
data2.age.unique()


# gender column
data2.gender.unique()


data2['gender'].replace(to_replace = ['Male', 'male', 'Male ', 'M', 'm',
       'man', 'Cis male', 'Male.', 'male 9:1 female, roughly', 'Male (cis)', 'Man', 'Sex is male',
       'cis male', 'Malr', 'Dude', "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? ",
       'mail', 'M|', 'Male/genderqueer', 'male ',
       'Cis Male', 'Male (trans, FtM)',
       'cisdude', 'cis man', 'MALE'], value = 'male', inplace = True)
data2['gender'].replace(to_replace = ['Female', 'female', 'I identify as female.', 'female ',
       'Female assigned at birth ', 'F', 'Woman', 'fm', 'f', 'Cis female ', 'Transitioned, M2F',
       'Genderfluid (born female)', 'Female or Multi-Gender Femme', 'Female ', 'woman', 'female/woman',
       'Cisgender Female', 'fem', 'Female (props for making this a freeform field, though)',
       ' Female', 'Cis-woman', 'female-bodied; no feelings about gender',
       'AFAB'], value = 'female', inplace = True)
data2['gender'].replace(to_replace = ['Bigender', 'non-binary', 'Other/Transfeminine',
       'Androgynous', 'Other', 'nb masculine',
       'none of your business', 'genderqueer', 'Human', 'Genderfluid',
       'Enby', 'genderqueer woman', 'mtf', 'Queer', 'Agender', 'Fluid',
       'Nonbinary', 'human', 'Unicorn', 'Genderqueer',
       'Genderflux demi-girl', 'Transgender woman'], value = 'other', inplace = True)




data2.gender.unique()


data2.gender.value_counts()



## Cleaning the role_in_company
tech_list = []
tech_list.append(data2[data2['role_in_company'].str.contains('Back-end')]['role_in_company'].tolist())
tech_list.append(data2[data2['role_in_company'].str.contains('Front-end')]['role_in_company'].tolist())
tech_list.append(data2[data2['role_in_company'].str.contains('Dev')]['role_in_company'].tolist())
tech_list.append(data2[data2['role_in_company'].str.contains('DevOps')]['role_in_company'].tolist())
flat_list = [item for sublist in tech_list for item in sublist]
flat_list = list(dict.fromkeys(flat_list))


## Replace tech role=1 and other=0 in a new tech role operation
data2['tech_role']=data2['role_in_company']
data2['tech_role'].replace(to_replace=flat_list,value=1,inplace=True)
remain_list=data2['tech_role'].unique()[1:]
data2['tech_role'].replace(to_replace=remain_list,value=0,inplace=True)



data2.tech_role.value_counts()


data2=data2.drop(['role_in_company'],axis=1)


# > # **Handling Missing values**

data3=pd.concat([data2['type_of_disorder'],data2['US state'],data2['US state work']],axis=1)
print(data3.info())
data2=data2.drop(['type_of_disorder','US state','US state work'],axis=1)


data2.info()


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp.fit(data2)
imp_data=pd.DataFrame(data=imp.transform(data2),columns=data2.columns)


data4=pd.concat([imp_data,data3],axis=1)
data4.isnull().sum().to_frame()


data4




print(data4.shape)




data4.info()


# > # **Data Preperation*
data4.shape


# Here We Dropping unnecessary columns
y=data4.diagnosed_mental_health_condition
x=data4.drop(['diagnosed_mental_health_condition','treatment_from_professional','while_effective_treatment_mental_health_issue_interferes_work','while_not_effective_treatment_interferes_work ','type_of_disorder','US state','US state work'],axis=1)

print(x.shape)
print(y.shape)

# Splitting the data
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


cat_columns=['self_employed', 
             'no_of_employees', 
             'tech_company',
             'mental_healthcare_coverage',
             'knowledge_about_mental_healthcare_options_workplace',
             'employer_discussed_mental_health ',
             'employer_offer_resources_to_learn_about_mental_health',
             'medical_leave_from_work ',  
             'comfortable_discussing_with_coworkers',
             'employer_take_mental_health_seriously', 
             'openess_of_family_friends',
             'family_history_mental_illness', 
             'mental_health_disorder_past',
             'currently_mental_health_disorder', 
             'age', 
             'gender', 
             'country',
             'country work ', 
             'work_remotely', 
             'tech_role']

print(data4['diagnosed_mental_health_condition'].unique())


for col in cat_columns:
  print('The Unique value',col,'is')
  print(data4[col].unique())
  print()

from sklearn.preprocessing import LabelEncoder
import numpy as np


class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)

label_encode=LabelEncoderExt()

label_x_train=x_train.copy()
label_x_test=x_test.copy()

for col in cat_columns:
    label_x_train[col]=label_encode.fit(x_train[col])
    label_encode.classes_
    label_x_train[col]=label_encode.transform(x_train[col])
    label_x_test[col] = label_encode.transform(label_x_test[col])


label_x_train

label_x_test
df = pd.DataFrame(label_x_test)

for col in cat_columns:
  print('The Unique value',col,'is')
  print(df[col].unique())
  #print(type(df["Subjects"].unique()))

type(label_x_test)

# For Y label Encode
label_encode_1=LabelEncoder()
label_y_train_1=label_encode_1.fit_transform(y_train)
label_y_test_1=label_encode_1.transform(y_test)


st=pd.DataFrame(label_y_train_1)
print(st)


st=pd.DataFrame(label_y_test_1)
print(st)


# > # **1. Logistic Regression** 

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logistic=LogisticRegression(C=1,penalty='l1',solver='liblinear',random_state=0)

logistic.fit(label_x_train,label_y_train_1)
preds3=logistic.predict(label_x_test)
accuracy_score(label_y_test_1,preds3)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

results = confusion_matrix(label_y_test_1,preds3)
print ('Confusion Matrix :')
print(results)
print ('Accuracy Score is',accuracy_score(label_y_test_1,preds3))
print ('Classification Report : ')
print (classification_report(label_y_test_1,preds3))
print('AUC-ROC:',roc_auc_score(label_y_test_1,preds3))
print('LOGLOSS Value is',log_loss(label_y_test_1,preds3))


# > # **2. Decision Tree** 
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(label_x_train,label_y_train_1)
y_pred = clf.predict(label_x_test)
accuracy_score(label_y_test_1,y_pred)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

results = confusion_matrix(label_y_test_1,y_pred)
print ('Confusion Matrix :')
print(results)
print ('Accuracy Score is',accuracy_score(label_y_test_1,y_pred))
print ('Classification Report : ')
print (classification_report(label_y_test_1,y_pred))
print('AUC-ROC:',roc_auc_score(label_y_test_1,y_pred))
print('LOGLOSS Value is',log_loss(label_y_test_1,y_pred))


#3Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
model=RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
model.fit(label_x_train,label_y_train_1)
preds=model.predict(label_x_test)
accuracy_score(label_y_test_1,preds)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import seaborn as sns

results = confusion_matrix(label_y_test_1,preds)
print ('Confusion Matrix :')
print(results)
print ('Accuracy Score is',accuracy_score(label_y_test_1,preds))
print ('Classification Report : ')
print (classification_report(label_y_test_1,preds))
print('AUC-ROC:',roc_auc_score(label_y_test_1,preds))
print('LOGLOSS Value is',log_loss(label_y_test_1,preds))

# Generate confusion matrix plot
sns.set(font_scale=1.4)
sns.heatmap(results, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g')

plt.show()


# > # **4. KNN** 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(label_x_train)
label_x_train = scaler.transform(label_x_train) 
label_x_test = scaler.transform(label_x_test)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(label_x_train, label_y_train_1) 

y_pred1 = classifier.predict(label_x_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

results = confusion_matrix(label_y_test_1,y_pred1)
print ('Confusion Matrix :')
print(results)
print ('Accuracy Score is',accuracy_score(label_y_test_1,y_pred1))
print ('Classification Report : ')
print (classification_report(label_y_test_1,y_pred1))
print('AUC-ROC:',roc_auc_score(label_y_test_1,y_pred1))
print('LOGLOSS Value is',log_loss(label_y_test_1,y_pred1))

