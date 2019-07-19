import pandas as pd
import tensorflow as tf
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

X_train = pd.read_csv("D:/KDD99/kddcup.data_10_percent_corrected", sep=",", header=None,
                      names=['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
                             'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell',
                             'su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds',
                             'is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate',
                             'srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
                             'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
                             'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','category'])

Y_train = X_train.iloc[:,41]

#Convert datatype int64 and float64 to decrease memory usage
con_int = X_train.select_dtypes(include=['int64']).apply(pd.to_numeric,downcast='unsigned')
con_float = X_train.select_dtypes(include=['float64']).apply(pd.to_numeric,downcast='float')
X_train[con_int.columns] = con_int
X_train[con_float.columns] = con_float

#'''

#X_train = X_train.iloc[:,[4,5]]
#Y_test = X_test.iloc[:,41]

X_train.drop(['category'],axis=1,inplace=True)

#Convert object datatype using dummies
df_dummies_1 = pd.get_dummies(X_train['protocol_type'],prefix='protocol_type')
df_dummies_2 = pd.get_dummies(X_train['service'],prefix='service')
df_dummies_3 = pd.get_dummies(X_train['flag'],prefix='flag')
df_dummies = pd.concat([df_dummies_1,df_dummies_2,df_dummies_3], axis = 1)
X_train = pd.concat([X_train,df_dummies],axis=1)
X_train.drop(['protocol_type','service','flag'],axis = 1, inplace = True)

#Delete unused variable
lst = [df_dummies,df_dummies_1,df_dummies_2,df_dummies_3]
del lst

#Use scaling to make sure features are scaled from -1 to 1
X_scaled = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = X_scaled.transform(X_train)

#Split training and testing dataset
X_train_new,X_test_new,Y_train_new,Y_test_new = train_test_split(X_train, Y_train, test_size=0.2)

model_1 = svm.SVC(kernel='linear', C=1 ,verbose= True).fit(X_train_new,Y_train_new)
model_2 = DecisionTreeClassifier(max_depth=10).fit(X_train_new,Y_train_new)
model_3 = RandomForestClassifier(n_estimators=100,max_depth=10).fit(X_train_new,Y_train_new)

Y_pred_1 = model_1.predict(X_test_new)
Y_pred_2 = model_2.predict(X_test_new)
Y_pred_3 = model_3.predict(X_test_new)

print("Accuracy for SVM: ",accuracy_score(Y_test_new,Y_pred_1))
print("Accuracy for DT: ",accuracy_score(Y_test_new,Y_pred_2))
print("Accuracy for RF: ",accuracy_score(Y_test_new,Y_pred_3))

print(classification_report(Y_test_new,Y_pred_1))
print(classification_report(Y_test_new,Y_pred_2))
print(classification_report(Y_test_new,Y_pred_3))