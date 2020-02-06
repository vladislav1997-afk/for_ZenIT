import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import gc
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb

data = pd.read_csv('creditcard.csv')
print('Credit Card Fraud Detection Data -- rows:', data.shape[0], \
      'columns:', data.shape[1])
data.head()
data.describe()
data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending=False)
count_classes = pd.value_counts(data['Class'], sort=True)
count_classes.plot(kind='bar', rot=0)
plt.title("Transactions distribution")
plt.xticks(range(2))
plt.xlabel("Class")
plt.ylabel("Frequency")
class_0 = data.loc[data['Class'] == 0]['Time']
class_1 = data.loc[data['Class'] == 1]['Time']
hist_data = [class_0, class_1]
group_labels = ['Not Fraud', 'Fraud']

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
fig['layout'].update(title='Credit Card Transactions Time Density Plot', xaxis=dict(title='Time [Secds]'))
plot(fig, filename='dist_only')
target = 'Class'
predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
              'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
# train/validation/test split
valid_size = 0.2  # simple validation
test_size = 0.2
kfolds = 5  # number of KFolds for cross-validation
random_state = 2019
train_d, test_d = train_test_split(data, test_size=0.2, random_state=2019, shuffle=True)
train_d, valid_d = train_test_split(train_d, test_size=0.2, random_state=2019, shuffle=True)
from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression(random_state=2019, solver='liblinear')
param_grid = {'C': [0.1, 1, 10, 20], 'penalty': ['l1', 'l2']}
grid_search_lr = GridSearchCV(lgr, param_grid=param_grid, scoring='recall', cv=5)
grid_search_lr.fit(train_d[predictors], train_d[target].values)
print('The best recall scores:', grid_search_lr.best_score_, 'Best parameter for trainning set:',
      grid_search_lr.best_params_)

# Set models
lgr = LogisticRegression(random_state=2019, penalty='l2', C=1, solver='liblinear')
lgr.fit(train_d[predictors], train_d[target].values)
preds_t = lgr.predict(test_d[predictors])
roc_auc_score(test_d[target].values, preds_t)
target_names = ['class 0', 'class 1']
print(classification_report(test_d[target].values, preds_t, target_names=target_names))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(lgr, train_d[predictors], train_d[target].values, cv=5, scoring='roc_auc')
cross_val_score(lgr, test_d[predictors], test_d[target].values, cv=5, scoring='roc_auc')
RFC_METRIC = 'gini'  # validation criterion, metric used for RandomForrestClassifier
N_ESTIMATORS = 100  # number of estimators/trees used for RandomForrestClassifier
N_JOBS = 4  # number of parallel jobs used for RandomForrestClassifier

clf = RandomForestClassifier(n_jobs=4, random_state=2019, criterion='gini', n_estimators=100, verbose=False)
clf.fit(train_d[predictors], train_d[target].values)
preds = clf.predict(valid_d[predictors])
preds_t = clf.predict(test_d[predictors])

tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance', ascending=False)
plt.figure(figsize=(7, 4))
plt.title('Featureortance', fontsize=14)
s = sns.barplot(x='Feature', y='Feature importance', data=tmp)
s.set_xticklabels(s.get_xticklabels(), rotation=90)
plt.show()
