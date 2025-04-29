# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 14:30:06 2025

@author: bswan work
"""

"###WNBA fun """


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
df = pd.read_csv("C:\\Users\\bswan work\\Downloads\\WNBA-stats-master\\WNBA-stats-master\\wnba-team-elo-ratings.csv")
"""df.rename(columns={"Date": "date", "Numbers": "check_ins"}, inplace=True)"""

df2 = pd.read_csv("C:\\Users\\bswan work\\Downloads\\WNBA-stats-master\\WNBA-stats-master\\wnbaoffv1.csv")

"""
df2 = df2.merge(df[['prob1','elo1_pre','elo2_pre']], 
              left_on=['date', 'team1', 'team2'], right_on=['date', 'team1', 'team2'], 
              how='left')
"""

df['point_diff'] = df.score1 - df.score2
df['team1win'] = np.where(df['point_diff'] > 0, 1, 0)

"""if df['point_diff'] > 0 :
    df['team1win'] = 1 
else: df['team1win'] = 0"""

"logit model base model"


train = df.copy()
test = df.copy()
train = train.loc[train['season'] < 2016]
test = test.loc[test['season'] > 2016]

X = test[['prob1','elo1_pre','elo2_pre']]
y = test['team1win']

X_train = train[['prob1','elo1_pre','elo2_pre']]
y_train = train['team1win']
X_test = test[['prob1','elo1_pre','elo2_pre']]
y_test = test['team1win']


model = LogisticRegression(solver='liblinear', random_state=0)
result = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predicted = model.predict_proba(X_test)[:,1]
"print(y_pred.summary())"

"Wierdly the below gives you a summary, what you had before would not"
logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary())


"""  This is with your new measure of offense
"""
df2['point_diff'] = df2.score1 - df2.score2
df2['team1win'] = np.where(df2['point_diff'] > 0, 1, 0)

train = df2.copy()
test = df2.copy()
train = train.loc[train['season'] < 2016]
test = test.loc[test['season'] > 2016]

X = test[['prob1','elo1_pre','elo2_pre','team1_offense_metric','team2_offense_metric']]
y = test['team1win']

X_train = train[['prob1','elo1_pre','elo2_pre','team1_offense_metric','team2_offense_metric']]
y_train = train['team1win']
X_test = test[['prob1','elo1_pre','elo2_pre','team1_offense_metric','team2_offense_metric']]
y_test = test['team1win']


model = LogisticRegression(solver='liblinear', random_state=0)
result = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predicted = model.predict_proba(X_test)[:,1]

logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary())

"""
y_pred = model.predict(X_test)
model.summary(y_pred)
model.predict_proba(y_pred)
# Split the data"""
"""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"""
"""
# Create and train the logistic regression model
logit_model = LogisticRegression(random_state=42)
logit_model.fit(X_train, y_train)

# Get probabilities for the test set
test_probabilities = logit_model.predict_proba(X_test)[:, 1]

# Add probabilities to the test DataFrame
test.loc[X_test.index, 'bet_probability'] = test_probabilities

# Calculate the model's accuracy
accuracy = logit_model.score(X_test, y_test)
print(f"Logistic Regression Model Accuracy: {accuracy:.4f}")
"""


