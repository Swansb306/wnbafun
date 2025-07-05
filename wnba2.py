# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 16:09:45 2025
pip install pandas numpy matplotlib scikit-learn statsmodels
@author: bswan work
""" 
'This project is to try to predict winners of wnba games.'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
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

"lets do some descriptive statistics"
import seaborn as sns
print(df.describe())
df.value_counts()


_ = df.hist(figsize=(20, 14))
df.isnull().sum()


"start modeling"
train = df.copy()
test = df.copy()
train = train.loc[train['season'] < 2016]
test = test.loc[test['season'] > 2016]

X = test[['elo1_pre','elo2_pre']]
y = test['team1win']

X_train = train[['elo1_pre','elo2_pre']]
y_train = train['team1win']
X_test = test[['elo1_pre','elo2_pre']]
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

"some quick descriptive stats"
_ = df2.hist(figsize=(20, 14))
df2.isnull().sum()
print(df2.describe())
print("value counts")
df2.value_counts()

"""
x = sns.scatterplot(
    x="elo1_pre",
    y="team1_offense_metric",
    data=df2,
    alpha=0.5,
)

plt.show(x)"""

train = df2.copy()
test = df2.copy()
train = train.loc[train['season'] < 2016]
test = test.loc[test['season'] > 2016]

X = test[['elo1_pre','elo2_pre','team1_offense_metric','team2_offense_metric']]
y = test['team1win']

X_train = train[['elo1_pre','elo2_pre','team1_offense_metric','team2_offense_metric']]
y_train = train['team1win']
X_test = test[['elo1_pre','elo2_pre','team1_offense_metric','team2_offense_metric']]
y_test = test['team1win']

'this is the model in sklearn'

model = LogisticRegression(solver='liblinear', random_state=0)
result = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predicted = model.predict_proba(X_test)[:,1]
accuracy = model.score(X_test, y_test)
print(f"Accuracy of logistic regression: {accuracy:.3f}")


print(result)
'this is an attempt to get a summary table in sklearn'
summary = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_[0],
    'Odds Ratio': np.exp(model.coef_[0])
})

# Add Intercept (optional)
intercept = pd.DataFrame({
    'Feature': ['Intercept'],
    'Coefficient': [model.intercept_[0]],
    'Odds Ratio': [np.exp(model.intercept_[0])]
})

summary = pd.concat([intercept, summary], ignore_index=True)
print(summary)


"this is trying to run the model using sm"
X_train2 = sm.add_constant(X_train)
X_test2 = sm.add_constant(X_test)

sm_model = sm.Logit(y_train, X_train2)
sm_result = sm_model.fit()
print(sm_result.summary())

prediction1 = sm_result.predict(X_test2)



"""df2024 = pd.read_csv("C:\\Users\\bswan work\\Documents\\python quant project\\Python codes\\Python codes\\wnba2024.csv", header = 1)"""
df2024 = pd.read_csv("C:\\Users\\bswan work\\Documents\\python quant project\\Python codes\\Python codes\\wnba2024dvoa2.csv")


"""df2024.rename(columns={"PTS": "score1", "PTS.2": "score2"}, inplace=True)
"""
df2024['point_diff'] = df2024.score1 - df2024.score2
df2024['team1win'] = np.where(df2024['point_diff'] > 0, 1, 0)
newy = df2024['team1win']
newx = df2024[['team1_offense_metric','team2_offense_metric']]

'given that elo had such a small effect relative to the offensive metric I am cutting it'
X_train = train[['team1_offense_metric','team2_offense_metric']]
y_train = train['team1win']
X_test = test[['team1_offense_metric','team2_offense_metric']]
y_test = test['team1win']

model2024 = LogisticRegression(solver='liblinear', random_state=0)
result = model2024.fit(X_train, y_train)
pred2024 = model2024.predict(newx)
predicted = model2024.predict_proba(newx)[:,1]
accuracy = model2024.score(newx, newy)
print(f"Accuracy of logistic regression: {accuracy:.3f}")

'okay time for 2025 data!'

df2025 = pd.read_csv("C:\\Users\\bswan work\\Documents\\python quant project\\Python codes\\Python codes\\wnba2025withoffense_june11.csv")


df2025['point_diff'] = df2025.score1 - df2025.score2
df2025['team1win'] = np.where(df2025['point_diff'] > 0, 1, 0)
newy = df2025['team1win']
newx = df2025[['team1_offense_metric','team2_offense_metric']]

'given that elo had such a small effect relative to the offensive metric I am cutting it'
X_train = train[['team1_offense_metric','team2_offense_metric']]
y_train = train['team1win']
X_test = test[['team1_offense_metric','team2_offense_metric']]
y_test = test['team1win']

model2025 = LogisticRegression(solver='liblinear', random_state=0)
result = model2025.fit(X_train, y_train)
pred2025 = model2025.predict(newx)
predicted = model2025.predict_proba(newx)[:,1]
accuracy = model2025.score(newx, newy)
print(f"Accuracy of logistic regression: {accuracy:.3f}")

"week 11"
df2025week11 = pd.read_csv("C:\\Users\\bswan work\\Documents\\python quant project\\Python codes\\Python codes\\2025forwardpredictions\\weekofjune11.csv")



"newy = df2025week11['team1win']"
newx = df2025week11[['team1_offense_metric','team2_offense_metric']]


X_train = train[['team1_offense_metric','team2_offense_metric']]
y_train = train['team1win']
X_test = test[['team1_offense_metric','team2_offense_metric']]
y_test = test['team1win']

model2025w11 = LogisticRegression(solver='liblinear', random_state=0)
resultw11 = model2025w11.fit(X_train, y_train)
pred2025w11 = model2025w11.predict(newx)

""" these are commented out because we can't know the answer yet!
predicted = model2025w11.predict_proba(newx)[:,1]
accuracy = model2025w11.score(newx, newy)
print(f"Accuracy of logistic regression: {accuracy:.3f}")

"""

onlynumer = df2[['point_diff','team2_points_against', 'team1_offense_metric', 'team2_offense_metric','elo1_pre', 'elo2_pre']]

import seaborn as sns
target_column = int(df2["team1win"])

n_samples_to_plot = 10488

"""columns = ["age", "education-num", "hours-per-week"]
_ = sns.pairplot(
    data=df2[:n_samples_to_plot],
    vars=columns,
    hue=target_column,
    plot_kws={"alpha": 0.2},
    height=3,
    diag_kind="hist",
    diag_kws={"bins": 30},
)"""

_ = sns.scatterplot(
    x="elo1_pre",
    y="team1_offense_metric",
    data=df2[:n_samples_to_plot],
    hue=target_column,
    alpha=0.5,
)
dfnum = df2[["team1_points_against",	"team2_points_against",	"team1_offense_metric",	"team2_offense_metric",	"score1",	"score2",	"prob1",	"elo1_pre",	"elo2_pre",	"point_diff",	"team1win"]]
correlationmatrix1 = dfnum.corr()
print(correlationmatrix1)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(correlationmatrix1, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeRegressor





target = df2['team1win']

tree = DecisionTreeRegressor(random_state=0)
cv_results = cross_validate(tree, onlynumer, target, n_jobs=2)
scores = cv_results["test_score"]

print(
    "R2 score obtained by cross-validation: "
    f"{scores.mean():.3f} ± {scores.std():.3f}"
)

from sklearn.ensemble import BaggingRegressor

estimator = DecisionTreeRegressor(random_state=0)
bagging_regressor = BaggingRegressor(
    estimator=estimator, n_estimators=20, random_state=0
)

cv_results = cross_validate(bagging_regressor, onlynumer, target, n_jobs=2)
scores = cv_results["test_score"]

print(
    "R2 score obtained by cross-validation: "
    f"{scores.mean():.3f} ± {scores.std():.3f}"
)
"""
print(
    "R2 score obtained by cross-validation: "
    f"{scores.mean():.3f} ± {scores.std():.3f}"
"need to split between test and train!"
sns.scatterplot(
    x=data_train["Feature"], y=target, color="black", alpha=0.5
)
plt.plot(data_test["Feature"], y_pred, label="Fitted tree")
plt.legend()
_ = plt.title("Predictions by a single decision tree")
)
"""



"need to split into training and testing!"

from sklearn.model_selection import train_test_split
"""numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]
data[numerical_columns]
data_numeric = data[numerical_columns]

"""
onlynumer.describe()
target_column.describe()

data_train, data_test, target_train, target_test = train_test_split(
    onlynumer, target_column, random_state=42, test_size=0.25
)

cv_results = cross_validate(bagging_regressor, data_train, target_train, n_jobs=2)
scores = cv_results["test_score"]

scores.describe()

print(
    "R2 score obtained by cross-validation: "
    f"{scores.mean():.3f} ± {scores.std():.3f}"
)

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

bagging = BaggingRegressor(
    estimator=polynomial_regressor,
    n_estimators=100,
    random_state=0,
)
_ = bagging.fit(data_train, target_train)


sns.scatterplot(
    x=data_train["Feature"], y=target_train, color="black", alpha=0.5
)

bagged_trees_predictions = bagged_trees.predict(data_test)
plt.plot(
    data_test["Feature"],
    bagged_trees_predictions,
    color="tab:orange",
    label="Predictions of ensemble",
)
_ = plt.legend()
