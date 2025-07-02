
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(random_state=0)
cv_results = cross_validate(tree, data, target, n_jobs=2)
scores = cv_results["test_score"]

print(
    "R2 score obtained by cross-validation: "
    f"{scores.mean():.3f} ± {scores.std():.3f}"
)
"lets try to create a function that runs and displays ensemble"
" model results that we can import into other file"


from sklearn.ensemble import BaggingRegressor

estimator = DecisionTreeRegressor(random_state=0)
bagging_regressor = BaggingRegressor(
    estimator=estimator, n_estimators=20, random_state=0
)

cv_results = cross_validate(bagging_regressor, data, target, n_jobs=2)
scores = cv_results["test_score"]

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
