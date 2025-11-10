from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, RepeatedKFold, LeaveOneOut, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

alphas = [0.01, 0.1, 1, 10, 100, 1000]

# as we already did preprocessing no need to define it
ridge_pipeline = Pipeline([
    ("model", Ridge())
])

# 5-Fold Cross-Validation
cv5 = KFold(
    n_splits = 5,
    shuffle = True,
    random_state = 123
)

param_grid = {"model__alpha": alphas}

# GridSearchCV
ridge_grid_search = GridSearchCV(
    ridge_pipeline,
    param_grid = param_grid,
    cv = cv5,
    scoring = "neg_root_mean_squared_error",
    n_jobs = -1
)

ridge_grid_search.fit(X_train, y_train)

model_ridge_best = ridge_grid_search.best_estimator_

print("Best alpha:", ridge_grid_search.best_params_)
print("Best -RMSE:", ridge_grid_search.best_score_)
