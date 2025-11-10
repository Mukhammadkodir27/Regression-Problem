from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, RepeatedKFold, LeaveOneOut, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

alphas = np.logspace(-3, 4, 50)

lasso_pipeline = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('model', Lasso())
    ]
)

cv5 = KFold(
    n_splits = 5,
    shuffle = True,
    random_state = 123
)


param_grid = {
    'model__alpha': alphas
}

lasso_grid_search = GridSearchCV(
    estimator = lasso_pipeline,
    param_grid = param_grid,
    scoring = 'neg_mean_absolute_error', # could be R2 too
    cv = cv5,
    n_jobs = -1
)

lasso_grid_search.fit(X_train, y_train)

model_lasso_best = lasso_grid_search.best_estimator_

print('Best alpha:', lasso_grid_search.best_params_)
print('Best (negative) MAE:', lasso_grid_search.best_score_)
