from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, RepeatedKFold, LeaveOneOut, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

alphas = [0.01, 0.1, 1, 10]
l1_ratios = [0.1, 0.5, 0.9]

elastic_pipeline = Pipeline(
    [
        # ('scaler', StandardScaler()), # z-score scaling
        ('model', ElasticNet(max_iter=2000, tol=0.01))
    ]

)

param_grid = {
    'model__alpha': alphas,
    'model__l1_ratio': l1_ratios
}

elastic_grid_search = GridSearchCV(
    estimator = elastic_pipeline,
    param_grid = param_grid,
    scoring = 'neg_mean_absolute_error',
    cv = cv5,
    n_jobs = -1
)

elastic_grid_search.fit(X_train, y_train)

model_elastic_best = elastic_grid_search.best_estimator_

print('Best alpha:', elastic_grid_search.best_params_['model__alpha'])
print('Best l1_ratio:', elastic_grid_search.best_params_['model__l1_ratio'])
print('Best (negative) MAE:', elastic_grid_search.best_score_)