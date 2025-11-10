from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, RepeatedKFold, LeaveOneOut, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

# SVR pipeline
svr_pipeline = Pipeline([
    # ('scaler', StandardScaler()),
    ('model', SVR())
])

# # Hyperparameter grid
# param_grid = {
#     # model__C : 100 was computationally expensive so changing to 50
#     'model__C': [0.1, 1, 10, 50],
#     'model__epsilon': [0.01, 0.1, 0.5, 1],
#     'model__gamma': ['scale', 'auto'],  # or numeric values like 0.01, 0.1
#     'model__kernel': ['rbf']  # you can also try 'linear', 'poly', etc.
# }

# previous version was time consuming
param_grid = {
    'model__C': [0.1, 1, 10],
    'model__epsilon': [0.01, 0.1, 0.5],
    'model__gamma': ['scale'],
    'model__kernel': ['rbf']
}

# GridSearchCV
svr_grid_search = GridSearchCV(
    estimator=svr_pipeline,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=cv5,
    n_jobs=-1
)

# Fit
svr_grid_search.fit(X_train, y_train)

# Best model
model_svr_best = svr_grid_search.best_estimator_

print('Best C:', svr_grid_search.best_params_['model__C'])
print('Best epsilon:', svr_grid_search.best_params_['model__epsilon'])
print('Best gamma:', svr_grid_search.best_params_['model__gamma'])
print('Best (negative) MAE:', svr_grid_search.best_score_)


#! 2


# SVR with GridSearchCV took soo much time

svr_pipeline = Pipeline([
    # ('scaler', StandardScaler()),
    ('model', SVR())
])

param_dist = {
    'model__C': [50],
    'model__epsilon': [0.01],
    'model__gamma': [0.01],  # keep only scale for speed
    'model__kernel': ['rbf']
}

svr_random_search = RandomizedSearchCV(
    estimator=svr_pipeline,
    param_distributions=param_dist,
    n_iter=5,  # try 10 random combinations
    scoring='neg_mean_absolute_error',
    cv=3,       # fewer folds for speed
    n_jobs=-1,
    random_state=123
)

svr_random_search.fit(X_train, y_train)

model_svr_best = svr_random_search.best_estimator_
print(svr_random_search.best_params_)
print(svr_random_search.best_score_)

#! 3

svr_pipeline = Pipeline([
    # ('scaler', StandardScaler()),  # optional if X already scaled
    ('model', SVR(C=1.0, epsilon=0.1, kernel='rbf', gamma='scale', max_iter=2000))
])
ß
# Fit the model
svr_pipeline.fit(X_train, y_train)