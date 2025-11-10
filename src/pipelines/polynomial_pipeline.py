import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, RepeatedKFold, LeaveOneOut, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge, Lasso, ElasticNet


cv5 = KFold(
    n_splits = 5,
    shuffle = True,
    random_state = 123
)

polynomial_pipeline = Pipeline(
    [
        ('generator', PolynomialFeatures()),
        ('model', LinearRegression())
    ]
)

# degrees = list(range(1, 5))
degrees = [1, 2, 3, 4, 5]

# we need to define a grid of hyperparameters for the PolynomailFeatures() transformer
# it has to be a dictionary with the name of the step and the name of the hyperparameter
# as defined in the pipeline and algorithm documentation

# ex: here we use the name "generator__degree" because
# our pipeline step is named "generator" and the hyperparameter of the 
# PolynomialFeatures() transformer is named "degree"
polynomial_grid = {'generator__degree': degrees}

# define a GridSearchCV object to find the optimal degree
polynomial_grid_search = GridSearchCV(
    polynomial_pipeline, # model or pipeline to tune
    param_grid = polynomial_grid, # dictionary with hyperparameters
    cv = cv5, # cross-validation 
    # python assumes the higher the better, hence we use negative RMSE
    scoring = 'neg_root_mean_squared_error', # evaluation metric
    n_jobs = -1
)

polynomial_grid_search.fit(X_train, y_train)

# adding "_best" to GridSearchCV result
model_polynomial_best = polynomial_grid_search.best_estimator_

print('Best degree:', polynomial_grid_search.best_params_)
print('Best -RMSE:', polynomial_grid_search.best_score_)