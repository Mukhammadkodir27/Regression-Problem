# define a function that will perform backward elimination
# using AIC or BIS as the criterion

def backward_elimination_aic_bic(X, y, criterion = 'AIC'):
    '''
    Perform backward elimination using AIC or BIC as the criterion.

    Parameters:
        X (DataFrame): Feature matrix with a constant column.
        y (Series): Target variable.
        criterion (str): 'AIC' or 'BIC' (default: 'AIC').

    Returns:
        statsmodels OLS fitted model with selected features.
    '''
    model = sm.OLS(y, X).fit()

    while len(X.columns) > 1: # at least one predictor + constant
        best_criterion = model.aic if criterion == 'AIC' else model.bic

        # compute AIC/BIC for models without each predictor
        aic_bic_values = {}
        for col in X.columns[1:]:  # skip intercept
            X_new = X.drop(columns = [col])
            new_model = sm.OLS(y, X_new).fit()
            aic_bic_values[col] = new_model.aic if criterion == 'AIC' else new_model.bic

        # find the feature whose removal lowers AIC/BIC the most
        worst_feature = min(aic_bic_values, key = aic_bic_values.get)
        worst_aic_bic = aic_bic_values[worst_feature]

        # stop if no improvement
        if worst_aic_bic >= best_criterion:
            break

        # remove the feature and update the model
        X = X.drop(columns = [worst_feature])
        model = sm.OLS(y, X).fit()

    return model