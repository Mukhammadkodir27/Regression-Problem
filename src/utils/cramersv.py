import pandas as pd
import numpy as np

# check mutual relationships between categorical predictors Xi with Xi+1 in a way similar
# to correlation matrix

# the strength of relation between two Categorical variables can be tested using the
# Cramer's V coefficient. Higher values mean stronger relationship.
# It takes values from 0 to 1.
# If both variables have only two levels, Cramer's V take values from -1 to 1


def cramers_v(contingency_table):
    ''' Calculate Cramer's V statistic from a contingency table. '''
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    return np.sqrt(chi2 / (n*(min(contingency_table.shape) - 1)))


def calculate_cramers_v_matrix(dataset, cat_vars):
    ''' Calculate the Cramer's V for each pair of categorical variables. '''
    # create an empty matrix to store the results
    cramers_v_matrix = pd.DataFrame(np.zeros((len(cat_vars), len(cat_vars))),
                                    columns=cat_vars,
                                    index=cat_vars)
    for i in range(len(cat_vars)):
        for j in range(i, len(cat_vars)):  # to avoid recalculating for the same pair
            var_1 = cat_vars[i]
            var_2 = cat_vars[j]

            # create a contingency table for the two variables
            contingency_table = pd.crosstab(dataset[var_1], dataset[var_2])

            # calculate Cramer's V
            cramers_v_value = cramers_v(contingency_table)

            cramers_v_matrix.loc[var_1, var_2] = cramers_v_value
            # symmetric matrix
            cramers_v_matrix.loc[var_2, var_1] = cramers_v_value

    return cramers_v_matrix
