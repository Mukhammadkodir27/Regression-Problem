import pandas as pd

# Suppose you have metrics dictionaries for each model
metrics_polynomial = {'RMSE': 0.133, 'MAE': 0.107, 'MedAE': 0.095, 'MAPE': 0.797, 'R2': 0.930}
metrics_ridge = {'RMSE': 0.200, 'MAE': 0.150, 'MedAE': 0.120, 'MAPE': 1.05, 'R2': 0.85}

# Combine into a DataFrame
results_df = pd.DataFrame({
    'Polynomial Regression': metrics_polynomial,
    'Ridge Regression': metrics_ridge
}).T  # transpose to have models as rows

# Round for better readability
results_df = results_df.round(3)

# Display
print(results_df)
