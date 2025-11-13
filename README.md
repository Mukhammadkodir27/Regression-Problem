# Apartment Price Prediction – Regression Project

## Project Overview
This project focuses on predicting apartment prices in a specific European region using advanced machine learning techniques. The dataset contains detailed apartment attributes, including size, number of rooms, floor information, building year, distances to nearby amenities, and neighborhood characteristics. The goal is to build a robust regression model that accurately estimates apartment prices (`price_z`) based on these predictors.

This regression problem reflects a **real-world scenario**, where accurate property valuation is critical for buyers, sellers, real estate agencies, and urban planners. The developed model can provide quick, data-driven price estimates, account for local market dynamics, and assist in investment or policy decisions.

## Dataset
The dataset consists of:

- **Training set:** 156,454 observations with 34 features + target (`price_z`)  
- **Test set:** 39,114 observations with 33 features (target not included)  

Key features include: apartment size (`dim_m2`), number of rooms (`n_rooms`), building characteristics, distances to amenities, ownership type, condition, and additional synthetic indicators such as `market_volatility`, `infrastructure_quality`, `popularity_index`, and `green_space_ratio`.

## Project Structure
The project is organized to reflect a professional workflow:
├── notebooks/ # Separate notebooks for preprocessing, EDA, model building, validation, and testing
├── data/ # Raw and processed datasets
├── src/
│ └── utils/ # Helper functions and pipeline utilities
├── visualizations/ # Plots and charts for EDA and model evaluation
├── models/ # Saved models using pickle
├── pipelines/ # Preprocessing and modeling pipelines
├── README.md
└── .gitignore


Version control was managed using **Git**, and large model computations were performed in **Google Colab**. Local development and testing were done using **VS Code** and **Jupyter Notebook**.

## Data Preprocessing
- Missing values handled using **KNN imputation**.  
- Target variable (`price_z`) was **log-scaled** to normalize distribution.  
- Features standardized using **StandardScaler** in pipelines.  
- Categorical variables encoded for model compatibility.  
- Dataset split: **50% train / 25% validation / 25% test** to ensure proper evaluation.

## Modeling Approach
The project followed a structured **machine learning workflow**:

1. **Pipeline Creation** – Preprocessing + model steps were combined into pipelines for reproducibility.
2. **Model Exploration** – Base algorithms included:  
   - Linear Regression, Ridge, Lasso  
   - Support Vector Regression (SVR)  
   - Polynomial Regression  
   - Random Forest  
3. **Hyperparameter Tuning** – Applied `GridSearchCV` and `RandomizedSearchCV` for algorithm optimization:
   - Polynomial degree tuning  
   - Regularization (Lasso/Ridge)  
   - SVR kernel and C parameters  
   - Tree depth, number of estimators, and subsample rates  
4. **Ensemble Methods** – To boost predictive performance:  
   - **Boosting:** XGBoost, LightGBM, CatBoost (gradient boosting to reduce bias)  
   - **Bagging:** Random Forest, Bagged Decision Trees (variance reduction)  
   - **Stacking:** Meta-model combining XGBoost, LightGBM, CatBoost, and Random Forest for superior accuracy  
5. **Model Evaluation** – Metrics computed: RMSE, MAE, Median AE, MAPE, R².  
6. **Model Persistence** – Final models, preprocessed data, and selected features saved using **pickle** for production deployment.

### Validation and Test Results
During validation, multiple models were compared. The **meta-stacked model** combining boosting and bagging methods outperformed all individual models.

**Test Set Performance (Stacked Meta-Model):**

| Metric | Value |
|--------|-------|
| RMSE   | 0.1091 |
| MAE    | 0.0925 |
| MedAE  | 0.0879 |
| MAPE   | 0.6895 |
| R²     | 0.9525 |

- The stacked model slightly outperformed **XGBoost**, **LightGBM**, and **CatBoost** individually, achieving **95% explained variance**.  
- This demonstrates that **combining multiple models** via stacking can capture complementary patterns in the data, improving prediction stability and accuracy.

**Individual Boosting Model Example (XGBoost):**

| Metric | Value |
|--------|-------|
| RMSE   | 0.1094 |
| MAE    | 0.0926 |
| MedAE  | 0.0880 |
| MAPE   | 0.6900 |
| R²     | 0.9523 |

## Tools and Technologies
- **Python:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Modeling:** Linear models, Polynomial Regression, Ridge/Lasso, SVR, Random Forest, Boosting (XGBoost, LightGBM, CatBoost), Stacking  
- **Hyperparameter Tuning:** GridSearchCV, RandomizedSearchCV  
- **Persistence:** Pickle  
- **Version Control:** Git & GitHub  
- **Computation:** Google Colab for heavy computations, local VS Code/Jupyter for development  

## Future Work
- Apply additional ensemble techniques (Gradient Boosting variations, Voting Regressor).  
- Analyze **feature importance** from boosting and bagging models for interpretability.  
- Include **time-dependent economic indicators** for seasonal or market trend capture.  
- Deploy as **production-ready service** with real-time predictions and API integration.

## Real-World Impact
This regression model can serve as a reliable **price estimation tool** for the real estate market:  
- Quickly estimate apartment prices based on multiple predictors.  
- Understand the influence of location, amenities, and apartment characteristics on price.  
- Support data-driven investment, sales, or policy decisions.  
- Stack-based meta-model ensures **high reliability and robustness**, suitable for production deployment in applications or dashboards.

---

This project demonstrates an end-to-end **machine learning solution**, from preprocessing and model building to ensemble optimization and production-ready deployment, following **best practices for professional ML workflows**.
