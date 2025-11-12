# Apartment Price Prediction – Regression Project

## Project Overview
This project focuses on predicting apartment prices in a specific European region using machine learning techniques. The dataset contains detailed apartment attributes, including size, number of rooms, floor information, building year, distances to nearby amenities, and neighborhood characteristics. The goal is to build a robust regression model that accurately estimates apartment prices (`price_z`) based on these predictors.

This regression problem reflects a **real-world scenario**, where accurate property valuation is critical for buyers, sellers, real estate agencies, and urban planners. The developed model can provide quick, data-driven price estimates, account for local market dynamics, and assist in investment or policy decisions.

## Dataset
The dataset consists of:

- **Training set:** 156,454 observations with 34 features + target (`price_z`)  
- **Test set:** 39,114 observations with 33 features (target not included)  

Key features include: apartment size (`dim_m2`), number of rooms (`n_rooms`), building characteristics, distances to amenities, ownership type, condition, and additional synthetic indicators such as `market_volatility`, `infrastructure_quality`, `popularity_index`, and `green_space_ratio`.

## Project Structure
The project was organized to reflect a professional workflow:

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
- Missing values were handled using **KNN imputation**.  
- Target variable (`price_z`) was **log-scaled** to normalize distribution.  
- Features were standardized using **StandardScaler** in pipelines.  
- Categorical variables were properly encoded for model compatibility.  
- Dataset split: **50% train / 25% validation / 25% test** to ensure proper evaluation.

## Modeling Approach
The project followed a structured **machine learning workflow**:

1. **Pipeline Creation** – Preprocessing + model steps were combined into pipelines for reproducibility.
2. **Model Selection** – Explored multiple algorithms: Linear Regression, Ridge, Lasso, Support Vector Regression (SVR), Polynomial Regression, and Random Forest.  
3. **Hyperparameter Tuning** – Applied `GridSearchCV` and `RandomizedSearchCV` for algorithm optimization, including polynomial degree, regularization strength, and SVR parameters.  
4. **Model Evaluation** – Metrics computed: RMSE, MAE, Median AE, MAPE, R².  
5. **Model Persistence** – Final models, preprocessed data, and selected features saved using **pickle** for production deployment.

### Validation and Test Results
During validation, multiple models were compared. The **tuned polynomial regression model** (degree 2) demonstrated the best performance.  

**Test Set Performance:**

| Metric | Value |
|--------|-------|
| RMSE   | 0.134 |
| MAE    | 0.108 |
| MedAE  | 0.095 |
| MAPE   | 0.806 |
| R²     | 0.930 |

This shows the model generalizes well to unseen data, explaining approximately **93% of the variance** in apartment prices. Compared to other models, it delivers the lowest prediction errors and high reliability.

## Tools and Technologies
- **Python:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Modeling:** Linear models, Polynomial Regression, Ridge/Lasso, SVR, Random Forest, with pipelines  
- **Hyperparameter Tuning:** GridSearchCV, RandomizedSearchCV  
- **Persistence:** Pickle  
- **Version Control:** Git & GitHub  
- **Computation:** Google Colab for heavy computations, local VS Code/Jupyter for development  

## Future Work
- Implement **XGBoost**, Gradient Boosting, and other ensemble techniques to potentially improve accuracy.  
- Explore **feature importance** analysis for better interpretability.  
- Integrate **time-dependent economic indicators** to capture seasonal or market trends.  
- Optimize pipeline for production deployment with real-time predictions.

## Real-World Impact
This regression model can serve as a reliable **price estimation tool** for the real estate market. It allows stakeholders to:  
- Quickly estimate apartment prices based on multiple predictors.  
- Understand the influence of location, amenities, and apartment characteristics on price.  
- Make data-driven investment, sales, or policy decisions.  
- Deploy in applications or dashboards to provide end-users with accurate, production-ready price predictions.

---

This project demonstrates an end-to-end **machine learning solution**, from preprocessing and model building to evaluation, hyperparameter tuning, and deployment readiness. It was built in multiple steps and notebooks, reflecting **best practices in production-ready ML workflows**.
