from pipelines.polynomial import polynomial_pipeline
from pipelines.knn import knn_pipeline
from utils.preprocessing import load_data, split_data
from sklearn.metrics import mean_squared_error

def main():
    # 1. Load data
    X, y = load_data()

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Choose a model pipeline
    model = polynomial_pipeline()

    # 4. Fit
    model.fit(X_train, y_train)

    # 5. Evaluate
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print("RMSE:", rmse)

    # 6. Save model if needed
    # joblib.dump(model, "models/polynomial.pkl")

if __name__ == "__main__":
    main()
