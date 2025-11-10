from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

numeric_features = ["age", "income", "size"]
categorical_features = ["city", "type"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", "passthrough", categorical_features)
    ]
)


pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", ridge)
])


pipeline = Pipeline([
    ("model", ridge)
])
