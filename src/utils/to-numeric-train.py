obj_cols = df_train_encoded[selected_features].select_dtypes(include='object').columns
obj_cols

# ['has_park', 'has_balcony', 'has_sec', 'has_store', 'src_month']

bool_cols = ['has_park', 'has_balcony', 'has_sec', 'has_store']

df_train_encoded[bool_cols] = df_train_encoded[bool_cols].replace({'yes': 1, 'no': 0})


df_train_encoded['src_month'] = pd.to_datetime(df_train_encoded['src_month'])
df_train_encoded['src_month'] = df_train_encoded['src_month'].map(lambda x: x.toordinal())


# df_train_encoded['src_month'] = pd.to_datetime(df_train_encoded['src_month'])
# df_train_encoded['src_month'] = df_train_encoded['src_month'].dt.year * 100 + df_train_encoded['src_month'].dt.month


df_train_encoded = df_train_encoded.apply(pd.to_numeric)
