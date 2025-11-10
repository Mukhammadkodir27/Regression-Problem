encoded_map = {}

for col in ['obj_type', 'build_mat', 'has_lift']:
    encoded_map[col] = [c for c in df_train_encoded.columns if c.startswith(col + '_')]

#KeyError: "['obj_type', 'build_mat', 'has_lift'] not in index"

final_features = []

for feat in selected_features:
    if feat in encoded_map:        # categorical → expand dummy columns
        final_features.extend(encoded_map[feat])
    else:                          # numeric → keep as is
        final_features.append(feat)


df_train_final = df_train_encoded[final_features]
df_valid_final = df_valid_encoded[final_features]
df_test_final  = df_test_encoded[final_features]




selected_features = ['dim_m2',
 'n_rooms', 
 'year_built',
 'dist_centre',
 'n_poi',
 'has_park',
 'has_balcony',
 'has_sec',
 'has_store',
 'price_z',
 'src_month',
 'market_volatility',
 'infrastructure_quality',
 'green_space_ratio',
 'estimated_maintenance_cost',
 'obj_type_0d6c4dfc',
 'obj_type_2a6d5c01',
 'obj_type_Unknown',
 'build_mat_7f8c00f9',
 'build_mat_Unknown',
 'has_lift_no',
 'has_lift_yes']