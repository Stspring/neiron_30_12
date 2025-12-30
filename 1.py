# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
# from sklearn.metrics import mean_squared_error

# # Загрузка данных
# X_train = pd.read_csv("X_train.csv", index_col=0).drop(columns=["Id"]) 
# X_test = pd.read_csv("X_test.csv", index_col=0).drop(columns=["Id"])
# y_train = pd.read_csv("y_train.csv", index_col=0)["Price"] 
# y_test = pd.read_csv("y_test.csv", index_col=0)["Price"]  

# # Определение признаков
# categorical_features = ["Location", "Condition", "Garage"]
# numeric_features = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]

# # Предобработка
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
#         ("num", "passthrough", numeric_features)
#     ]
# )

# # --- RandomForest модель ---
# rf_model = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("regressor", RandomForestRegressor(n_estimators=10, random_state=42))
# ])

# rf_model.fit(X_train, y_train)
# rf_pred = np.round(rf_model.predict(X_test)).astype(int)

# # --- XGBoost модель ---
# reg_params = {
#     'learning_rate': 0.018256092238479713,
#     'n_estimators': 260,
#     'max_depth': 3,
#     'min_child_weight': 5,
#     'subsample': 0.9751933727541958,
#     'colsample_bytree': 0.7462352150133582,
#     'lambda': 1,
#     'alpha': 0,
#     'objective': 'reg:squarederror',
#     'eval_metric': 'logloss',
#     'base_score': y_train.mean(),
#     'booster': 'dart',
#     'random_state': 10,
#     'gamma': 0.9992475012345792
# }

# xgb_model = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("regressor", xgb.XGBRegressor(**reg_params))
# ])

# xgb_model.fit(X_train, y_train)
# xgb_pred = np.round(xgb_model.predict(X_test)).astype(int)

# # --- Метрики RMSE ---
# rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
# xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

# print(f"RMSE RandomForest: {rf_rmse:.2f}")
# print(f"RMSE XGBoost: {xgb_rmse:.2f}")

# # --- График сравнения ---
# plt.figure(figsize=(12, 6))
# plt.plot(y_test.values, label="Реальные значения", color="blue")
# plt.plot(rf_pred, label="RandomForest предсказания", color="red", linestyle="--")
# plt.plot(xgb_pred, label="XGBoost предсказания", color="green", linestyle=":")
# plt.title("Сравнение предсказаний разных моделей с реальными значениями")
# plt.ylabel("Цена")
# plt.legend()
# plt.show()







# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import RandomizedSearchCV

# # --- Загрузка данных ---
# X_train = pd.read_csv("X_train.csv", index_col=0).drop(columns=["Id"]) 
# X_test = pd.read_csv("X_test.csv", index_col=0).drop(columns=["Id"])
# y_train = pd.read_csv("y_train.csv", index_col=0)["Price"] 
# y_test = pd.read_csv("y_test.csv", index_col=0)["Price"]  

# # --- Feature Engineering ---
# def add_features(df):
#     df = df.copy()
#     df["HouseAge"] = 2025 - df["YearBuilt"]
#     df["AreaPerRoom"] = df["Area"] / (df["Bedrooms"] + df["Bathrooms"] + 1)
#     df["HasGarage"] = (df["Garage"] != "None").astype(int)
#     return df

# X_train = add_features(X_train)
# X_test = add_features(X_test)

# # --- Определение признаков ---
# categorical_features = ["Location", "Condition", "Garage"]
# numeric_features = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt", "HouseAge", "AreaPerRoom", "HasGarage"]

# # --- Предобработка ---
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
#         ("num", Pipeline(steps=[("scaler", StandardScaler())]), numeric_features)
#     ]
# )

# # --- RandomForest ---
# rf_pipe = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1))
# ])

# rf_params = {
#     "regressor__n_estimators": [400, 800, 1000],
#     "regressor__max_depth": [15, 20, 30],
#     "regressor__min_samples_split": [2, 5, 10],
#     "regressor__min_samples_leaf": [1, 2, 4],
#     "regressor__max_features": ["sqrt", "log2"]
# }

# rf_search = RandomizedSearchCV(
#     rf_pipe, rf_params, n_iter=30, cv=3,
#     scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1
# )
# rf_search.fit(X_train, y_train)
# rf_best = rf_search.best_estimator_
# rf_pred = np.round(rf_best.predict(X_test)).astype(int)

# # --- XGBoost ---
# xgb_pipe = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("regressor", xgb.XGBRegressor(objective="reg:squarederror", random_state=42))
# ])

# xgb_params = {
#     "regressor__n_estimators": [800, 1200, 2000],
#     "regressor__learning_rate": [0.005, 0.01, 0.02, 0.05],
#     "regressor__max_depth": [4, 6, 8, 10],
#     "regressor__subsample": [0.8, 0.9, 1.0],
#     "regressor__colsample_bytree": [0.7, 0.85, 1.0],
#     "regressor__min_child_weight": [1, 3, 5],
#     "regressor__gamma": [0, 0.1, 0.3]
# }

# xgb_search = RandomizedSearchCV(
#     xgb_pipe, xgb_params, n_iter=30, cv=3,
#     scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1
# )
# xgb_search.fit(X_train, y_train)
# xgb_best = xgb_search.best_estimator_
# xgb_pred = np.round(xgb_best.predict(X_test)).astype(int)

# # --- Метрики RMSE ---
# rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
# xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

# print("Лучшие параметры RF:", rf_search.best_params_)
# print(f"RMSE RandomForest: {rf_rmse:.2f}")
# print("Лучшие параметры XGB:", xgb_search.best_params_)
# print(f"RMSE XGBoost: {xgb_rmse:.2f}")

# # --- График сравнения ---
# plt.figure(figsize=(12, 6))
# plt.plot(y_test.values, label="Реальные значения", color="blue")
# plt.plot(rf_pred, label="RandomForest предсказания", color="red", linestyle="--")
# plt.plot(xgb_pred, label="XGBoost предсказания", color="green", linestyle=":")
# plt.title("Сравнение предсказаний разных моделей с реальными значениями")
# plt.ylabel("Цена")
# plt.legend()
# plt.show()



