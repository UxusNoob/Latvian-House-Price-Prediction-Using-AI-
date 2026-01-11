import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import time

DATA_FILE = 'riga.csv'
MODEL_FILE = 'latvia_rent_model_tf.keras'
ENCODER_FILE = 'latvia_rent_encoder.pkl'
SCALER_FILE = 'latvia_rent_scaler.pkl'

COLUMNS = ['listing_type','area','address','rooms','area_sqm','floor',
           'total_floors','building_type','construction','amenities',
           'price','latitude','longitude']

NUMERICAL = ['rooms','area_sqm','floor','total_floors','latitude','longitude']
CATEGORICAL = ['listing_type','area','building_type','construction','amenities']
TARGET = 'price'

df = pd.read_csv(DATA_FILE, header=None)
df.columns = COLUMNS
df = df.drop(columns=['address'])

for col in NUMERICAL + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df[NUMERICAL] = df[NUMERICAL].fillna(df[NUMERICAL].median())
df[TARGET] = df[TARGET].fillna(df[TARGET].median())

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat = encoder.fit_transform(df[CATEGORICAL])

scaler = StandardScaler()
X_num = scaler.fit_transform(df[NUMERICAL])

X = np.hstack([X_num, X_cat])
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    return model

model = build_model(X.shape[1])
start_time = time.time()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
)

model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

training_time = round(time.time() - start_time, 2)

model.save(MODEL_FILE)
joblib.dump(encoder, ENCODER_FILE)
joblib.dump(scaler, SCALER_FILE)
print("Model trained and saved successfully")

y_pred_test = model.predict(X_test).flatten()

ss_res = np.sum((y_test - y_pred_test) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2_test = 1 - (ss_res / ss_tot)

mae_test = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)

print("\nTest Set Metrics:")
print(f"R²: {r2_test:.2f}")
print(f"MAE: {mae_test:.2f}")
print(f"MSE: {mse_test:.2f}")
print(f"RMSE: {rmse_test:.2f}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = np.zeros(y.shape, dtype=float)

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    cv_model = build_model(X.shape[1])
    early_stop_cv = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    reduce_lr_cv = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    cv_model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[early_stop_cv, reduce_lr_cv],
        verbose=0
    )
    y_pred_cv[val_idx] = cv_model.predict(X_val).flatten()

ss_res_cv = np.sum((y - y_pred_cv) ** 2)
ss_tot_cv = np.sum((y - np.mean(y)) ** 2)
r2_cv = 1 - (ss_res_cv / ss_tot_cv)

mae_cv = mean_absolute_error(y, y_pred_cv)
mse_cv = mean_squared_error(y, y_pred_cv)
rmse_cv = np.sqrt(mse_cv)

print("\n5-Fold Cross-Validation Metrics:")
print(f"R²: {r2_cv:.2f}")
print(f"MAE: {mae_cv:.2f}")
print(f"MSE: {mse_cv:.2f}")
print(f"RMSE: {rmse_cv:.2f}")