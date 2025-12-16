import os
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# CONFIG
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

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Dataset not found")
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"CSV column count mismatch: expected {len(COLUMNS)}, got {df.shape[1]}")
    df.columns = COLUMNS
    return df

def preprocess_data(df):
    df = df.drop(columns=['address'])
    for col in NUMERICAL + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[NUMERICAL] = df[NUMERICAL].fillna(df[NUMERICAL].median())
    df[TARGET] = df[TARGET].fillna(df[TARGET].median())
    return df

def encode_and_scale(df, fit=True, encoder=None, scaler=None):
    # categorial
    if fit:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_cat = encoder.fit_transform(df[CATEGORICAL])
    else:
        X_cat = encoder.transform(df[CATEGORICAL])

    # numeric
    if fit:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df[NUMERICAL])
    else:
        X_num = scaler.transform(df[NUMERICAL])

    X = pd.DataFrame(
        data = np.hstack([X_num, X_cat]),
        columns = [f"num_{c}" for c in NUMERICAL] + list(encoder.get_feature_names_out(CATEGORICAL))
    )

    return X, encoder, scaler

def build_tf_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # regression
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def train_tf_model():
    df = load_csv(DATA_FILE)
    df = preprocess_data(df)

    # introducing X, y
    X, encoder, scaler = encode_and_scale(df, fit=True)
    y = df[TARGET].values

    # build, and learning
    model = build_tf_model(X.shape[1])
    model.fit(X.values, y, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

    # saving artiffacts
    model.save(MODEL_FILE)
    joblib.dump(encoder, ENCODER_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("TensorFlow model, encoder and scaler saved")

def predict_rent_tf(user_input: dict) -> float:
    # loading artiffacts
    model = tf.keras.models.load_model(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)

    df = pd.DataFrame([user_input])
    # types
    for col in NUMERICAL:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[NUMERICAL] = df[NUMERICAL].fillna(df[NUMERICAL].median())

    # coding and scaling (без fit)
    X, _, _ = encode_and_scale(df, fit=False, encoder=encoder, scaler=scaler)
    price = float(model.predict(X.values)[0][0])
    return price

def update_model_with_real_price_tf(user_input, real_price):
    df = pd.read_csv(DATA_FILE, header=None)
    df.columns = COLUMNS
    new_row = user_input.copy()
    new_row['price'] = real_price
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False, header=False)

    train_tf_model()

if __name__ == "__main__":
    import numpy as np

    # learning
    train_tf_model()

    # simple CLI-predict
    user_input = {
        'listing_type': input("Listing type: "),
        'area': input("Area: "),
        'rooms': float(input("Rooms: ")),
        'area_sqm': float(input("Area sqm: ")),
        'floor': float(input("Floor: ")),
        'total_floors': float(input("Total floors: ")),
        'building_type': input("Building type: "),
        'construction': input("Construction: "),
        'amenities': input("Amenities: "),
        'latitude': float(input("Latitude: ")),
        'longitude': float(input("Longitude: "))
    }

    predicted_price = predict_rent_tf(user_input)
    print(f"\n💰 Predicted rent (TF): €{predicted_price:.2f}")

    feedback = input("Do you know the real price for this property? (y/n): ").lower()
    if feedback == 'y':
        real_price = float(input("Enter the real price: "))
        update_model_with_real_price_tf(user_input, real_price)
        print(f"Model updated with new price: €{real_price}")
