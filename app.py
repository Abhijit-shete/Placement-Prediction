import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Title
st.title("🎓 Placement Prediction App (Deep Learning)")

# Load data
df = pd.read_csv("placement.csv")

# Encode target
le = LabelEncoder()
df['placement'] = le.fit_transform(df['placement'])

# Split features & target
X = df.drop('placement', axis=1)
y = df['placement']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_scaled, y, epochs=50, verbose=0)

# User input
st.header("Enter Student Details")

inputs = []
for col in X.columns:
    val = st.number_input(f"{col}")
    inputs.append(val)

if st.button("Predict Placement"):
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    pred = model.predict(input_scaled)

    st.write("Placement Probability:", float(pred))

    if pred > 0.5:
        st.success("✅ Student will be Placed")
    else:
        st.error("❌ Student will NOT be Placed")
