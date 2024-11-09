import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Configuración de la interfaz de Streamlit
st.title('Estimación de Ventas Diarias')

# Parámetros de la Red Neuronal en el panel lateral
st.sidebar.title("Parámetros de la Red Neuronal")
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.0, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Neurona en capa oculta", 1, 100, 5)
train_button = st.sidebar.button("Entrenar")

# Cargar datos de ventas desde un archivo CSV
data = pd.read_csv("ventas.csv")
st.write("Datos de ventas diarias:")
st.write(data)

# Normalizar los datos
def normalize_data(data):
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_x.fit_transform(data[['dia']].values)
    y = scaler_y.fit_transform(data[['ventas']].values)
    return X, y, scaler_x, scaler_y

X, y, scaler_x, scaler_y = normalize_data(data)

# Definir la estructura de la Red Neuronal
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Entrenamiento de la Red Neuronal
def train_model(X, y, learning_rate, epochs, hidden_neurons):
    model = NeuralNetwork(input_size=1, hidden_size=hidden_neurons, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    inputs = torch.tensor(X, dtype=torch.float32)
    targets = torch.tensor(y, dtype=torch.float32)
    cost_history = []

    # Barra de progreso
    progress_bar = st.progress(0)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        cost_history.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)

    return model, cost_history

# Graficar Predicciones
def plot_predictions(model, X, y, scaler_x, scaler_y):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X, dtype=torch.float32)).numpy()
    predictions = scaler_y.inverse_transform(predictions)
    true_values = scaler_y.inverse_transform(y)

    plt.figure(figsize=(10, 6))
    plt.plot(data['dia'], true_values, label="Ventas reales", color="blue")
    plt.plot(data['dia'], predictions, label="Predicción", linestyle="--", color="red")
    plt.xlabel("Día")
    plt.ylabel("Ventas")
    plt.legend()
    st.pyplot(plt)

# Entrenar y visualizar resultados
if train_button:
    st.write("Entrenando la red neuronal, por favor espera...")
    model, cost_history = train_model(X, y, learning_rate, epochs, hidden_neurons)
    
    # Mostrar mensaje de éxito
    st.write("¡Entrenamiento finalizado con éxito!")
    
    # Mostrar evolución del costo
    st.write("Evolución del costo durante el entrenamiento:")
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, label="Costo")
    plt.xlabel("Épocas")
    plt.ylabel("Costo")
    plt.title("Evolución del Costo")
    st.pyplot(plt)

    # Graficar las predicciones
    st.write("Predicción de ventas diarias:")
    plot_predictions(model, X, y, scaler_x, scaler_y)
