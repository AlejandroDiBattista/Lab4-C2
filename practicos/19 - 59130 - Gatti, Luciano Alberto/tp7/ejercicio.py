import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Título de la aplicación
st.title('Estimación de Ventas Diarias')

# Parámetros de la red en la barra lateral
st.sidebar.header("Parámetros de la Red Neuronal")
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.0, 1.0, 0.1, 0.01)
epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100, 10)
hidden_neurons = st.sidebar.slider("Cantidad de neuronas en la capa oculta", 1, 100, 5, 1)

# Leer datos
data = pd.read_csv('ventas.csv')
dias = data[['dia']].values
ventas = data[['ventas']].values

# Normalizar los datos
def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min()), data.min(), data.max()

ventas_normalized, ventas_min, ventas_max = normalize_data(ventas)

# Convertir datos a tensores de PyTorch
X = torch.tensor(dias, dtype=torch.float32)
y = torch.tensor(ventas_normalized, dtype=torch.float32)

# Definir la red neuronal
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_neurons):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(1, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, 1)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        return self.output(x)

# Entrenar la red neuronal
def train_network(learning_rate, epochs, hidden_neurons):
    model = NeuralNetwork(hidden_neurons)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_history = []

    # Barra de progreso en Streamlit
    progress_bar = st.progress(0)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)

    return model, loss_history

# Botón para iniciar el entrenamiento
if st.sidebar.button("Entrenar"):
    model, loss_history = train_network(learning_rate, epochs, hidden_neurons)
    
    # Gráfico de la evolución de la función de costo
    st.subheader("Evolución de la función de costo")
    fig, ax = plt.subplots()
    ax.plot(loss_history, color="red")
    ax.set_xlabel("Épocas")
    ax.set_ylabel("Costo")
    st.pyplot(fig)
    
    # Predicción con la red neuronal entrenada
    with torch.no_grad():
        predictions = model(X).numpy()
        predictions = predictions * (ventas_max - ventas_min) + ventas_min  # Desnormalizar

    # Gráfico de ventas y predicciones
    st.subheader("Predicción de ventas")
    fig, ax = plt.subplots()
    ax.plot(dias, ventas, label="Ventas reales", color="blue")
    ax.plot(dias, predictions, label="Predicción", color="orange")
    ax.legend()
    st.pyplot(fig)
    
    # Mensaje de éxito
    st.success("Entrenamiento finalizado con éxito")