import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Cargar los datos
@st.cache
def load_data():
    data = pd.read_csv('ventas.csv')
    return data

# Definir el modelo de red neuronal
class VentasNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VentasNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# Configuración de la interfaz de Streamlit
st.title("Predicción de Ventas Diarias con Redes Neuronales")

# Parámetros de entrada en el panel izquierdo
st.sidebar.header("Parámetros de la Red Neuronal")
learning_rate = st.sidebar.slider("Tasa de Aprendizaje", 0.0, 1.0, 0.1, 0.01)
epochs = st.sidebar.slider("Cantidad de Épocas", 10, 10000, 100)
hidden_size = st.sidebar.slider("Cantidad de Neuronas en la Capa Oculta", 1, 100, 5)
train_button = st.sidebar.button("Entrenar")

# Cargar y visualizar los datos
data = load_data()
st.write("Datos de ventas diarias:")
st.write(data)

# Escalar los datos
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['día', 'ventas']])

# Preparar datos para el modelo
X = torch.tensor(scaled_data[:, 0], dtype=torch.float32).unsqueeze(1)
y = torch.tensor(scaled_data[:, 1], dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Visualización inicial de los datos
fig, ax = plt.subplots()
ax.plot(data['día'], data['ventas'], 'bo-', label="Ventas reales")
st.pyplot(fig)

if train_button:
    # Crear el modelo y el optimizador
    model = VentasNN(input_size=1, hidden_size=hidden_size, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Barra de progreso
    progress_bar = st.progress(0)

    # Entrenamiento de la red neuronal
    cost_history = []
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        cost_history.append(total_loss / len(dataloader))
        progress_bar.progress((epoch + 1) / epochs)

    st.success("Entrenamiento completado con éxito")
    
    # Gráfico de evolución de la función de costo
    fig, ax = plt.subplots()
    ax.plot(cost_history, label="Función de costo")
    ax.set_xlabel("Épocas")
    ax.set_ylabel("Costo")
    ax.legend()
    st.pyplot(fig)

    # Predicciones de la red neuronal
    with torch.no_grad():
        predictions = model(X).detach().numpy()
        predictions = scaler.inverse_transform(
            np.concatenate((scaled_data[:, :1], predictions), axis=1)
        )[:, 1]
    
    # Visualización de las predicciones
    fig, ax = plt.subplots()
    ax.plot(data['día'], data['ventas'], 'bo-', label="Ventas reales")
    ax.plot(data['día'], predictions, 'r-', label="Predicción")
    ax.legend()
    st.pyplot(fig)

## Crear Red Neuronal
## Leer Datos
## Normalizar Datos
## Entrenar Red Neuronal
## Guardar Modelo
## Graficar Predicciones
st.title('Estimación de Ventas Diarias')