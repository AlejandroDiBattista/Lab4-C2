import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

st.title('Estimación de Ventas Diarias')

# Panel lateral para ingresar parámetros
st.sidebar.header('Parámetros de la Red Neuronal')
learning_rate = st.sidebar.slider('Tasa de aprendizaje', 0.0, 1.0, 0.1)
epochs = st.sidebar.slider('Cantidad de épocas', 10, 10000, 100)
hidden_neurons = st.sidebar.slider('Neuronas en la capa oculta', 1, 100, 5)
train_button = st.sidebar.button('Entrenar')

# Función para graficar datos iniciales
def plot_data():
    data = pd.read_csv('ventas.csv')
    plt.figure()
    plt.scatter(data['dia'], data['ventas'], label='Datos Reales')
    plt.xlabel('dia')
    plt.ylabel('Ventas')
    plt.title('Ventas Diarias')
    plt.legend()
    st.pyplot(plt)

plot_data()

if train_button:
    # Leer datos
    data = pd.read_csv('ventas.csv')

    # Normalizar datos
    X = data[['dia']].values.astype(np.float32)
    y = data[['ventas']].values.astype(np.float32)

    X_min, X_max = X.min(), X.max()
    y_min, y_max = y.min(), y.max()

    X_norm = (X - X_min) / (X_max - X_min)
    y_norm = (y - y_min) / (y_max - y_min)

    X_tensor = torch.from_numpy(X_norm)
    y_tensor = torch.from_numpy(y_norm)

    # Definir la red neuronal
    class RedNeuronal(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RedNeuronal, self).__init__()
            self.hidden = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.output = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out = self.hidden(x)
            out = self.relu(out)
            out = self.output(out)
            return out

    modelo = RedNeuronal(input_size=1, hidden_size=hidden_neurons, output_size=1)

    # Función de pérdida y optimizador
    criterio = nn.MSELoss()
    optimizador = torch.optim.SGD(modelo.parameters(), lr=learning_rate)

    # Entrenamiento con barra de progreso
    perdida_lista = []

    progreso = st.progress(0)
    estado = st.empty()

    for epoca in range(epochs):
        # Paso hacia adelante
        salidas = modelo(X_tensor)
        perdida = criterio(salidas, y_tensor)

        # Paso hacia atrás y optimización
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()

        perdida_lista.append(perdida.item())

        # Actualizar barra de progreso
        progreso.progress(int((epoca + 1) / epochs * 100))
        estado.text(f'Entrenando... Época {epoca + 1}/{epochs}')

    progreso.empty()
    estado.text('Entrenamiento completado con éxito.')

    # Graficar función de pérdida
    plt.figure()
    plt.plot(range(epochs), perdida_lista)
    plt.xlabel('Épocas')
    plt.ylabel('Función de Costo')
    plt.title('Evolución de la Función de Costo')
    st.pyplot(plt)

    # Predicciones
    with torch.no_grad():
        predicciones = modelo(X_tensor).numpy()
        predicciones_denorm = predicciones * (y_max - y_min) + y_min  # Desnormalizar

    # Graficar datos y predicciones
    plt.figure()
    plt.scatter(data['dia'], data['ventas'], color='blue', label='Datos Reales')
    plt.plot(data['dia'], predicciones_denorm, color='red', label='Predicción')
    plt.xlabel('Dia')
    plt.ylabel('Ventas')
    plt.title('Ventas Diarias y Predicción')
    plt.legend()
    st.pyplot(plt)

    