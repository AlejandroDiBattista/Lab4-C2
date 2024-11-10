import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


st.title('Estimación de Ventas Diarias')
def cargar_datos():
    datos = pd.read_csv('ventas.csv')
    return datos


class RedNeuronal(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RedNeuronal, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

def normalizar_datos(X, y):
    X_min, X_max = X.min(), X.max()
    y_min, y_max = y.min(), y.max()

    X_norm = (X - X_min) / (X_max - X_min)
    y_norm = (y - y_min) / (y_max - y_min)

    return X_norm, y_norm, X_min, X_max, y_min, y_max

def desnormalizar_datos(pred, y_min, y_max):
    return pred * (y_max - y_min) + y_min


def entrenar_red(modelo, X, y, learning_rate, epochs):
    X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    optimizador = torch.optim.Adam(modelo.parameters(), lr=learning_rate)
    criterio = nn.MSELoss()
    perdidas = []

    progreso = st.progress(0)
    for epoch in range(epochs):
        predicciones = modelo(X_tensor)
        loss = criterio(predicciones, y_tensor)

        optimizador.zero_grad()
        loss.backward()
        optimizador.step()

        perdidas.append(loss.item())
        progreso.progress((epoch + 1) / epochs)

    return perdidas

def guardar_modelo(modelo, nombre_archivo='modelo_ventas.pth'):
    torch.save(modelo.state_dict(), nombre_archivo)

def graficar_predicciones(datos, modelo, X_norm, y_min, y_max):
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).view(-1, 1)
    prediccion_norm = modelo(X_tensor).detach().numpy()
    prediccion = desnormalizar_datos(prediccion_norm, y_min, y_max)

    plt.figure(figsize=(10, 5))
    plt.plot(datos['dia'], datos['ventas'], label='Ventas reales', color='blue')
    plt.plot(datos['dia'], prediccion, label='Predicción de ventas', color='red')
    plt.xlabel('Día del mes')
    plt.ylabel('Ventas')
    plt.legend()
    st.pyplot(plt)

datos = cargar_datos()

st.sidebar.header("Parámetros de la Red Neuronal")
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.0, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Cantidad de neuronas en la capa oculta", 1, 100, 5)
X = datos['dia'].values.astype(float)
y = datos['ventas'].values.astype(float)
X_norm, y_norm, X_min, X_max, y_min, y_max = normalizar_datos(X, y)

modelo = RedNeuronal(input_size=1, hidden_size=hidden_neurons, output_size=1)

if st.sidebar.button("Entrenar"):
    st.write("Entrenando la red neuronal...")
    perdidas = entrenar_red(modelo, X_norm, y_norm, learning_rate, epochs)
    st.success("Entrenamiento finalizado con éxito.")

    guardar_modelo(modelo)

    plt.figure(figsize=(10, 5))
    plt.plot(perdidas, label='Función de costo', color='purple')
    plt.xlabel('Épocas')
    plt.ylabel('Costo')
    plt.legend()
    st.pyplot(plt)

    st.subheader("Resultados de la predicción")
    graficar_predicciones(datos, modelo, X_norm, y_min, y_max)