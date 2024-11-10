import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

st.title('Predicción de Ventas Diarias')

# Cargar y procesar datos
data_df = pd.read_csv('ventas.csv')
dias = data_df['dia'].values
ventas = data_df['ventas'].values

# Normalización de los datos
escalador = MinMaxScaler()
dias_escalados = escalador.fit_transform(dias.reshape(-1, 1))
ventas_escaladas = escalador.fit_transform(ventas.reshape(-1, 1))

# Configuración de parámetros en la barra lateral
st.sidebar.header("Configuración de Entrenamiento")
tasa_aprendizaje = st.sidebar.slider("Tasa de Aprendizaje", 0.0, 1.0, 0.1)
epocas = st.sidebar.slider("Número de Épocas", 10, 10000, 100)
neuronas_ocultas = st.sidebar.slider("Cantidad de Neuronas Ocultas", 1, 100, 5)
boton_iniciar = st.sidebar.button("Iniciar Entrenamiento")

# Clase de la red neuronal
class ModeloVentas(nn.Module):
    def __init__(self, neuronas_ocultas):
        super(ModeloVentas, self).__init__()
        self.capa_oculta = nn.Linear(1, neuronas_ocultas)
        self.capa_salida = nn.Linear(neuronas_ocultas, 1)

    def forward(self, x):
        x = torch.relu(self.capa_oculta(x))
        x = self.capa_salida(x)
        return x

# Función para entrenar el modelo
def entrenar_modelo(tasa_aprendizaje, epocas, neuronas_ocultas):
    modelo = ModeloVentas(neuronas_ocultas)
    funcion_perdida = nn.MSELoss()
    optimizador = optim.SGD(modelo.parameters(), lr=tasa_aprendizaje)

    x_entrenamiento = torch.tensor(dias_escalados, dtype=torch.float32)
    y_entrenamiento = torch.tensor(ventas_escaladas, dtype=torch.float32)
    historial_perdida = []

    progreso = st.empty()
    barra_progreso = st.progress(0)

    for epoca in range(epocas):
        optimizador.zero_grad()
        predicciones = modelo(x_entrenamiento)
        perdida = funcion_perdida(predicciones, y_entrenamiento)
        perdida.backward()
        optimizador.step()

        historial_perdida.append(perdida.item())
        barra_progreso.progress((epoca + 1) / epocas)
        progreso.text(f'Época {epoca + 1}/{epocas} - Pérdida: {perdida.item():.6f}')

    return modelo, historial_perdida

# Ejecución del entrenamiento y visualización
if boton_iniciar:
    modelo, historial_perdida = entrenar_modelo(tasa_aprendizaje, epocas, neuronas_ocultas)

    # Gráfico de la pérdida durante el entrenamiento
    fig, ax = plt.subplots()
    ax.plot(historial_perdida, color='green')
    ax.set_title("Historial de Pérdida")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
    st.pyplot(fig)

    # Predicciones del modelo
    x_prueba = torch.tensor(dias_escalados, dtype=torch.float32)
    with torch.no_grad():
        predicciones_escaladas = modelo(x_prueba).numpy()
    predicciones_finales = escalador.inverse_transform(predicciones_escaladas)

    # Gráfico de resultados
    fig, ax = plt.subplots()
    ax.scatter(dias, ventas, color='blue', label='Datos Observados')
    ax.plot(dias, predicciones_finales, color='red', label='Predicción')
    ax.set_title("Predicción de Ventas Diarias")
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)

    st.success("Entrenamiento finalizado exitosamente")
