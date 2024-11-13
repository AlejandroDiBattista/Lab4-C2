# Importar librerías necesarias
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Configuración de título en Streamlit
st.title('Estimación de Ventas Diarias')

# Cargar datos de ventas
try:
    # Leer el archivo CSV
    ventas = pd.read_csv('ventas.csv')
    st.write("Datos cargados correctamente:")
    st.write(ventas.head())
except Exception as e:
    st.error(f"Error al cargar datos: {e}")

# Preprocesamiento de datos
if 'ventas' in locals():
    # Asegurarse de que los datos no tienen valores nulos
    ventas = ventas.dropna()

    # Separar características y etiquetas
    X = ventas.iloc[:, :-1].values  # Todas las columnas excepto la última
    y = ventas.iloc[:, -1].values   # Solo la última columna (Ventas)

    # Escalado de características
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convertir a tensores
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Definir la arquitectura de la red neuronal
    class RedNeuronal(nn.Module):
        def __init__(self):
            super(RedNeuronal, self).__init__()
            self.fc1 = nn.Linear(X_train.shape[1], 64)  # Tamaño entrada -> capa oculta
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)  # Capa de salida

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Crear instancia de la red neuronal
    modelo = RedNeuronal()
    criterio = nn.MSELoss()  # Función de pérdida
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.01)

    # Entrenamiento del modelo
    epochs = 100
    historial_perdida = []

    for epoch in range(epochs):
        modelo.train()
        optimizador.zero_grad()
        prediccion = modelo(X_train)
        perdida = criterio(prediccion, y_train)
        perdida.backward()
        optimizador.step()
        historial_perdida.append(perdida.item())

        # Mostrar progreso en cada 10 épocas
        if (epoch+1) % 10 == 0:
            st.write(f"Época {epoch+1}/{epochs}, Pérdida: {perdida.item():.4f}")

    # Evaluación del modelo
    modelo.eval()
    with torch.no_grad():
        predicciones = modelo(X_test)
        perdida_prueba = criterio(predicciones, y_test)
        st.write(f"Pérdida en prueba: {perdida_prueba.item():.4f}")

    # Graficar el historial de pérdida
    plt.figure(figsize=(10, 5))
    plt.plot(historial_perdida, label='Pérdida de entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    st.pyplot(plt)

    # Graficar predicciones vs datos reales
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Real')
    plt.plot(predicciones, label='Predicción')
    plt.legend()
    st.pyplot(plt)
