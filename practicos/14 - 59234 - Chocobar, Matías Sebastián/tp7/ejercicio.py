import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Título de la aplicación
st.title('Estimación de Ventas Diarias')

# Cargar los datos de ventas
def cargar_datos():
    datos = pd.read_csv('ventas.csv')
    return datos

# Crear Red Neuronal
class RedNeuronal(nn.Module):
    def _init_(self, input_size, hidden_size, output_size):
        super(RedNeuronal, self)._init_()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Normalizar los datos de entrada y salida
def normalizar_datos(X, y)
    X_min, X_max = X.min(), X.max()
    y_min, y_max = y.min(), y.max()
    
    X_norm = (X - X_min) / (X_max - X_min)
    y_norm = (y - y_min) / (y_max - y_min)
    
    return X_norm, y_norm, X_min, X_max, y_min, y_max

# Desnormalizar los datos
def desnormalizar_datos(pred, y_min, y_max):
    return pred * (y_max - y_min) + y_min

# Entrenar la Red Neuronal
def entrenar_red(modelo, X, y, learning_rate, epochs):
    # Convertir a tensores
    X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # Configurar el optimizador y la función de pérdida
    optimizador = torch.optim.Adam(modelo.parameters(), lr=learning_rate)
    criterio = nn.MSELoss()

    # Lista para registrar la pérdida en cada época
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

# Guardar el modelo entrenado
def guardar_modelo(modelo, nombre_archivo='modelo_ventas.pth'):
    torch.save(modelo.state_dict(), nombre_archivo)

# Graficar predicciones y datos reales
def graficar_predicciones(datos, modelo, X_norm, y_min, y_max):
    # Predicción de ventas
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).view(-1, 1)
    prediccion_norm = modelo(X_tensor).detach().numpy()
    prediccion = desnormalizar_datos(prediccion_norm, y_min, y_max)

    # Gráfico de ventas reales y predicciones
    plt.figure(figsize=(10, 5))
    plt.plot(datos['dia'], datos['ventas'], label='Ventas reales', color='blue')
    plt.plot(datos['dia'], prediccion, label='Predicción de ventas', color='red')
    plt.xlabel('Día del mes')
    plt.ylabel('Ventas')
    plt.legend()
    st.pyplot(plt)

# Configuración de la interfaz de Streamlit
datos = cargar_datos()

# Configuración de los parámetros de la red neuronal en el panel izquierdo
st.sidebar.header("Parámetros de la Red Neuronal")
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.0, 1.0, 0.1)
epochs = st.sidebar.slider("Cantidad de épocas", 10, 10000, 100)
hidden_neurons = st.sidebar.slider("Cantidad de neuronas en la capa oculta", 1, 100, 5)

# Normalizar los datos
X = datos['dia'].values.astype(float)
y = datos['ventas'].values.astype(float)
X_norm, y_norm, X_min, X_max, y_min, y_max = normalizar_datos(X, y)

# Crear y entrenar el modelo con los parámetros seleccionados
modelo = RedNeuronal(input_size=1, hidden_size=hidden_neurons, output_size=1)

if st.sidebar.button("Entrenar"):
    st.write("Entrenando la red neuronal...")
    perdidas = entrenar_red(modelo, X_norm, y_norm, learning_rate, epochs)
    st.success("Entrenamiento finalizado con éxito.")

    # Guardar el modelo
    guardar_modelo(modelo)

    # Graficar la función de costo
    plt.figure(figsize=(10, 5))
    plt.plot(perdidas, label='Función de costo', color='purple')
    plt.xlabel('Épocas')
    plt.ylabel('Costo')
    plt.legend()
    st.pyplot(plt)

    # Graficar ventas y predicciones
    st.subheader("Resultados de la predicción")
    graficar_predicciones(datos, modelo, X_norm, y_min, y_max)

