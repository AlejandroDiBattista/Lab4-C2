import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Cargar los datos de ventas
@st.cache_data
def cargar_datos():
    data = pd.read_csv('ventas.csv')
    return data

# Normalizar los datos
def normalizar_datos(data):
    scaler = MinMaxScaler()
    data['ventas'] = scaler.fit_transform(data[['ventas']])
    return data, scaler

# Crear la red neuronal
class RedNeuronal(nn.Module):
    def __init__(self, neuronas_ocultas):
        super(RedNeuronal, self).__init__()
        self.capa_entrada = nn.Linear(1, neuronas_ocultas)
        self.capa_oculta = nn.Linear(neuronas_ocultas, 1)

    def forward(self, x):
        x = torch.relu(self.capa_entrada(x))
        x = self.capa_oculta(x)
        return x

# Función para entrenar la red neuronal
def entrenar_red(tasa_aprendizaje, cantidad_epocas, neuronas_ocultas, datos_entrada, datos_salida):
    # Crear la red neuronal
    modelo = RedNeuronal(neuronas_ocultas)
    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)
    
    # Convertir los datos a tensores
    entradas = torch.tensor(datos_entrada, dtype=torch.float32).view(-1, 1)
    salidas = torch.tensor(datos_salida, dtype=torch.float32).view(-1, 1)

    # Lista para guardar la evolución de la función de costo
    costos = []
    
    # Inicializar la barra de progreso
    progreso = st.progress(0)

    # Entrenamiento de la red
    for epoca in range(cantidad_epocas):
        optimizador.zero_grad()
        predicciones = modelo(entradas)
        costo = criterio(predicciones, salidas)
        costo.backward()
        optimizador.step()

        # Guardar el costo en cada época
        costos.append(costo.item())

        # Actualizar la barra de progreso
        progreso.progress((epoca + 1) / cantidad_epocas)
    
    # Guardar el modelo entrenado
    torch.save(modelo.state_dict(), 'modelo_entrenado.pth')
    
    return modelo, costos

# Función para predecir con la red entrenada
def predecir(modelo, datos_entrada):
    entradas = torch.tensor(datos_entrada, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        predicciones = modelo(entradas)
    return predicciones.numpy()

# Función para graficar los resultados
def graficar_resultados(datos_entrada, datos_salida, predicciones, costos):
    # Gráfico de ventas originales y predicciones
    st.subheader("Ventas originales vs Predicciones")
    fig, ax = plt.subplots()
    ax.plot(datos_entrada, datos_salida, label='Ventas Originales')
    ax.plot(datos_entrada, predicciones, label='Predicciones de la Red Neuronal', linestyle='--')
    ax.set_xlabel('Día')
    ax.set_ylabel('Ventas')
    ax.set_title('Estimación de Ventas Diarias')
    ax.legend()
    ax.grid(True)  # Agregar cuadriculado
    st.pyplot(fig)
    
    # Gráfico de la función de costo
    st.subheader("Evolución de la Función de Costo")
    fig, ax = plt.subplots()
    ax.plot(costos, label='Función de Costo')
    ax.set_xlabel('Épocas')
    ax.set_ylabel('Costo')
    ax.set_title('Evolución del Costo durante el Entrenamiento')
    ax.legend()
    ax.grid(True)  # Agregar cuadriculado
    st.pyplot(fig)

# Función principal para la aplicación Streamlit
def main():
    st.title('Estimación de Ventas Diarias con Red Neuronal')

    # Sidebar para la entrada de parámetros
    with st.sidebar:
        st.header("Parámetros de la Red Neuronal")
        tasa_aprendizaje = st.slider('Tasa de Aprendizaje', 0.0, 1.0, 0.1, 0.01)
        cantidad_epocas = st.slider('Cantidad de Épocas', 10, 10000, 100, 10)
        neuronas_ocultas = st.slider('Cantidad de Neuronas en la Capa Oculta', 1, 100, 5, 1)
        st.write("##")
        entrenar_button = st.button('Entrenar', key='entrenar')

    # Cargar y mostrar los datos
    data = cargar_datos()
    st.write("## Datos de Ventas Diarias:")
    st.write(data.transpose())  # Mostrar la tabla de ventas de forma horizontal
    
    # Normalizar los datos
    data_normalizada, scaler = normalizar_datos(data)
    datos_entrada = data['dia'].values
    datos_salida = data_normalizada['ventas'].values

    # Entrenar la red neuronal
    if entrenar_button:
        modelo, costos = entrenar_red(tasa_aprendizaje, cantidad_epocas, neuronas_ocultas, datos_entrada, datos_salida)
        
        # Hacer predicciones
        predicciones = predecir(modelo, datos_entrada)
        
        # Mostrar mensaje de éxito
        st.success('Entrenamiento Completo!')
        
        # Graficar resultados
        graficar_resultados(datos_entrada, datos_salida, predicciones, costos)

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
