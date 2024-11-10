import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


## Crear Red Neuronal
## Leer Datos
## Normalizar Datos
## Entrenar Red Neuronal
## Guardar Modelo
## Graficar Predicciones
st.title('Estimación de Ventas Diarias')
@st.cache_data
def cargar_data():
    data = pd.read_csv('ventas.csv')
    return data

data = cargar_data()

x = data[['dia']].astype(np.float32)
y = data[['ventas']].astype(np.float32)
x_normalizado = (x - x.min()) / (x.max() - x.min())
y_normalizado = (y - y.min()) / (y.max() - y.min())

nuevoModelo = None

def entrenar(tasa_de_aprendizaje, epocas, neuronas, x, y):
    modelo = RedNeuronal(neuronas_ocultas)
    criterio = nn.MSELoss()
    optimizador = torch.optim.SGD(modelo.parameters(), lr=tasa_de_aprendizaje)
    progress = st.progress(0, "Entrenando...")
    for epoca in range(epocas):
        prediccion = modelo(x)
        perdida = criterio(prediccion,y) 
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
        print(f'epoca: {epoca} de {epocas}')
        progress.progress((epoca + 1) / epocas, "Entrenando...")
    torch.save(modelo.state_dict(), 'modelo_ventas.pth')
    return modelo

tasa_de_aprendizaje = st.sidebar.slider('Tasa de Aprendizaje : ',0.0,1.0,0.1)
num_de_epocas = st.sidebar.slider('Cantidad de épocas', 10, 10000, 100)
neuronas_ocultas = st.sidebar.slider('Neuronas en la capa oculta', 1, 100, 5)

class RedNeuronal(nn.Module):
    def __init__(self, neuronas_ocultas):
        super(RedNeuronal, self).__init__()
        self.oculta = nn.Linear(1, neuronas_ocultas)
        self.salida = nn.Linear(neuronas_ocultas, 1)

    def forward(self, x):
        x = torch.relu(self.oculta(x))
        x = self.salida(x)
        return x

x_tensor = torch.from_numpy(x_normalizado.values)
y_tensor = torch.from_numpy(y_normalizado.values)

def buttonCallback():
    nuevoModelo = entrenar(
        tasa_de_aprendizaje=tasa_de_aprendizaje,
        epocas=num_de_epocas,
        neuronas=neuronas_ocultas,
        x=x_tensor,
        y=y_tensor
    )
    predicciones = nuevoModelo(x_tensor).detach().numpy() * (y.max()[0] - y.min()[0]) + y.min()[0]
    plt.figure()
    plt.plot(x, y, 'bo', label='Datos')
    plt.plot(x, predicciones, 'r-', label='Predicción')
    plt.xlabel('Día')
    plt.ylabel('Ventas')
    plt.legend()
    st.pyplot(plt)

boton_entrenar = st.sidebar.button(
    'Entrenar',
    on_click=buttonCallback
)

