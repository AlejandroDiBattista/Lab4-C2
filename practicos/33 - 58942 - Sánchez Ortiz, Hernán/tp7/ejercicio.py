import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#Configurar interfaz
st.title('Estimación de Ventas Diarias')
st.sidebar.header('Parametros de Entrenamiento')
tasaAprendizaje = st.sidebar.slider('Tasa de Aprendizaje', 0.0, 1.0, 0.1, 0.01)
repeticiones = st.sidebar.slider('Repeticiones', 10, 10000, 100)
neuronasCapaOculta = st.sidebar.slider('Neuronas Capa Oculta', 1, 100, 5)
botonEntrenar = st.sidebar.button('Entrenar')

#Cargar y preparar datos
datos = pd.read_csv('./ventas.csv')
dias = torch.tensor(datos['dia'].values, dtype=torch.float32).view(-1, 1)
ventas = torch.tensor(datos['ventas'].values, dtype=torch.float32).view(-1, 1)

#Normalizacion
diasNormalizado = (dias - dias.min()) / (dias.max() - dias.min())
ventasNormalizado = (ventas - ventas.min()) / (ventas.max() - ventas.min())

#Definicion de la red neuronal
class RedVentas(nn.Module):
    def __init__(self, neuronasCapaOculta):
        super(RedVentas, self).__init__()
        self.oculta = nn.Linear(1, neuronasCapaOculta)
        self.salida = nn.Linear(neuronasCapaOculta, 1)
    def forward(self, x):
        x = torch.relu(self.oculta(x))
        x = self.salida(x)
        return x

#Entrenamiento de la red neuronal
if botonEntrenar:
    modelo = RedVentas(neuronasCapaOculta)
    criterio = nn.MSELoss()
    optimizador = optim.SGD(modelo.parameters(), lr=tasaAprendizaje)
    barraProgreso = st.progress(0, "...")
    errores = []
    mensajeEpoca = st.empty()
    for epoca in range(repeticiones):
        optimizador.zero_grad()
        predicciones = modelo(diasNormalizado)
        perdida = criterio(predicciones, ventasNormalizado)
        perdida.backward()
        optimizador.step()
        errores.append(perdida.item())

        #Actualizacion de la barra de progreso
        barraProgreso.progress((epoca+1) / repeticiones)
        mensajeEpoca.text(f"Epoca {epoca+1}/{repeticiones} - Error: {perdida.item():.5f}")
    st.success('Entrenamiento exitoso')

    #Grafica de perdida
    plt.figure(figsize=(9, 4))
    plt.plot(range(repeticiones), errores, color='green', label='Perdida')
    plt.xlabel('Repeticiones')
    plt.ylabel('Perdida')
    plt.legend()
    st.pyplot(plt)

    #Prediccion y visualizacion
    predicciones = modelo(diasNormalizado).detach() * (ventas.max() - ventas.min()) + ventas.min() #Desnormalizar
    plt.figure(figsize=(8, 4))
    plt.scatter(datos['dia'], datos['ventas'], color='blue', label='Datos Reales')
    plt.plot(datos['dia'], predicciones.numpy(), color='red', label='Curva de Ajuste')
    plt.xlabel('Dia del Mes')
    plt.ylabel('Ventas')
    plt.legend()
    st.pyplot(plt)