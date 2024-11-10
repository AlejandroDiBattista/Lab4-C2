import streamlit as st
import pandas as pd
import numpy as np
import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt


## Crear Red Neuronal
## Leer Datos
## Normalizar Datos
## Entrenar Red Neuronal
## Guardar Modelo
## Graficar Predicciones
st.title('Estimación de Ventas Diarias :')
st.sidebar.header('Subir datos')

#Leer datos
uploadfile = st.sidebar.file_uploader('Subir archivo de datos', type=['csv'])

if uploadfile is not None:
    data = pd.read_csv(uploadfile)
    tabla = pd.DataFrame(data)
    
    st.text('Ventas de los ultimos 30 días:')
    st.write(tabla)

    tabla_normalizada = tabla.copy()
 
 
    # Creación de Tensores
    tabla_normalizada["dia"] = (tabla_normalizada["dia"] - tabla_normalizada["dia"].min()) / (tabla_normalizada["dia"].max() - tabla_normalizada["dia"].min())                                                         
    tabla_normalizada["ventas"] = (tabla_normalizada["ventas"] - tabla_normalizada["ventas"].min()) / (tabla_normalizada["ventas"].max() - tabla_normalizada["ventas"].min())  
    
    X = t.tensor(tabla_normalizada["dia"].values, dtype=t.float32).view(-1, 1)  
    Y = t.tensor(tabla_normalizada["ventas"].values, dtype=t.float32).view(-1, 1) 
    
    
    # Creación de la red neuronal
    class RedNeuronal(nn.Module):
        def __init__(self):
            super(RedNeuronal, self).__init__()
            self.capa1 = nn.Linear(1, 10)  
            self.capa2 = nn.Linear(10, 1) 

        def forward(self, x):
            x = t.relu(self.capa1(x))  
            x = self.capa2(x) 
            return x
    
    
    # Crear el modelo
    modelo = RedNeuronal()

    criterio = nn.MSELoss() 
    optimizador = t.optim.Adam(modelo.parameters(), lr=0.01) 

    num_epocas = 1000
    
    for epoca in range(num_epocas):

        predicciones = modelo(X)
        
        perdida = criterio(predicciones, Y)

        optimizador.zero_grad()


        perdida.backward()

        optimizador.step()

        if epoca % 100 == 0:
                st.write(f"Época {epoca}/{num_epocas} - Pérdida: {perdida.item():.4f}")

    modelo.eval()
    predicciones_finales = modelo(X).detach().numpy() 

    # Grafica
    st.title("Gráfico")
    plt.figure(figsize=(10, 6))
    plt.plot(tabla_normalizada["dia"], Y.numpy(), label="Ventas reales", color='blue', marker='o')
    plt.plot(tabla_normalizada["dia"], predicciones_finales, label="Ventas predichas", color='red', linestyle='--')
    plt.xlabel("Día (normalizado)")
    plt.ylabel("Ventas (normalizadas)")
    plt.title("Estimación de Ventas Diarias")
    plt.legend()
    st.pyplot(plt)
