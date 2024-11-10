import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

st.title('Estimación de Ventas Diarias')

# Subir archivo de ventas
uploaded_file = st.file_uploader("Sube el archivo CSV con los datos de ventas", type=["csv"])

if uploaded_file is not None:
  
    df = pd.read_csv(uploaded_file)
    
    df.columns = df.columns.str.strip()

    # Verificar que las columnas necesarias están presentes
    if 'Dia' not in df.columns or 'Ventas' not in df.columns:
        st.error('El archivo CSV debe contener las columnas "Dia" y "Ventas".')
    else:
        st.write(df.head())  

       
        if df.isnull().any().any():
            st.warning('El archivo contiene valores nulos. Asegúrese de limpiarlos antes de proceder.')
            st.write(df[df.isnull().any(axis=1)])  
      
        learning_rate = st.slider("Tasa de aprendizaje", 0.0, 1.0, 0.1)
        epochs = st.slider("Cantidad de épocas", 10, 10000, 100)
        hidden_neurons = st.slider("Cantidad de neuronas en la capa oculta", 1, 100, 5)

        # Función para crear la red neuronal
        class SalesPredictor(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(SalesPredictor, self).__init__()
                self.hidden = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.output = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                x = self.relu(self.hidden(x))
                x = self.output(x)
                return x

     
        X = df['Dia'].values.reshape(-1, 1)
        y = df['Ventas'].values

        # Normalización de los datos
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

        # Convertir los datos a tensores
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # Crear el modelo
        model = SalesPredictor(input_size=1, hidden_size=hidden_neurons, output_size=1)

        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Entrenamiento del modelo
        if st.button('Entrenar'):
            epochs_list = []
            loss_list = []
            with st.spinner('Entrenando...'):
                for epoch in range(epochs):
              
                    model.train()
                    y_pred = model(X_train_tensor)

                   
                    loss = criterion(y_pred, y_train_tensor)
                    loss_list.append(loss.item())
                    epochs_list.append(epoch)

                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                   
                    if (epoch + 1) % 50 == 0:
                        st.progress((epoch + 1) / epochs)

            # Mostrar mensaje de éxito
            st.success('Entrenamiento completado con éxito.')

            # Graficar la evolución de la función de costo
            fig, ax = plt.subplots()
            ax.plot(epochs_list, loss_list, label='Función de Costo', color='red')
            ax.set_xlabel('Épocas')
            ax.set_ylabel('Pérdida')
            ax.set_title('Evolución de la Función de Costo')
            ax.grid(True)  
            st.pyplot(fig)

 # Predicción sobre los datos de prueba
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor)

# Desnormalizar las predicciones
y_test_pred = scaler_y.inverse_transform(y_test_pred.numpy())
y_test_actual = scaler_y.inverse_transform(y_test_tensor.numpy())

# Tomar solo los días correspondientes a los datos de prueba
test_days = df['Dia'].iloc[X_train.shape[0]:X_train.shape[0] + X_test.shape[0]].reset_index(drop=True)

# Graficar las ventas reales vs las predicciones
fig, ax = plt.subplots()

# Graficar las ventas reales para todo el DataFrame
ax.plot(df['Dia'], df['Ventas'], label='Ventas Reales', color='blue')  

# Graficar las predicciones sobre los datos de prueba
ax.plot(test_days, y_test_pred, label='Predicción', color='red')  

ax.set_xlabel('Día del Mes')
ax.set_ylabel('Ventas')
ax.set_title('Ventas Reales vs Predicción')
ax.legend()
ax.grid(True)  
st.pyplot(fig)
