# -*- coding: utf-8 -*-
"""
App Streamlit para predicción de ventas de pantalones STOP JEANS
"""

# ========================
# Importar librerías
# ========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import streamlit as st

# ========================
# Cargar modelo entrenado
# ========================
with open("modelo-TimeSeries.pkl", "rb") as file:
    forecaster_cargado, data_cargada = pickle.load(file)

# ========================
# Interfaz de la aplicación
# ========================
st.title("📈 Estimación de ventas mensuales de pantalón - STOP JEANS")
st.markdown("Esta aplicación permite estimar la **Cantidad** de ventas futuras.")

# ========================
# Inputs del usuario
# ========================
n_periodos = st.slider(
    "Selecciona cuántos meses deseas pronosticar:",
    min_value=1, max_value=24, value=6
)

# ========================
# Botón para generar predicción
# ========================
import altair as alt

if st.button("Generar pronóstico"):
    # Generar predicciones
    y_pred = forecaster_cargado.predict(steps=n_periodos)

    # Crear DataFrame con resultados
    resultado = pd.DataFrame({
        "Fecha": y_pred.index.to_period("M").to_timestamp(),
        "Pronostico_Cantidad": y_pred.values.round(0).astype(int)
    })
    # Columna de texto mes-año
    resultado["Fecha_str"] = resultado["Fecha"].dt.strftime("%b-%Y")

    # Mostrar tabla
    st.subheader("📊 Resultados del pronóstico")
    st.dataframe(resultado[["Fecha_str", "Pronostico_Cantidad"]])

    # Gráfico con Altair
    chart = alt.Chart(resultado).mark_line(point=True).encode(
        x=alt.X("Fecha_str:O", title="Fecha"),   # eje categórico
        y=alt.Y("Pronostico_Cantidad:Q", title="Cantidad"),
        tooltip=["Fecha_str", "Pronostico_Cantidad"]
    ).properties(
        title="Pronóstico de ventas mensuales",
        width=700,
        height=400
    )

    st.altair_chart(chart, use_container_width=True)

