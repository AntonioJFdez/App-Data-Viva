import os
import logging
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import matplotlib.pyplot as plt
import seaborn as sns

# Configura el logger para depuración
logging.basicConfig(level=logging.DEBUG)

# -- CONFIGURACIÓN FLASK --
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey")  # Asegura seguridad básica para mensajes

# --- FUNCIONES UTILITARIAS ---

def cargar_dataset(archivo):
    """
    Carga cualquier CSV o Excel con robustez y devuelve DataFrame limpio.
    """
    try:
        nombre = archivo.filename
        logging.debug(f"Cargando archivo: {nombre}")  # Log del archivo que se está cargando
        if nombre.endswith('.csv'):
            df = pd.read_csv(archivo, encoding='utf-8')
        elif nombre.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(archivo)
        else:
            raise ValueError("Formato de archivo no soportado.")
        if df.empty:
            raise ValueError("El archivo está vacío.")
        logging.debug(f"DataFrame cargado con éxito: {df.head()}")  # Log de las primeras filas del dataframe
        return df
    except Exception as e:
        logging.error(f"Error al cargar los datos: {str(e)}")  # Log de error
        raise ValueError(f"Error al cargar los datos: {str(e)}")

def resultado_mensaje(mensaje, exito=True):
    """
    Devuelve un mensaje amigable y visual para el usuario, ajustado al branding.
    """
    estilo = "color: #2c3e50; background: #debfb0; border-radius: 1rem; padding: 1.3rem; font-size:1.15rem; text-align:center;"
    if not exito:
        estilo = "color: #fff; background: #e74c3c; border-radius: 1rem; padding: 1.3rem; font-size:1.12rem; text-align:center;"
    return f'<div style="{estilo}">{mensaje}</div>'

def generar_grafico(df):
    """
    Genera un gráfico de barras basado en el DataFrame cargado. 
    Aquí puedes personalizar este gráfico según las variables que se pasen.
    """
    try:
        logging.debug("Generando gráfico de barras...")  # Log de generación de gráfico
        # Selección de las primeras columnas numéricas para el gráfico
        columnas_numericas = df.select_dtypes(include='number').columns
        if len(columnas_numericas) > 0:
            df[columnas_numericas].mean().plot(kind='bar', figsize=(8, 6), color='skyblue')
            plt.title('Promedio de Variables Numéricas', fontsize=14)
            plt.ylabel('Promedio', fontsize=12)
            plt.xlabel('Variables', fontsize=12)
            plt.tight_layout()
            grafico_path = "static/graficos/grafico.png"
            plt.savefig(grafico_path)
            plt.close()
            logging.debug(f"Gráfico guardado en: {grafico_path}")  # Log de la ubicación del gráfico generado
            return grafico_path
        else:
            raise ValueError("No hay datos numéricos para graficar.")
    except Exception as e:
        logging.error(f"Error al generar gráfico: {str(e)}")  # Log de error
        raise ValueError(f"Error al generar gráfico: {str(e)}")

# -- RUTAS PRINCIPALES --

@app.route('/', methods=['GET'])
def home():
    """
    Página principal con el resumen de las soluciones.
    """
    logging.debug("Accediendo a la página principal...")  # Log de la página principal
    return render_template('indice.html')

@app.route('/perfilado', methods=['GET', 'POST'])
def perfilado():
    """
    Carga un dataset y realiza el perfilado automático de datos.
    """
    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            logging.debug(f"DataFrame cargado en perfilado: {df.head()}")  # Log del DataFrame en perfilado
            # Generar gráfico
            grafico_path = generar_grafico(df)
            info = df.describe(include='all').transpose().reset_index()
            resumen = info.to_html(classes="table table-striped", index=False)
            flash(resultado_mensaje("¡Perfilado realizado con éxito! Revisa el informe más abajo."), "success")
            return render_template('perfilado.html', resumen=resumen, grafico=grafico_path)
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
            logging.error(f"Error en perfilado: {e}")  # Log del error
    return render_template('perfilado.html')




