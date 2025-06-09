# modulos_modelos.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

def segmentacion_clientes(df, columnas_segmento, n_clusters=4, output_dir="static/graficos"):
    # Preprocesamiento
    df_segmento = df[columnas_segmento].copy()
    df_segmento.fillna(df_segmento.mean(), inplace=True)
    scaler = StandardScaler()
    datos_normalizados = scaler.fit_transform(df_segmento)

    # KMeans clustering
    modelo = KMeans(n_clusters=n_clusters, random_state=42)
    etiquetas = modelo.fit_predict(datos_normalizados)

    # Visualización con PCA
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(datos_normalizados)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    grafico_path = os.path.join(output_dir, "segmentacion_clientes.png")
    plt.figure(figsize=(8, 6))
    plt.scatter(componentes[:, 0], componentes[:, 1], c=etiquetas, cmap='viridis', alpha=0.6)
    plt.title("Segmentación de Clientes - Clustering K-Means")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(grafico_path)
    plt.close()

    # Devuelve el DataFrame original con segmento y la ruta del gráfico generado
    df['segmento'] = etiquetas
    return df, grafico_path
