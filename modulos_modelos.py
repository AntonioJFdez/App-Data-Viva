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

# modulos_modelos.py (scoring_leads)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def scoring_leads(df, target_col='convertido', output_dir="static/graficos"):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    reporte = classification_report(y_test, y_pred, output_dict=True)

    # Aplicar scoring a todo el DataFrame
    df['score'] = modelo.predict_proba(X)[:, 1]
    df_ordenado = df.sort_values(by='score', ascending=False)

    # Exportar resultados
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_csv = os.path.join(output_dir, "leads_score.csv")
    output_xlsx = os.path.join(output_dir, "leads_score.xlsx")
    df_ordenado.to_csv(output_csv, index=False)
    try:
        df_ordenado.to_excel(output_xlsx, index=False)
    except Exception as e:
        pass

    # Guardar modelo
    modelo_path = os.path.join(output_dir, "modelo_scoring_leads.pkl")
    joblib.dump(modelo, modelo_path)
    return output_csv, output_xlsx, reporte

