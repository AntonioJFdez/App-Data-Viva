# modulos_agricolas/kriging.py

import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import os

def ejecutar_kriging(ruta_csv, variable='ndvi'):
    df = pd.read_csv(ruta_csv)
    OK = OrdinaryKriging(
        df['x'], df['y'], df[variable],
        variogram_model='spherical',
        verbose=False,
        enable_plotting=False
    )
    gridx = np.linspace(df['x'].min(), df['x'].max(), 100)
    gridy = np.linspace(df['y'].min(), df['y'].max(), 100)
    z, ss = OK.execute('grid', gridx, gridy)

    # Guardar gráfico
    out_dir = 'static/graficos'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'kriging_{variable}.png')
    plt.figure(figsize=(8, 6))
    plt.imshow(
        z,
        origin='lower',
        extent=(df['x'].min(), df['x'].max(), df['y'].min(), df['y'].max()),
        cmap='YlGn'
    )
    plt.title(f"Mapa interpolado de {variable.upper()}")
    plt.colorbar(label=variable.upper())
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

  # modulos_agricolas/biomasa3d.py

import laspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import os

def analizar_biomasa_las(ruta_las, grid_size=1):
    las = laspy.read(ruta_las)
    points = np.vstack((las.x, las.y, las.z)).transpose()

    xmin, ymin, zmin = np.min(points, axis=0)
    xmax, ymax, zmax = np.max(points, axis=0)

    x_bins = np.arange(xmin, xmax, grid_size)
    y_bins = np.arange(ymin, ymax, grid_size)

    stat_dsm, _, _, _ = scipy.stats.binned_statistic_2d(
        points[:,0], points[:,1], points[:,2], statistic='max', bins=[x_bins, y_bins])
    stat_dtm, _, _, _ = scipy.stats.binned_statistic_2d(
        points[:,0], points[:,1], points[:,2], statistic='min', bins=[x_bins, y_bins])

    diff = stat_dsm - stat_dtm
    diff[np.isnan(diff)] = 0

    cell_area = grid_size ** 2
    volumen_biomasa = np.sum(diff) * cell_area

    # Guardar gráficos
    out_dir = 'static/graficos'
    os.makedirs(out_dir, exist_ok=True)
    dsm_path = os.path.join(out_dir, 'dsm_parcela.png')
    dtm_path = os.path.join(out_dir, 'dtm_parcela.png')

    plt.imshow(stat_dsm.T, cmap='terrain', origin='lower')
    plt.colorbar(label='Altura (m)')
    plt.title('DSM - Altura máxima')
    plt.tight_layout()
    plt.savefig(dsm_path, dpi=150)
    plt.close()

    plt.imshow(stat_dtm.T, cmap='terrain', origin='lower')
    plt.colorbar(label='Altura Suelo (m)')
    plt.title('DTM - Altura mínima')
    plt.tight_layout()
    plt.savefig(dtm_path, dpi=150)
    plt.close()

    # Guardar resumen en CSV
    resumen_csv = os.path.join(out_dir, 'resumen_volumen_biomasa.csv')
    pd.DataFrame({"volumen_biomasa_m3": [volumen_biomasa]}).to_csv(resumen_csv, index=False)

    return dsm_path, dtm_path, volumen_biomasa

  # modulos_agricolas/salud_cultivo.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def analizar_salud_cultivo(ruta_csv, graficos_dir='static/graficos'):
    df = pd.read_csv(ruta_csv).dropna()
    features = ['ndvi', 'gndvi', 'ndwi', 'bgr', 'red_edge']
    X = df[features]
    y = df['estres']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(graficos_dir, 'modelo_prediccion_estres.pkl'))

    # Boxplot NDVI según estrés
    os.makedirs(graficos_dir, exist_ok=True)
    boxplot_path = os.path.join(graficos_dir, 'boxplot_ndvi_estres.png')
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x='estres', y='ndvi')
    plt.title('Distribución NDVI según Estrés')
    plt.xlabel('Nivel de Estrés')
    plt.ylabel('NDVI')
    plt.tight_layout()
    plt.savefig(boxplot_path)
    plt.close()

    # Tabla resumen de medias por grupo
    resumen = df.groupby('estres')[features].mean().reset_index()
    resumen_path = os.path.join(graficos_dir, 'resumen_salud_cultivo.csv')
    resumen.to_csv(resumen_path, index=False)

    return boxplot_path, resumen_path, model

  # modulos_agricolas/segmentacion_ndvi.py

import numpy as np
import pandas as pd
import rasterio
from sklearn.cluster import KMeans
import geopandas as gpd
import folium
import os

def segmentar_parcela_ndvi(
    ruta_ndvi,
    ruta_parcela,
    n_clusters=4,
    output_dir="static/graficos"
):
    # Cargar NDVI raster
    with rasterio.open(ruta_ndvi) as src:
        ndvi = src.read(1)
        mask = ndvi > 0
        coords = np.array(list(src.xy(*idx) for idx in np.argwhere(mask)))
        valores_ndvi = ndvi[mask].reshape(-1, 1)
        crs = src.crs

    # Cargar límites de parcela
    gdf_parcela = gpd.read_file(ruta_parcela)

    # Datos para clustering
    data_clust = np.hstack([valores_ndvi, coords])
    modelo = KMeans(n_clusters=n_clusters, random_state=42)
    zonas = modelo.fit_predict(data_clust)

    # Crear GeoDataFrame de puntos segmentados
    gdf_zonas = gpd.GeoDataFrame({
        'ndvi': valores_ndvi.flatten(),
        'zona': zonas
    }, geometry=gpd.points_from_xy(coords[:,0], coords[:,1]), crs=crs)

    # Guardar resultados como GeoJSON y CSV
    os.makedirs(output_dir, exist_ok=True)
    geojson_path = os.path.join(output_dir, "zonas_intra_parcela.geojson")
    csv_path = os.path.join(output_dir, "zonas_intra_parcela.csv")
    gdf_zonas.to_file(geojson_path, driver="GeoJSON")
    gdf_zonas[['ndvi', 'zona']].to_csv(csv_path, index=False)

    # Mapa interactivo
    m = folium.Map(
        location=[coords[:,1].mean(), coords[:,0].mean()],
        zoom_start=16
    )
    folium.GeoJson(gdf_parcela.geometry).add_to(m)
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'yellow', 'pink', 'grey']
    for z in range(n_clusters):
        pts = gdf_zonas[gdf_zonas['zona'] == z]
        for _, row in pts.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=2,
                color=colors[z % len(colors)],
                fill=True,
                fill_opacity=0.6
            ).add_to(m)
    mapa_html = os.path.join(output_dir, "mapa_variabilidad_intra_parcela.html")
    m.save(mapa_html)

    # Estadísticas y recomendaciones por zona
    resumen = []
    for zona in range(n_clusters):
        datos = gdf_zonas[gdf_zonas['zona'] == zona]['ndvi']
        media, minv, maxv = datos.mean(), datos.min(), datos.max()
        recomendacion = "Aumentar fertilización y riego" if media < 0.4 else "Mantener manejo actual"
        resumen.append({
            'zona': zona,
            'ndvi_medio': round(media, 3),
            'ndvi_min': round(minv, 3),
            'ndvi_max': round(maxv, 3),
            'recomendacion': recomendacion
        })
    df_recomendaciones = pd.DataFrame(resumen)
    rec_csv_path = os.path.join(output_dir, "recomendaciones_zonas.csv")
    df_recomendaciones.to_csv(rec_csv_path, index=False)

    return mapa_html, rec_csv_path, geojson_path, csv_path

  # modulos_agricolas/visor_parcelas.py

import geopandas as gpd
import folium
import os

def generar_visor_parcelas(ruta_geojson, output_dir="static/graficos"):
    # Cargar el archivo de parcelas (GeoJSON o SHP)
    gdf = gpd.read_file(ruta_geojson)
    # Centrar el mapa en el centroide
    lon = gdf.geometry.centroid.x.mean()
    lat = gdf.geometry.centroid.y.mean()
    mapa = folium.Map(location=[lat, lon], zoom_start=15)

    # Añadir cada parcela con tooltip (nombre, cultivo…)
    for _, row in gdf.iterrows():
        folium.GeoJson(
            row['geometry'],
            name=row.get('nombre', 'Parcela'),
            tooltip=f"Parcela: {row.get('nombre', 'N/A')}<br>Cultivo: {row.get('cultivo', 'N/A')}",
        ).add_to(mapa)

    # Guardar el mapa
    os.makedirs(output_dir, exist_ok=True)
    mapa_html = os.path.join(output_dir, "visor_interactivo_parcelas.html")
    mapa.save(mapa_html)
    return mapa_html

  # modulos_agricolas/optimizacion_riego.py

import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.tree import DecisionTreeRegressor

def entrenar_modelo_riego(ruta_csv, output_dir="static/graficos"):
    df = pd.read_csv(ruta_csv)
    features = ['humedad_suelo', 'temperatura', 'precipitacion', 'evapotranspiracion']
    X = df[features]
    y = df['riego_recomendado_litros']

    # Entrenar modelo
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X, y)
    modelo_path = os.path.join(output_dir, 'modelo_recomendacion_riego.pkl')
    joblib.dump(model, modelo_path)

    # Gráfico evolución de humedad
    if 'fecha' in df.columns:
        plt.plot(df['fecha'], df['humedad_suelo'])
        plt.title('Evolución de Humedad del Suelo')
        plt.xlabel('Fecha')
        plt.ylabel('Humedad (%)')
        plt.tight_layout()
        grafico_path = os.path.join(output_dir, 'panel_humedad.png')
        plt.savefig(grafico_path)
        plt.close()
    else:
        grafico_path = None

    return modelo_path, grafico_path

def predecir_riego(modelo_path, humedad, temperatura, precipitacion, evapotranspiracion):
    model = joblib.load(modelo_path)
    datos = [[humedad, temperatura, precipitacion, evapotranspiracion]]
    return model.predict(datos)[0]

  # modulos_agricolas/deteccion_plagas.py

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import joblib
import os

# Definición de clases ejemplo (personaliza según tu dataset)
NOMBRES_CLASES = ["Sano", "Mildiu", "Oidio", "Araña Roja"]

def cargar_modelo_pytorch(ruta_pesos, num_classes=4):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(ruta_pesos, map_location='cpu'))
    model.eval()
    return model

def predecir_imagen(modelo, ruta_imagen, class_names=NOMBRES_CLASES):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(ruta_imagen).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = modelo(image)
        _, pred = torch.max(output, 1)
    return class_names[pred.item()]

# Entrenamiento batch simplificado (opcional, normalmente se ejecuta por terminal)
def entrenar_red_plagas(carpeta_imgs, labels_csv, pesos_salida="static/graficos/modelo_plagas_enfermedades.pth",
                        num_classes=4, batch_size=16, epochs=10):
    # Para entrenamiento serio, usar Jupyter/terminal, pero se puede lanzar el proceso aquí si hace falta.
    from torch.utils.data import Dataset, DataLoader
    class DroneDataset(Dataset):
        def __init__(self, csv_file, root_dir, transform=None):
            self.labels_frame = pd.read_csv(csv_file)
            self.root_dir = root_dir
            self.transform = transform
        def __len__(self):
            return len(self.labels_frame)
        def __getitem__(self, idx):
            img_name = os.path.join(self.root_dir, self.labels_frame.iloc[idx, 0])
            image = Image.open(img_name).convert('RGB')
            label = int(self.labels_frame.iloc[idx, 1])
            if self.transform:
                image = self.transform(image)
            return image, label
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = DroneDataset(labels_csv, carpeta_imgs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    torch.save(model.state_dict(), pesos_salida)
    return pesos_salida

  # modulos_agricolas/monitorizacion_fenologica.py

import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

def analizar_evolucion_fenologica(carpeta_ndvi, umbral_brote=0.08):
    """Procesa imágenes NDVI temporales y detecta brotes significativos"""
    fechas = sorted(os.listdir(carpeta_ndvi))
    datos_temporales = []
    for fecha in fechas:
        carpeta_fecha = os.path.join(carpeta_ndvi, fecha)
        if not os.path.isdir(carpeta_fecha):
            continue
        archivos = [f for f in os.listdir(carpeta_fecha) if f.endswith('_NDVI.tif')]
        for archivo in archivos:
            ruta = os.path.join(carpeta_fecha, archivo)
            with rasterio.open(ruta) as src:
                ndvi = src.read(1)
                mask = ndvi > 0
                ndvi_medio = np.mean(ndvi[mask]) if np.any(mask) else 0
                datos_temporales.append({
                    'fecha': fecha,
                    'parcela': archivo.replace('_NDVI.tif', ''),
                    'ndvi_medio': ndvi_medio
                })
    if not datos_temporales:
        return None, None, None
    df = pd.DataFrame(datos_temporales)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values(['parcela', 'fecha'])
    df['ndvi_delta'] = df.groupby('parcela')['ndvi_medio'].diff()
    df['brote'] = df['ndvi_delta'] > umbral_brote
    # Exporta CSVs (opcional)
    df.to_csv('static/graficos/evolucion_ndvi_temporal.csv', index=False)
    df_brotes = df[df['brote']]
    df_brotes.to_csv('static/graficos/brotes_detectados.csv', index=False)
    # Genera y guarda gráficos para cada parcela
    parcelas = df['parcela'].unique()
    img_paths = []
    for parcela in parcelas:
        dfp = df[df['parcela'] == parcela]
        plt.figure(figsize=(8,3))
        plt.plot(dfp['fecha'], dfp['ndvi_medio'], marker='o', label='NDVI medio')
        plt.scatter(dfp[dfp['brote']]['fecha'], dfp[dfp['brote']]['ndvi_medio'],
                    color='red', label='Brote detectado')
        plt.title(f'Evolución Fenológica - {parcela}')
        plt.xlabel('Fecha')
        plt.ylabel('NDVI medio')
        plt.legend()
        plt.tight_layout()
        img_path = f'static/graficos/evolucion_fenologica_{parcela}.png'
        plt.savefig(img_path)
        img_paths.append((parcela, img_path))
        plt.close()
    return df, df_brotes, img_paths

  # modulos_agricolas/indices_vegetacion.py

import numpy as np
import pandas as pd
import rasterio
import glob
import os
import matplotlib.pyplot as plt

def calcular_indices_vegetacion(carpeta_datos):
    """Procesa imágenes multiespectrales, calcula NDVI/GNDVI/SAVI y visualiza la evolución."""
    campanias = sorted(os.listdir(carpeta_datos))
    resultados = []
    img_indices = []
    for campania in campanias:
        archivos_tif = glob.glob(os.path.join(carpeta_datos, campania, "*.tif"))
        for ruta_tif in archivos_tif:
            try:
                with rasterio.open(ruta_tif) as src:
                    red = src.read(1).astype('float32')
                    green = src.read(2).astype('float32')
                    nir = src.read(4).astype('float32')
                    ndvi = np.where((nir + red) == 0, 0, (nir - red) / (nir + red))
                    gndvi = np.where((nir + green) == 0, 0, (nir - green) / (nir + green))
                    savi = np.where((nir + red + 0.5) == 0, 0, ((nir - red) / (nir + red + 0.5)) * 1.5)
            except Exception as e:
                continue
            resultados.append({
                "campania": campania,
                "archivo": os.path.basename(ruta_tif),
                "NDVI_medio": np.nanmean(ndvi),
                "GNDVI_medio": np.nanmean(gndvi),
                "SAVI_medio": np.nanmean(savi)
            })
            # (Opcional) Exporta NDVI como ráster para GIS
            out_raster = ruta_tif.replace(".tif", "_NDVI.tif")
            with rasterio.open(ruta_tif) as src:
                with rasterio.open(
                    out_raster, 'w',
                    driver='GTiff',
                    height=ndvi.shape[0], width=ndvi.shape[1],
                    count=1, dtype='float32',
                    crs=src.crs, transform=src.transform
                ) as dst:
                    dst.write(ndvi, 1)
    if not resultados:
        return None, None
    df_indices = pd.DataFrame(resultados)
    # Gráfico de evolución temporal de índices
    plt.figure(figsize=(12, 6))
    for idx in ["NDVI_medio", "GNDVI_medio", "SAVI_medio"]:
        plt.plot(df_indices["campania"], df_indices[idx], marker="o", label=idx)
    plt.title("Evolución histórica de índices vegetativos")
    plt.xlabel("Campaña / Fecha")
    plt.ylabel("Valor medio del índice")
    plt.legend()
    plt.tight_layout()
    grafico = 'static/graficos/evolucion_indices_vegetacion.png'
    plt.savefig(grafico)
    plt.close()
    img_indices.append(grafico)
    # Exporta resumen
    df_indices.to_csv("static/graficos/resumen_indices_vegetacion.csv", index=False)
    return df_indices, img_indices

  # modulos_agricolas/estres_hidrico_termico.py

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import folium
import os

def detectar_estres_hidrico(
    ruta_termica='imagenes_termicas/parcela1_2024-05-20.tif',
    ruta_parcelas='parcelas.geojson',
    limite_critico=30.0,
    cwsi_critico=0.7
):
    # Procesamiento de imagen térmica
    with rasterio.open(ruta_termica) as src:
        termica = src.read(1).astype('float32')
        bounds = src.bounds
        mask = termica > 0

    # Cálculo del índice de estrés hídrico (CWSI)
    Tmin = np.percentile(termica[mask], 5)
    Tmax = np.percentile(termica[mask], 95)
    cwsi = (termica - Tmin) / (Tmax - Tmin)
    cwsi = np.clip(cwsi, 0, 1)

    # Zonas críticas
    zonas_criticas = (termica >= limite_critico) | (cwsi >= cwsi_critico)
    porcentaje_critico = 100 * np.sum(zonas_criticas) / np.sum(mask)

    # Exporta coordenadas de zonas críticas
    fila, columna = np.where(zonas_criticas)
    coordenadas = [src.xy(f, c) for f, c in zip(fila, columna)]
    df_alertas = pd.DataFrame(coordenadas, columns=["lat", "lon"])
    alertas_csv = "static/outputs/alertas_estres_hidrico.csv"
    df_alertas.to_csv(alertas_csv, index=False)

    # Visualización en mapa interactivo
    gdf_parcelas = gpd.read_file(ruta_parcelas)
    m = folium.Map(
        location=[(bounds.top + bounds.bottom)/2, (bounds.left + bounds.right)/2],
        zoom_start=16
    )
    folium.GeoJson(gdf_parcelas.geometry).add_to(m)
    for _, row in df_alertas.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3,
            color='red',
            fill=True,
            fill_opacity=0.8
        ).add_to(m)
    mapa_html = "static/outputs/mapa_estres_hidrico.html"
    m.save(mapa_html)

    # Resumen por parcela
    resumen = []
    for parcela in gdf_parcelas.itertuples():
        puntos_criticos = df_alertas[df_alertas.apply(
            lambda x: parcela.geometry.contains(gpd.points_from_xy([x.lon], [x.lat])[0]), axis=1)]
        area_critica = len(puntos_criticos)
        resumen.append({
            'parcela': getattr(parcela, 'nombre', parcela.Index),
            'num_alertas': area_critica,
            'alerta': 'CRÍTICO' if area_critica > 0 else 'OK'
        })
    df_resumen = pd.DataFrame(resumen)
    resumen_csv = "static/outputs/resumen_alertas_estres_hidrico.csv"
    df_resumen.to_csv(resumen_csv, index=False)

    return {
        "porcentaje_critico": round(porcentaje_critico, 2),
        "alertas_csv": alertas_csv,
        "mapa_html": mapa_html.replace("static/", ""),
        "resumen_csv": resumen_csv
    }

  # modulos_agricolas/prediccion_rendimiento.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

def predecir_rendimiento(
    archivo_datos='datos_rendimiento_parcelas.csv',
    columnas_features=['ndvi_medio', 'precipitacion', 'temperatura', 'superficie'],
    columna_objetivo='rendimiento_t_ha'
):
    # 1. Carga de datos
    df = pd.read_csv(archivo_datos)

    # 2. Selección de variables
    X = df[columnas_features]
    y = df[columna_objetivo]

    # 3. Entrenamiento del modelo
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "static/outputs/modelo_prediccion_rendimiento.pkl")

    # 4. Predicción y generación de informe
    df['prediccion'] = model.predict(X)
    informe = df[['parcela', columna_objetivo, 'prediccion']]
    output_excel = "static/outputs/informe_rendimiento_parcelas.xlsx"
    informe.to_excel(output_excel, index=False)

    # 5. Métrica opcional
    error = abs(df[columna_objetivo] - df['prediccion']).mean()

    return {
        "ruta_excel": output_excel.replace("static/", ""),
        "error_medio": round(error, 3),
        "num_parcelas": len(df['parcela'].unique())
    }

  # modulos_agricolas/prescripcion_manejo.py

import pandas as pd
import geopandas as gpd
import folium
import os

def prescribir_manejo(
    archivo_csv='zonas_analizadas.csv',
    output_csv='static/outputs/prescripcion_final.csv',
    output_geojson='static/outputs/prescripciones.geojson',
    output_mapa='static/outputs/mapa_prescripcion_nutricional.html'
):
    # 1. Cargar datos
    df = pd.read_csv(archivo_csv)

    # 2. Prescripción general
    def prescripcion_general(row):
        if row['ndvi'] < 0.4 or row['humedad'] < 0.25:
            return "Alta fertilización + Riego extra"
        elif row['ndvi'] < 0.6:
            return "Fertilización media + Riego ajustado"
        else:
            return "Mantenimiento"
    df['prescripcion_manejo'] = df.apply(prescripcion_general, axis=1)

    # 3. Fertilización específica NPK
    df['fertilizacion_n'] = df['n'].apply(lambda x: 'Alta' if x < 2 else 'Baja')
    df['fertilizacion_p'] = df['p'].apply(lambda x: 'Alta' if x < 1 else 'Baja')
    df['fertilizacion_k'] = df['k'].apply(lambda x: 'Alta' if x < 3 else 'Baja')

    # 4. Exportación a CSV
    df.to_csv(output_csv, index=False)

    # 5. GeoJSON si hay geometría
    if 'geometry' in df.columns:
        gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['geometry']), crs="EPSG:25830")
        gdf.to_file(output_geojson, driver='GeoJSON')

    # 6. Mapa interactivo
    lat_centro = df['lat'].mean() if 'lat' in df.columns else 40.0
    lon_centro = df['lon'].mean() if 'lon' in df.columns else -3.0
    m = folium.Map(location=[lat_centro, lon_centro], zoom_start=15)
    for _, row in df.iterrows():
        popup_text = (
            f"Zona: {row.get('zona','N/A')}<br>"
            f"Prescripción manejo: {row.get('prescripcion_manejo','N/A')}<br>"
            f"Fertilización NPK: N={row.get('fertilizacion_n','')}, "
            f"P={row.get('fertilizacion_p','')}, K={row.get('fertilizacion_k','')}"
        )
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=6,
            popup=popup_text,
            color='blue',
            fill=True,
            fill_opacity=0.7
        ).add_to(m)
    m.save(output_mapa)

    return {
        "output_csv": output_csv.replace("static/", ""),
        "output_geojson": output_geojson.replace("static/", "") if 'geometry' in df.columns else None,
        "output_mapa": output_mapa.replace("static/", "")
    }

  # modulos_agricolas/water_stress_ml.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

def predecir_estres_hidrico(
    archivo_csv='dataset_estres_hidrico_multianual.csv',
    output_csv='static/outputs/alertas_predictivas_estres.csv'
):
    # 1. Cargar datos
    df = pd.read_csv(archivo_csv)
    features = ['termico_medio', 'ndvi', 'temp_aire', 'humedad', 'precipitacion']
    X = df[features]
    y = df['etiqueta_estres']

    # 2. Entrenar modelo Random Forest (entrenamiento simple)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 3. Predicción y filtrado de zonas con estrés
    df['prediccion_estres'] = model.predict(X)
    df_criticas = df[df['prediccion_estres'] == 1]
    df_criticas.to_csv(output_csv, index=False)

    return {
        "output_csv": output_csv.replace("static/", ""),
        "num_alertas": len(df_criticas)
    }

  # modulos_agricolas/yield_prediction_analytics.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

def predecir_rendimiento_parcelas(
    archivo_csv='datos_rendimiento_parcelas.csv',
    output_excel='static/outputs/informe_rendimiento_parcelas.xlsx',
    modelo_pkl='static/modelos/modelo_prediccion_rendimiento.pkl'
):
    # 1. Cargar datos
    df = pd.read_csv(archivo_csv)
    features = ['ndvi_medio', 'precipitacion', 'temperatura', 'superficie']
    X = df[features]
    y = df['rendimiento_t_ha']

    # 2. Entrenamiento del modelo
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, modelo_pkl)

    # 3. Predicción y exportación de informe
    df['prediccion'] = model.predict(X)
    informe = df[['parcela', 'rendimiento_t_ha', 'prediccion']]
    informe.to_excel(output_excel, index=False)

    return {
        "output_excel": output_excel.replace("static/", ""),
        "modelo_pkl": modelo_pkl.replace("static/", ""),
        "num_registros": len(df)
    }
