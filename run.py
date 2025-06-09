import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import matplotlib.pyplot as plt
import seaborn as sns
from modulos_modelos import segmentacion_clientes
import os
from modulos_modelos import scoring_leads
from modulos_modelos import prediccion_churn
from modulos_modelos import panel_rendimiento_agentes
from modulos_modelos import pricing_dinamico
from modulos_modelos import marketing_personalizado
from modulos_modelos import fidelizacion_clientes
from modulos_modelos import analisis_sentimiento_nps
from modulos_modelos import analizar_siniestros
from modulos_modelos import priorizar_clientes_dormidos
from modulos_modelos import digitalizar_renovaciones
from modulos_modelos import calcular_panel_kpis
from flask import Flask, render_template, request, flash
from modulos_agricolas.kriging import ejecutar_kriging
import os
from modulos_agricolas.biomasa3d import analizar_biomasa_las
import os
from modulos_agricolas.salud_cultivo import analizar_salud_cultivo
import os
from modulos_agricolas.segmentacion_ndvi import segmentar_parcela_ndvi
from modulos_agricolas.visor_parcelas import generar_visor_parcelas
from modulos_agricolas.optimizacion_riego import entrenar_modelo_riego, predecir_riego
from modulos_agricolas.deteccion_plagas import cargar_modelo_pytorch, predecir_imagen, entrenar_red_plagas, NOMBRES_CLASES
from modulos_agricolas.monitorizacion_fenologica import analizar_evolucion_fenologica
from modulos_agricolas.indices_vegetacion import calcular_indices_vegetacion
from modulos_agricolas.estres_hidrico_termico import detectar_estres_hidrico
from modulos_agricolas.prediccion_rendimiento import predecir_rendimiento
from modulos_agricolas.prescripcion_manejo import prescribir_manejo
from modulos_agricolas.water_stress_ml import predecir_estres_hidrico
from modulos_agricolas.yield_prediction_analytics import predecir_rendimiento_parcelas


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
        if nombre.endswith('.csv'):
            df = pd.read_csv(archivo, encoding='utf-8')
        elif nombre.endswith(('.xlsx', '.xls')):  # Maneja Excel también
            df = pd.read_excel(archivo)
        else:
            raise ValueError("Formato de archivo no soportado.")
        if df.empty:
            raise ValueError("El archivo está vacío.")
        return df
    except Exception as e:
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
            return grafico_path
        else:
            raise ValueError("No hay datos numéricos para graficar.")
    except Exception as e:
        raise ValueError(f"Error al generar gráfico: {str(e)}")

# -- RUTAS PRINCIPALES --

@app.route('/', methods=['GET'])
def home():
    """
    Página principal con el resumen de las soluciones.
    """
    return render_template('index.html')

@app.route('/perfilado', methods=['GET', 'POST'])
def perfilado():
    """
    Carga un dataset y realiza el perfilado automático de datos.
    """
    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            # Generar gráfico
            grafico_path = generar_grafico(df)
            info = df.describe(include='all').transpose().reset_index()
            resumen = info.to_html(classes="table table-striped", index=False)
            flash(resultado_mensaje("¡Perfilado realizado con éxito! Revisa el informe más abajo."), "success")
            return render_template('perfilado.html', resumen=resumen, grafico=grafico_path)
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('perfilado.html')

@app.route('/smartfilter', methods=['GET', 'POST'])
def smartfilter():
    """
    Realiza el filtrado inteligente de los datos (ajusta esta funcionalidad).
    """
    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            # Filtrado inteligente aquí, ejemplo: filtrar columnas con NaN o valores bajos
            df_clean = df.dropna()  # Simple ejemplo de eliminación de NaN
            flash(resultado_mensaje("¡Filtrado inteligente realizado con éxito!"), "success")
            return render_template('smartfilter.html', df=df_clean.head())
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('smartfilter.html')

@app.route('/prediccion', methods=['GET', 'POST'])
def prediccion():
    """
    Predicción automática de KPIs usando Random Forest.
    """
    if request.method == 'POST':
        archivo = request.files.get('data_kpi')
        objetivo = request.form.get('target')
        try:
            df = cargar_dataset(archivo)
            # Aquí va el modelo de predicción, por ejemplo con Random Forest
            # Predicción ejemplo: df[objetivo].mean() (esto es solo un ejemplo)
            prediccion = df[objetivo].mean()  # Usar modelo real aquí
            flash(resultado_mensaje(f"Predicción realizada para la variable {objetivo}. Resultado: {prediccion}", "success"))
            return render_template('prediccion.html', prediccion=prediccion)
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('prediccion.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    """
    Crear un dashboard exploratorio basado en los datos cargados.
    """
    if request.method == 'POST':
        archivo = request.files.get('dashboard_data')
        try:
            df = cargar_dataset(archivo)
            # Aquí agregarías el código de dashboard interactivo
            flash(resultado_mensaje("Dashboard generado con éxito. Visualiza el gráfico abajo."), "success")
            return render_template('dashboard.html', df=df.head())
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('dashboard.html')

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    """
    Endpoint de comprobación de salud para garantizar que la aplicación está funcionando.
    """
    return "OK", 200

# -- OTRAS RUTAS --

@app.route('/descargar', methods=['POST'])
def descargar():
    """
    Ruta para descargar archivos generados o reportes.
    """
    archivo_path = request.form.get('file_path')
    if archivo_path and os.path.exists(archivo_path):
        return send_file(archivo_path, as_attachment=True)
    else:
        flash(resultado_mensaje("El archivo no existe o no se puede descargar.", exito=False), "danger")
        return redirect(url_for('home'))

@app.route('/segmentacion', methods=['GET', 'POST'])
def segmentacion():
    """
    Segmentación avanzada de clientes (K-Means)
    """
    columnas_por_defecto = ['edad', 'num_pólizas', 'prima_anual', 'siniestros_12m']
    if request.method == 'POST':
        archivo = request.files.get('dataset')
        columnas = request.form.get('columnas')  # Esto será un string tipo 'edad,num_pólizas'
        n_clusters = int(request.form.get('n_clusters', 4))
        try:
            df = cargar_dataset(archivo)
            columnas = [c.strip() for c in (columnas.split(',') if columnas else columnas_por_defecto)]
            df_segmentado, grafico_path = segmentacion_clientes(df, columnas, n_clusters)
            # Exporta resultados
            output_csv = os.path.join('static', 'graficos', 'clientes_segmentados.csv')
            df_segmentado.to_csv(output_csv, index=False)
            flash(resultado_mensaje("¡Segmentación realizada con éxito! Descarga el resultado y visualiza el gráfico."), "success")
            return render_template(
                'segmentacion.html',
                grafico=grafico_path,
                descargar_csv=output_csv
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('segmentacion.html')

    @app.route('/scoring-leads', methods=['GET', 'POST'])
def scoring_leads_view():
    """
    Scoring y priorización de leads con Random Forest.
    """
    if request.method == 'POST':
        archivo = request.files.get('dataset')
        target_col = request.form.get('target_col', 'convertido')
        try:
            df = cargar_dataset(archivo)
            output_csv, output_xlsx, reporte = scoring_leads(df, target_col=target_col)
            flash(resultado_mensaje("¡Scoring de leads realizado con éxito! Descarga el ranking y consulta el reporte."), "success")
            return render_template(
                'scoring_leads.html',
                descargar_csv=output_csv,
                descargar_xlsx=output_xlsx,
                reporte=reporte
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('scoring_leads.html')

    @app.route('/churn', methods=['GET', 'POST'])
def churn_view():
    """
    Predicción de churn de clientes (cancelación) usando Random Forest.
    """
    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            columnas = request.form.get('columnas')
            # Puedes permitir que el usuario indique las columnas, por defecto las que indica la función
            if columnas:
                columnas = [c.strip() for c in columnas.split(',')]
            reporte_csv, reporte_xlsx, predicciones_csv, reporte_dict = prediccion_churn(df, columnas=columnas)
            flash(resultado_mensaje("Predicción de churn realizada con éxito. Descarga los resultados y revisa el reporte."), "success")
            return render_template(
                'churn.html',
                reporte_csv=reporte_csv,
                reporte_xlsx=reporte_xlsx,
                predicciones_csv=predicciones_csv,
                reporte=reporte_dict
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('churn.html')

    @app.route('/panel-agentes', methods=['GET', 'POST'])
def panel_agentes_view():
    """
    Panel 360° de rendimiento de agentes.
    """
    resumen = None
    panel_csv = None
    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            panel_csv, resumen = panel_rendimiento_agentes(df)
            flash(resultado_mensaje("Panel generado con éxito. Descarga los resultados o revisa el resumen abajo."), "success")
            # Mostrar tabla HTML resumida (las primeras filas)
            resumen_html = resumen.head().to_html(classes="table table-striped", index=False)
            return render_template(
                'panel_agentes.html',
                panel_csv=panel_csv,
                resumen_html=resumen_html
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('panel_agentes.html')

    @app.route('/pricing', methods=['GET', 'POST'])
def pricing_view():
    """
    Pricing dinámico y simulación de escenarios.
    """
    resultados = None
    sim_df = None
    csv_path = None
    graf_path = None

    if request.method == 'POST':
        archivo = request.files.get('dataset')
        margen = float(request.form.get('margen', 0.25))
        try:
            df = cargar_dataset(archivo)
            csv_path, graf_path, resultados, sim_df = pricing_dinamico(df, margen_deseado=margen)
            flash(resultado_mensaje("Cálculo de precios y simulación realizados con éxito. Descarga los resultados o revisa la gráfica."), "success")
            resultados_html = resultados.head().to_html(classes="table table-striped", index=False)
            sim_html = sim_df.to_html(classes="table table-bordered", index=False)
            return render_template(
                'pricing.html',
                resultados_html=resultados_html,
                sim_html=sim_html,
                csv_path=csv_path,
                graf_path=graf_path
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('pricing.html')

    @app.route('/marketing', methods=['GET', 'POST'])
def marketing_view():
    """
    Marketing personalizado 1:1 para campañas automáticas de clientes.
    """
    mensajes_html = None
    output_path = None

    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            # Es importante convertir fechas si no vienen en formato datetime64
            for col in ['fecha_nacimiento', 'fecha_renovacion', 'fecha_siniestro']:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            df_mensajes, output_path = marketing_personalizado(df)
            flash(resultado_mensaje("Mensajes generados correctamente. Descarga el archivo o revisa los mensajes abajo."), "success")
            mensajes_html = df_mensajes.head(10).to_html(classes="table table-striped", index=False)
            return render_template(
                'marketing.html',
                mensajes_html=mensajes_html,
                output_path=output_path
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('marketing.html')

    @app.route('/fidelizacion', methods=['GET', 'POST'])
def fidelizacion_view():
    """
    Calcula el programa de fidelidad y segmenta los clientes.
    """
    tabla_html = None
    output_path = None

    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            # Convertir fecha_alta si no está en formato datetime
            df['fecha_alta'] = pd.to_datetime(df['fecha_alta'], errors='coerce')
            df_resultado, output_path = fidelizacion_clientes(df)
            flash(resultado_mensaje("Programa de fidelidad calculado y clientes segmentados correctamente."), "success")
            tabla_html = df_resultado.head(10).to_html(classes="table table-striped", index=False)
            return render_template(
                'fidelizacion.html',
                tabla_html=tabla_html,
                output_path=output_path
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('fidelizacion.html')

    @app.route('/sentimiento', methods=['GET', 'POST'])
def sentimiento_view():
    """
    Analiza sentimiento de comentarios y calcula NPS.
    """
    tabla_html = None
    nps_score = None
    csv_out = None

    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            df_resultado, nps_score, csv_out, txt_out = analisis_sentimiento_nps(df)
            flash(resultado_mensaje("Análisis de sentimiento y NPS completado."), "success")
            tabla_html = df_resultado[['comentario', 'sentimiento', 'estado', 'nps']].head(10).to_html(classes="table table-striped", index=False)
            return render_template(
                'sentimiento.html',
                tabla_html=tabla_html,
                nps_score=nps_score,
                csv_out=csv_out
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('sentimiento.html')

    @app.route('/siniestros', methods=['GET', 'POST'])
def siniestros_view():
    """
    Analiza los tiempos de gestión de siniestros, detecta cuellos de botella y sugiere optimización.
    """
    df_resultado = None
    cuellos = None
    recomendaciones = None

    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            df_resultado, cuellos, recomendaciones = analizar_siniestros(df)
            flash(resultado_mensaje("Análisis de siniestros completado."), "success")
            return render_template(
                'siniestros.html',
                df_resultado=df_resultado.head(10).to_html(classes="table table-striped", index=False),
                cuellos=cuellos.to_html(classes="table table-striped", index=True),
                recomendaciones=recomendaciones
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('siniestros.html')

    @app.route('/clientes_dormidos', methods=['GET', 'POST'])
def clientes_dormidos_view():
    """
    Identifica y prioriza clientes dormidos para reactivación comercial.
    """
    resultado = None
    file_export = None

    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            resultado, file_export = priorizar_clientes_dormidos(df)
            flash(resultado_mensaje("Clientes priorizados por score de reactivación. Puedes descargar el archivo Excel abajo."), "success")
            return render_template(
                'clientes_dormidos.html',
                resultado=resultado.head(10).to_html(classes="table table-striped", index=False),
                file_export=file_export
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('clientes_dormidos.html')

    @app.route('/renovaciones', methods=['GET', 'POST'])
def renovaciones_view():
    """
    Detecta pólizas próximas a vencer y genera recordatorios para el equipo comercial o email.
    """
    resultado = None
    file_export = None
    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            resultado, file_export = digitalizar_renovaciones(df)
            flash(resultado_mensaje("¡Recordatorios generados con éxito! Descarga el archivo CSV abajo."), "success")
            return render_template(
                'renovaciones.html',
                resultado=resultado.head(10).to_html(classes="table table-striped", index=False),
                file_export=file_export
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('renovaciones.html')

    @app.route('/panel_kpis', methods=['GET', 'POST'])
def panel_kpis_view():
    """
    Calcula y muestra los principales KPIs aseguradores.
    """
    resultado = None
    file_export = None
    if request.method == 'POST':
        archivo = request.files.get('dataset')
        try:
            df = cargar_dataset(archivo)
            resultado, file_export = calcular_panel_kpis(df)
            flash(resultado_mensaje("Panel de KPIs calculado con éxito. Descarga el CSV abajo."), "success")
            return render_template(
                'panel_kpis.html',
                resultado=resultado.to_html(classes="table table-striped", index=False),
                file_export=file_export
            )
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('panel_kpis.html')

    @app.route('/correlacion_perfil_producto', methods=['GET', 'POST'])
def correlacion_perfil_producto():
    """
    Permite subir un archivo CSV de clientes/productos y analiza la correlación.
    """
    archivos_generados = None
    if request.method == 'POST':
        archivo = request.files.get('dataset')
        if not archivo:
            flash(resultado_mensaje("Debes subir un archivo CSV.", exito=False), "danger")
            return render_template('correlacion_perfil_producto.html')
        try:
            df = pd.read_csv(archivo)
            if not {'perfil_cliente', 'producto'}.issubset(df.columns):
                raise ValueError("El archivo debe contener las columnas 'perfil_cliente' y 'producto'.")
            archivo_matriz, archivo_corr = analizar_perfiles_productos(df)
            archivos_generados = {
                "Matriz Frecuencia": archivo_matriz,
                "Correlación Productos": archivo_corr
            }
            flash(resultado_mensaje("Análisis completado. Descarga los resultados abajo."), "success")
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('correlacion_perfil_producto.html', archivos=archivos_generados)

    @app.route('/cross_sell', methods=['GET', 'POST'])
def cross_sell():
    """
    Sube un archivo CSV y detecta oportunidades de cross-sell.
    """
    archivo_generado = None
    num_candidatos = None
    if request.method == 'POST':
        archivo = request.files.get('dataset')
        if not archivo:
            flash(resultado_mensaje("Debes subir un archivo CSV.", exito=False), "danger")
            return render_template('cross_sell.html')
        try:
            df = pd.read_csv(archivo)
            if 'cliente_id' not in df.columns or 'producto' not in df.columns:
                raise ValueError("El archivo debe tener las columnas 'cliente_id' y 'producto'.")
            archivo_generado, num_candidatos = detectar_cross_sell(df)
            flash(resultado_mensaje(f"{num_candidatos} clientes detectados con potencial de cross-sell."), "success")
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('cross_sell.html', archivo=archivo_generado)

    @app.route('/benchmarking_precios', methods=['GET', 'POST'])
def benchmarking_precios():
    """
    Sube dos archivos CSV (propio y competencia), compara precios y exporta resultados.
    """
    archivo_generado = None
    num_productos = None
    if request.method == 'POST':
        archivo_propio = request.files.get('archivo_propio')
        archivo_competencia = request.files.get('archivo_competencia')
        if not archivo_propio or not archivo_competencia:
            flash(resultado_mensaje("Debes subir ambos archivos CSV." , exito=False), "danger")
            return render_template('benchmarking_precios.html')
        try:
            coare = pd.read_csv(archivo_propio)
            competencia = pd.read_csv(archivo_competencia)
            if not {'producto', 'precio'}.issubset(coare.columns) or not {'producto', 'precio'}.issubset(competencia.columns):
                raise ValueError("Ambos archivos deben tener columnas 'producto' y 'precio'.")
            archivo_generado, num_productos = comparar_precios(coare, competencia)
            flash(resultado_mensaje(f"{num_productos} productos comparados. Descarga el Excel con los resultados."), "success")
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('benchmarking_precios.html', archivo=archivo_generado)

    @app.route('/ltv_clientes', methods=['GET', 'POST'])
def ltv_clientes():
    archivo_generado = None
    num_clientes = None
    if request.method == 'POST':
        archivo = request.files.get('archivo_ltv')
        if not archivo:
            flash(resultado_mensaje("Debes subir un archivo CSV.", exito=False), "danger")
            return render_template('ltv_clientes.html')
        try:
            df = pd.read_csv(archivo)
            if not {'cliente_id', 'fecha_compra', 'importe'}.issubset(df.columns):
                raise ValueError("El archivo debe tener las columnas: cliente_id, fecha_compra, importe.")
            archivo_generado, num_clientes = calcular_ltv(df)
            flash(resultado_mensaje(f"LTV calculado para {num_clientes} clientes. Descarga el Excel."), "success")
        except Exception as e:
            flash(resultado_mensaje(f"Error: {e}", exito=False), "danger")
    return render_template('ltv_clientes.html', archivo=archivo_generado)

    @app.route('/kriging', methods=['GET', 'POST'])
def kriging():
    grafico_path = None
    if request.method == 'POST':
        archivo = request.files.get('dataset')
        variable = request.form.get('variable', 'ndvi')
        if archivo:
            ruta = os.path.join('uploads', archivo.filename)
            archivo.save(ruta)
            try:
                grafico_path = ejecutar_kriging(ruta, variable)
                flash("Mapa generado correctamente.", "success")
            except Exception as e:
                flash(f"Error: {e}", "danger")
        else:
            flash("No se ha subido ningún archivo.", "danger")
    return render_template('kriging.html', grafico=grafico_path)

    @app.route('/biomasa3d', methods=['GET', 'POST'])
def biomasa3d():
    dsm_path = dtm_path = None
    volumen_biomasa = None
    if request.method == 'POST':
        archivo = request.files.get('archivo_las')
        grid_size = request.form.get('grid_size', 1, type=float)
        if archivo:
            ruta = os.path.join('uploads', archivo.filename)
            archivo.save(ruta)
            try:
                dsm_path, dtm_path, volumen_biomasa = analizar_biomasa_las(ruta, grid_size)
                flash(f"Biomasa analizada correctamente. Volumen estimado: {volumen_biomasa:.2f} m³", "success")
            except Exception as e:
                flash(f"Error: {e}", "danger")
        else:
            flash("No se ha subido ningún archivo LAS.", "danger")
    return render_template('biomasa3d.html', dsm=dsm_path, dtm=dtm_path, volumen=volumen_biomasa)

    @app.route('/salud_cultivo', methods=['GET', 'POST'])
def salud_cultivo():
    boxplot_path = resumen_path = None
    if request.method == 'POST':
        archivo = request.files.get('archivo_csv')
        if archivo:
            ruta = os.path.join('uploads', archivo.filename)
            archivo.save(ruta)
            try:
                boxplot_path, resumen_path, _ = analizar_salud_cultivo(ruta)
                flash("Análisis de salud de cultivos realizado correctamente.", "success")
            except Exception as e:
                flash(f"Error: {e}", "danger")
        else:
            flash("Debes subir un archivo CSV con los datos de índices.", "danger")
    return render_template('salud_cultivo.html', boxplot=boxplot_path, resumen=resumen_path)

    @app.route('/segmentacion_ndvi', methods=['GET', 'POST'])
def segmentacion_ndvi():
    mapa_html = recomendaciones = geojson = csv = None
    if request.method == 'POST':
        archivo_ndvi = request.files.get('archivo_ndvi')
        archivo_parcela = request.files.get('archivo_parcela')
        n_clusters = int(request.form.get('n_clusters', 4))
        if archivo_ndvi and archivo_parcela:
            ruta_ndvi = os.path.join('uploads', archivo_ndvi.filename)
            ruta_parcela = os.path.join('uploads', archivo_parcela.filename)
            archivo_ndvi.save(ruta_ndvi)
            archivo_parcela.save(ruta_parcela)
            try:
                mapa_html, recomendaciones, geojson, csv = segmentar_parcela_ndvi(
                    ruta_ndvi, ruta_parcela, n_clusters)
                flash("Segmentación realizada con éxito.", "success")
            except Exception as e:
                flash(f"Error: {e}", "danger")
        else:
            flash("Debes subir ambos archivos (NDVI y GeoJSON).", "danger")
    return render_template(
        'segmentacion_ndvi.html',
        mapa=mapa_html,
        recomendaciones=recomendaciones,
        geojson=geojson,
        csv=csv
    )

    @app.route('/visor_parcelas', methods=['GET', 'POST'])
def visor_parcelas():
    mapa = None
    if request.method == 'POST':
        archivo_parcelas = request.files.get('archivo_parcelas')
        if archivo_parcelas:
            ruta_parcelas = os.path.join('uploads', archivo_parcelas.filename)
            archivo_parcelas.save(ruta_parcelas)
            try:
                mapa = generar_visor_parcelas(ruta_parcelas)
                flash("Visor generado correctamente.", "success")
            except Exception as e:
                flash(f"Error: {e}", "danger")
        else:
            flash("Debes subir un archivo GeoJSON o SHP.", "danger")
    return render_template('visor_parcelas.html', mapa=mapa)

    @app.route('/optimizacion_riego', methods=['GET', 'POST'])
def optimizacion_riego():
    grafico = None
    prediccion = None
    modelo_path = os.path.join('static/graficos', 'modelo_recomendacion_riego.pkl')
    if request.method == 'POST':
        # Si se sube dataset y se pide entrenar modelo
        archivo_datos = request.files.get('datos_riego')
        if archivo_datos:
            ruta = os.path.join('uploads', archivo_datos.filename)
            archivo_datos.save(ruta)
            try:
                modelo_path, grafico = entrenar_modelo_riego(ruta)
                flash("Modelo de riego entrenado correctamente. Ya puedes hacer predicciones.", "success")
            except Exception as e:
                flash(f"Error al entrenar: {e}", "danger")
        # Si se introduce un formulario para predecir
        elif request.form.get('humedad_suelo'):
            try:
                humedad = float(request.form['humedad_suelo'])
                temperatura = float(request.form['temperatura'])
                precipitacion = float(request.form['precipitacion'])
                evapotranspiracion = float(request.form['evapotranspiracion'])
                prediccion = predecir_riego(modelo_path, humedad, temperatura, precipitacion, evapotranspiracion)
                flash("Predicción de riego generada.", "success")
            except Exception as e:
                flash(f"Error al predecir: {e}", "danger")
    return render_template('optimizacion_riego.html', grafico=grafico, prediccion=prediccion)

    @app.route('/deteccion_plagas', methods=['GET', 'POST'])
def deteccion_plagas():
    prediccion = None
    imagen_cargada = None
    if request.method == 'POST':
        # Entrenamiento (opcional, subir carpeta de imagenes y CSV)
        if 'labels_csv' in request.files and request.files['labels_csv']:
            labels_csv = request.files['labels_csv']
            carpeta_imgs = request.form.get('carpeta_imgs', 'imagenes_dron') # default
            ruta_labels = os.path.join('uploads', labels_csv.filename)
            labels_csv.save(ruta_labels)
            pesos_path = entrenar_red_plagas(carpeta_imgs, ruta_labels)
            flash("Modelo de plagas entrenado y guardado.", "success")
        # Predicción (subida de imagen individual)
        elif 'imagen_prediccion' in request.files and request.files['imagen_prediccion']:
            imagen = request.files['imagen_prediccion']
            ruta_img = os.path.join('uploads', imagen.filename)
            imagen.save(ruta_img)
            pesos_modelo = 'static/graficos/modelo_plagas_enfermedades.pth'
            try:
                modelo = cargar_modelo_pytorch(pesos_modelo)
                resultado = predecir_imagen(modelo, ruta_img)
                prediccion = resultado
                imagen_cargada = '/' + ruta_img
                flash("Predicción realizada correctamente.", "success")
            except Exception as e:
                flash(f"Error al predecir: {e}", "danger")
    return render_template('deteccion_plagas.html', prediccion=prediccion, imagen_cargada=imagen_cargada)

    @app.route('/monitorizacion_fenologica', methods=['GET', 'POST'])
def monitorizacion_fenologica():
    brotes = None
    imagenes = []
    resumen = None
    if request.method == 'POST':
        carpeta_ndvi = request.form.get('carpeta_ndvi', 'imagenes_ndvi_temporal')
        umbral = float(request.form.get('umbral_brote', 0.08))
        df, df_brotes, img_paths = analizar_evolucion_fenologica(carpeta_ndvi, umbral_brote=umbral)
        if df is not None:
            resumen = df.head(10).to_html(classes="table table-striped", index=False)
            brotes = df_brotes[['fecha','parcela','ndvi_delta']].to_html(classes="table table-bordered", index=False)
            imagenes = img_paths
        else:
            flash("No se encontraron datos para analizar.", "warning")
    return render_template('monitorizacion_fenologica.html', brotes=brotes, imagenes=imagenes, resumen=resumen)

    @app.route('/indices_vegetacion', methods=['GET', 'POST'])
def indices_vegetacion():
    resumen = None
    imagenes = []
    if request.method == 'POST':
        carpeta_datos = request.form.get('carpeta_datos', 'imagenes_multiespectrales/')
        df_indices, img_indices = calcular_indices_vegetacion(carpeta_datos)
        if df_indices is not None:
            resumen = df_indices.to_html(classes="table table-striped", index=False)
            imagenes = img_indices
        else:
            flash("No se encontraron datos de índices vegetativos en la ruta indicada.", "warning")
    return render_template('indices_vegetacion.html', resumen=resumen, imagenes=imagenes)

    @app.route('/estres_hidrico', methods=['GET', 'POST'])
def estres_hidrico():
    resultado = None
    if request.method == 'POST':
        ruta_termica = request.form.get('ruta_termica', 'imagenes_termicas/parcela1_2024-05-20.tif')
        ruta_parcelas = request.form.get('ruta_parcelas', 'parcelas.geojson')
        limite_critico = float(request.form.get('limite_critico', 30.0))
        cwsi_critico = float(request.form.get('cwsi_critico', 0.7))
        resultado = detectar_estres_hidrico(
            ruta_termica, ruta_parcelas, limite_critico, cwsi_critico
        )
    return render_template('estres_hidrico.html', resultado=resultado)

    @app.route('/prediccion_rendimiento', methods=['GET', 'POST'])
def prediccion_rendimiento():
    resultado = None
    if request.method == 'POST':
        archivo = request.form.get('archivo_datos', 'datos_rendimiento_parcelas.csv')
        # Aquí podrías dejar que el usuario elija features/objetivo en el formulario avanzado.
        resultado = predecir_rendimiento(archivo)
    return render_template('prediccion_rendimiento.html', resultado=resultado)

    @app.route('/prescripcion_manejo', methods=['GET', 'POST'])
def prescripcion_manejo():
    resultado = None
    if request.method == 'POST':
        archivo = request.form.get('archivo_csv', 'zonas_analizadas.csv')
        resultado = prescribir_manejo(archivo_csv=archivo)
    return render_template('prescripcion_manejo.html', resultado=resultado)

    @app.route('/water_stress_ml', methods=['GET', 'POST'])
def water_stress_ml():
    resultado = None
    if request.method == 'POST':
        archivo = request.form.get('archivo_csv', 'dataset_estres_hidrico_multianual.csv')
        resultado = predecir_estres_hidrico(archivo_csv=archivo)
    return render_template('water_stress_ml.html', resultado=resultado)

    @app.route('/yield_prediction_analytics', methods=['GET', 'POST'])
def yield_prediction_analytics():
    resultado = None
    if request.method == 'POST':
        archivo = request.form.get('archivo_csv', 'datos_rendimiento_parcelas.csv')
        resultado = predecir_rendimiento_parcelas(archivo_csv=archivo)
    return render_template('yield_prediction_analytics.html', resultado=resultado)

# -- INICIO SEGURO (PRODUCCIÓN) --

if __name__ == "__main__":
    # Usa el puerto proporcionado por Render o 5000 como valor predeterminado
    port = int(os.environ.get('PORT', 5000))  # Render automáticamente proporciona la variable PORT
    app.run(host="0.0.0.0", port=port)  # Escucha en todas las interfaces y en el puerto proporcionado







