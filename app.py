import streamlit as st
import pandas as pd
import plotly.express as px
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Configuración de página
st.set_page_config(
    page_title="Análisis Comparativo de Airbnb",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        color: #FF5A5F;
    }
    .section-header {
        font-weight: 600;
        margin-top: 1rem;
        border-bottom: 2px solid #FF5A5F;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ——— FUNCIONES ———
@st.cache_resource
def load_data(city_files=None):
    """
    Carga y prepara los datos para el análisis de múltiples ciudades.
    Usa la estructura y tipos de datos del primer dataset como plantilla.
    
    Args:
        city_files (dict): Diccionario con nombres de ciudades como claves y rutas de archivo como valores
        
    Returns:
        tuple: DataFrame combinado, columnas numéricas, columnas de texto, valores únicos de room_type, 
               DataFrame numérico, diccionario de DataFrames por ciudad
    """
    if city_files is None:
        # Archivos por defecto para cada ciudad
        city_files = {
            'New York': 'listings - New York_clean.csv',
            'CDMX': 'listings - CDMX_clean.csv',
            'Florencia': 'listings_Florencia_clean.csv',
            'Bangkok': 'listings_bangkok.csv'
        }
    
    # Diccionario para almacenar los dataframes de cada ciudad
    dfs = {}
    first_city_df = None
    first_city_name = next(iter(city_files))
    
    # Primero cargar el dataset de la primera ciudad como plantilla
    try:
        first_city_df = pd.read_csv(city_files[first_city_name], index_col='id')
        first_city_df['city'] = first_city_name
        dfs[first_city_name] = first_city_df
    except FileNotFoundError:
        st.error(f"Archivo de la primera ciudad no encontrado: {city_files[first_city_name]}")
        return None, [], [], [], None, {}
    
    # Obtener columnas y tipos de datos del primer dataset
    first_city_columns = first_city_df.columns.tolist()
    first_city_dtypes = first_city_df.dtypes
    
    # Cargar los datasets de las demás ciudades
    for city, file_path in list(city_files.items())[1:]:
        try:
            df = pd.read_csv(file_path, index_col='id')
            
            # Mantener solo las columnas que existen en el primer dataset
            for col in df.columns:
                if col not in first_city_columns:
                    df = df.drop(columns=[col])
            
            # Añadir columnas faltantes con valores NaN
            for col in first_city_columns:
                if col not in df.columns:
                    df[col] = pd.NA
            
            # Reordenar columnas para que coincidan con el primer dataset
            df = df[first_city_columns]
            
            # Convertir tipos de datos para que coincidan con el primer dataset
            for col in first_city_columns:
                try:
                    df[col] = df[col].astype(first_city_dtypes[col])
                except:
                    pass  # Si falla la conversión, mantener el tipo actual
            
            # Añadir columna para identificar la ciudad
            df['city'] = city
            dfs[city] = df
        except FileNotFoundError:
            st.warning(f"Archivo no encontrado: {file_path}")
    
    if len(dfs) == 0:
        st.error("No se pudo cargar ningún dataset.")
        return None, [], [], [], None, {}
    
    # Combinar todos los dataframes
    all_df = pd.concat(dfs.values())
    
    # Obtener columnas numéricas del primer dataframe
    numeric_cols = first_city_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Obtener columnas de texto del primer dataframe
    text_cols = first_city_df.select_dtypes(include=['object']).columns.tolist()
    
    # Obtener valores únicos de room_type de todos los datasets
    if 'room_type' in first_city_columns:
        unique_room_type = all_df['room_type'].unique().tolist()
    else:
        unique_room_type = []
    
    # Obtener el dataframe numérico del dataset combinado
    numeric_df = all_df[numeric_cols]
    
    return all_df, numeric_cols, text_cols, unique_room_type, numeric_df, dfs

def display_general_summary(filtered_df, selected_cities):
    """
    Muestra el resumen general de los datos para las ciudades seleccionadas
    
    Args:
        filtered_df (DataFrame): DataFrame filtrado para el análisis
        selected_cities (list): Lista de ciudades seleccionadas
    """
    st.markdown('<h2 class="section-header">Resumen general</h2>', unsafe_allow_html=True)
    
    # Métricas comparativas entre ciudades
    price_avg = filtered_df.groupby('city')['price'].mean().reset_index()
    reviews_avg = filtered_df.groupby('city')['review_scores_rating'].mean().reset_index() if 'review_scores_rating' in filtered_df.columns else None
    count_by_city = filtered_df['city'].value_counts().reset_index()
    count_by_city.columns = ['city', 'count']
    
    # Gráficos comparativos
    col1, col2 = st.columns(2)
    
    with col1:
        fig_price = px.bar(price_avg, x='city', y='price', 
                         title="Precio promedio por ciudad",
                         color='city',
                         text_auto='.2f')
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        fig_count = px.bar(count_by_city, x='city', y='count', 
                          title="Número de alojamientos por ciudad",
                          color='city',
                          text_auto=True)
        st.plotly_chart(fig_count, use_container_width=True)
    
    # Métricas por ciudad
    st.subheader("Métricas por ciudad")
    city_cols = st.columns(len(selected_cities))
    
    for i, city in enumerate(selected_cities):
        city_df = filtered_df[filtered_df['city'] == city]
        
        with city_cols[i]:
            st.markdown(f'<div class="metric-card" style="text-align:center;"><h4>{city}</h4>', unsafe_allow_html=True)
            st.metric("Precio promedio", f"${city_df['price'].mean():.2f}")
            
            if 'review_scores_rating' in filtered_df.columns:
                st.metric("Puntuación media", f"{city_df['review_scores_rating'].mean():.2f}/5")
            
            st.metric("Total alojamientos", f"{len(city_df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Estadísticas descriptivas
    st.subheader("Estadísticas descriptivas")
    city_for_stats = st.selectbox("Selecciona una ciudad para ver estadísticas detalladas", selected_cities)
    city_df = filtered_df[filtered_df['city'] == city_for_stats]
    
    with st.expander(f"Estadísticas numéricas - {city_for_stats}", expanded=True):
        st.dataframe(city_df.describe(), use_container_width=True)

    # Reporte automático
    with st.expander(f"Reporte automático - {city_for_stats}", expanded=False):
        if st.button("Generar reporte detallado", key="gen_report"):
            with st.spinner('Generando reporte detallado...'):
                profile = ProfileReport(city_df, explorative=True, minimal=True)
                st_profile_report(profile)
        else:
            st.info("Haz clic en el botón para generar el reporte completo.")

def display_univariate_analysis(filtered_df, numeric_cols, selected_cities):
    """
    Muestra el análisis univariante con comparación entre ciudades
    
    Args:
        filtered_df (DataFrame): DataFrame filtrado para el análisis
        numeric_cols (list): Lista de columnas numéricas
        selected_cities (list): Lista de ciudades seleccionadas
    """
    st.markdown('<h2 class="section-header">Análisis univariante</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Elegir variable (excluyendo 'city')
        all_cols = [col for col in filtered_df.columns.tolist() if col != 'city']
        var = st.selectbox("Selecciona una variable", all_cols)
    
    with col2:
        # Elegir tipo de gráfico según el tipo de variable
        if var in numeric_cols:
            chart_type = st.selectbox("Tipo de gráfica", ['Histograma', 'Caja', 'Líneas'], key='numeric_chart')
        else:
            chart_type = st.selectbox("Tipo de gráfica", ['Barras', 'Pastel'], key='categorical_chart')
    
    # Comparación entre ciudades
    if var in numeric_cols:
        if chart_type == 'Histograma':
            fig = px.histogram(filtered_df, x=var, color='city', 
                              title=f"Histograma de {var} por ciudad",
                              marginal="box", opacity=0.7,
                              barmode="overlay")
            fig.update_layout(bargap=0.1)
        elif chart_type == 'Caja':
            fig = px.box(filtered_df, y=var, x='city', 
                        title=f"Box-plot de {var} por ciudad", 
                        color='city')
        else:  # Líneas
            # Calcular la distribución de densidad para cada ciudad
            density_data = []
            for city in selected_cities:
                city_data = filtered_df[filtered_df['city'] == city][var].dropna()
                if len(city_data) > 0:
                    # Generar puntos para la curva de densidad
                    hist, bin_edges = np.histogram(city_data, bins=20, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    for x, y in zip(bin_centers, hist):
                        density_data.append({"x": x, "y": y, "city": city})
            
            density_df = pd.DataFrame(density_data)
            fig = px.line(density_df, x="x", y="y", color="city",
                         title=f"Curva de densidad de {var} por ciudad",
                         labels={"x": var, "y": "Densidad"})
    else:
        # Para variables categóricas
        vc = filtered_df.groupby(['city', var]).size().reset_index(name='count')
        if chart_type == 'Barras':
            fig = px.bar(vc, x=var, y='count', color='city',
                        title=f"Distribución de {var} por ciudad",
                        barmode='group')
        else:  # Pastel
            fig = px.pie(vc, values='count', names=var, 
                        facet_col='city',
                        title=f"Distribución de {var} por ciudad",
                        color_discrete_sequence=px.colors.qualitative.Plotly)
    
    # Mejorar diseño del gráfico
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Estadísticas por ciudad
    st.subheader("Estadísticas por ciudad")
    city_for_stats = st.selectbox("Selecciona una ciudad", selected_cities, key="city_univar_stats")
    city_df = filtered_df[filtered_df['city'] == city_for_stats]
    
    st.write(city_df[var].describe())

def display_bivariate_analysis(filtered_df, numeric_cols, selected_cities):
    """
    Muestra el análisis bivariante con comparación entre ciudades
    
    Args:
        filtered_df (DataFrame): DataFrame filtrado para el análisis
        numeric_cols (list): Lista de columnas numéricas
        selected_cities (list): Lista de ciudades seleccionadas
    """
    st.markdown('<h2 class="section-header">Análisis bivariante</h2>', unsafe_allow_html=True)
    
    # Seleccionar tipo de análisis
    analysis_type = st.radio("Tipo de análisis", 
                            ["Regresión lineal simple", "Regresión lineal múltiple", "Regresión logística"], 
                            horizontal=True)
    
    # Selector de ciudades para comparar o analizar individualmente
    compare_mode = st.radio("Modo de visualización", ["Comparar ciudades", "Analizar ciudad individual"], horizontal=True)
    
    if analysis_type == "Regresión lineal simple":
        display_simple_linear_regression(filtered_df, numeric_cols, selected_cities, compare_mode)
    elif analysis_type == "Regresión lineal múltiple":
        display_multiple_linear_regression(filtered_df, numeric_cols, selected_cities, compare_mode)
    else:  # Regresión logística
        display_logistic_regression(filtered_df, numeric_cols, selected_cities, compare_mode)

def display_simple_linear_regression(filtered_df, numeric_cols, selected_cities, compare_mode):
    """
    Muestra análisis de regresión lineal simple con comparación entre ciudades
    
    Args:
        filtered_df (DataFrame): DataFrame filtrado para el análisis
        numeric_cols (list): Lista de columnas numéricas
        selected_cities (list): Lista de ciudades seleccionadas
        compare_mode (str): Modo de comparación ('Comparar ciudades' o 'Analizar ciudad individual')
    """
    col1, col2 = st.columns([1, 1])
    
    with col1:
        x_var = st.selectbox("Variable eje X", numeric_cols, index=0, key="simple_x")
    
    with col2:
        y_var = st.selectbox("Variable eje Y", numeric_cols, index=min(1, len(numeric_cols)-1), key="simple_y")
    
    add_trend = st.checkbox("Agregar línea de tendencia", value=True)
    
    # Comprobar si se seleccionaron las mismas variables
    if x_var == y_var:
        st.error("Por favor selecciona variables diferentes para los ejes X e Y.")
    else:
        if compare_mode == "Comparar ciudades":
            # Gráfico comparativo de dispersión
            fig = px.scatter(filtered_df, x=x_var, y=y_var, color='city',
                          trendline='ols' if add_trend else None,
                          title=f"{y_var} vs {x_var} - Comparación entre ciudades",
                          opacity=0.7)
            
            # Mostrar correlaciones por ciudad
            corr_by_city = {}
            for city in selected_cities:
                city_df = filtered_df[filtered_df['city'] == city]
                corr = city_df[[x_var, y_var]].corr().iloc[0, 1]
                corr_by_city[city] = corr
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar correlaciones en formato tabular
            corr_data = pd.DataFrame(list(corr_by_city.items()), columns=['Ciudad', 'Correlación'])
            st.write("Correlaciones de Pearson por ciudad:")
            st.dataframe(corr_data)
            
        else:  # Análisis individual
            city = st.selectbox("Selecciona una ciudad", selected_cities, key="city_simple_reg")
            city_df = filtered_df[filtered_df['city'] == city]
            
            fig = px.scatter(city_df, x=x_var, y=y_var,
                          trendline='ols' if add_trend else None,
                          title=f"{y_var} vs {x_var} - {city}",
                          color_discrete_sequence=['#FF5A5F'],
                          opacity=0.7)
            
            # Modificar color de línea de tendencia
            if add_trend:
                for trace in fig.data:
                    if hasattr(trace, 'mode') and trace.mode == 'lines':
                        trace.line.color = '#FF5A5F'
                        trace.line.width = 3
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar correlación
            corr = city_df[[x_var, y_var]].corr().iloc[0, 1]
            st.info(f"**Correlación de Pearson:** {corr:.2f}")

def display_multiple_linear_regression(filtered_df, numeric_cols, selected_cities, compare_mode):
    """
    Muestra análisis de regresión lineal múltiple con comparación entre ciudades
    
    Args:
        filtered_df (DataFrame): DataFrame filtrado para el análisis
        numeric_cols (list): Lista de columnas numéricas
        selected_cities (list): Lista de ciudades seleccionadas
        compare_mode (str): Modo de comparación ('Comparar ciudades' o 'Analizar ciudad individual')
    """
    col1, col2 = st.columns([1, 1])
    
    with col1:
        y_var = st.selectbox("Variable dependiente (Y)", numeric_cols, key="multi_y")
    
    with col2:
        x_vars = st.multiselect("Variables independientes (X)", 
                    [col for col in numeric_cols if col != y_var], 
                    default=[numeric_cols[0]] if numeric_cols and numeric_cols[0] != y_var else [])
    
    if not x_vars:
        st.warning("Selecciona al menos una variable independiente.")
    else:
        if compare_mode == "Comparar ciudades":
            # Análisis comparativo para cada ciudad
            r_values = {}
            
            for city in selected_cities:
                city_df = filtered_df[filtered_df['city'] == city]
                model_df = city_df[[y_var] + x_vars].dropna()
                
                if len(model_df) > 0:
                    X = sm.add_constant(model_df[x_vars])
                    y = model_df[y_var]
                    model = sm.OLS(y, X).fit()
                    # Calcular R (correlación múltiple) en lugar de R²
                    r_values[city] = np.sqrt(model.rsquared)
            
            # Mostrar R en un gráfico de barras
            r_df = pd.DataFrame(list(r_values.items()), columns=['Ciudad', 'R'])
            fig = px.bar(r_df, x='Ciudad', y='R', color='Ciudad', 
                        title="Coeficiente de correlación múltiple (R) por ciudad",
                        text_auto='.3f')
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Análisis individual
            city = st.selectbox("Selecciona una ciudad", selected_cities, key="city_multi_reg")
            city_df = filtered_df[filtered_df['city'] == city]
            
            # Eliminar filas con valores nulos en las variables seleccionadas
            model_df = city_df[[y_var] + x_vars].dropna()
            
            if len(model_df) == 0:
                st.error(f"No hay datos suficientes para el análisis en {city}.")
            else:
                # Crear y ajustar el modelo
                X = sm.add_constant(model_df[x_vars])
                y = model_df[y_var]
                model = sm.OLS(y, X).fit()
                
                # Calcular coeficientes y R
                multiple_r = np.sqrt(model.rsquared)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader(f"Resultados del modelo para {city}")
                    st.metric("Correlación múltiple (R)", f"{multiple_r:.4f}")
                    st.metric("Coeficiente de determinación (R²)", f"{model.rsquared:.4f}")
                
                with col2:
                    # Gráfico de valores reales vs predichos
                    predictions = model.predict(X)
                    pred_df = pd.DataFrame({'Actual': y, 'Predicción': predictions})
                    
                    fig = px.scatter(pred_df, x='Actual', y='Predicción',
                            title=f"Valores reales vs. predichos - {city}",
                            template="plotly_white")
                    
                    # Añadir línea diagonal de referencia
                    fig.add_shape(
                        type='line',
                        x0=pred_df['Actual'].min(),
                        y0=pred_df['Actual'].min(),
                        x1=pred_df['Actual'].max(),
                        y1=pred_df['Actual'].max(),
                        line=dict(color='#FF5A5F', dash='dash')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Añadir matriz de correlaciones
                st.subheader("Matriz de correlaciones")
                # Calcular la matriz de correlaciones para todas las variables del modelo
                corr_matrix = model_df.corr()
                
                # Crear el heatmap usando plotly
                corr_fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    title=f"Matriz de correlaciones - {city}",
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1
                )
                st.plotly_chart(corr_fig, use_container_width=True)

def display_logistic_regression(filtered_df, numeric_cols, selected_cities, compare_mode):
    """
    Muestra análisis de regresión logística con comparación entre ciudades
    """
    st.subheader("Regresión logística")
    
    # Selector para variable binaria
    binary_option = st.radio("Variable objetivo binaria", 
                            ["Usar variable existente", "Binarizar variable numérica"])
    
    binary_var = None
    
    if binary_option == "Usar variable existente":
        # Encontrar variables binarias o categóricas con dos valores en el dataframe
        binary_cols = []
        for col in filtered_df.columns:
            if col != 'city' and filtered_df[col].nunique() == 2:
                binary_cols.append(col)
        
        if not binary_cols:
            st.warning("No se encontraron variables binarias en el dataset")
        else:
            binary_var = st.selectbox("Selecciona variable binaria", binary_cols)
            
            # Convertir a numérico si es de tipo objeto
            if binary_var and filtered_df[binary_var].dtype == 'object':
                unique_values = filtered_df[binary_var].unique()
                # Crear copia temporal para evitar advertencias de SettingWithCopyWarning
                temp_df = filtered_df.copy()
                # Mapear valores a 0 y 1
                value_map = {unique_values[0]: 0, unique_values[1]: 1}
                temp_df[binary_var] = temp_df[binary_var].map(value_map)
                filtered_df = temp_df
                st.info(f"Variable '{binary_var}' convertida de categórica a numérica: {unique_values[0]}=0, {unique_values[1]}=1")
    
    else:  # Binarizar variable
        num_var_to_bin = st.selectbox("Variable para binarizar", numeric_cols)
        threshold = st.slider("Umbral para binarización", 
                            float(filtered_df[num_var_to_bin].min()),
                            float(filtered_df[num_var_to_bin].max()),
                            float(filtered_df[num_var_to_bin].median()))
        
        binary_var = f"{num_var_to_bin}_bin"
        # Crear copia para evitar advertencias
        temp_df = filtered_df.copy()
        temp_df[binary_var] = (temp_df[num_var_to_bin] > threshold).astype(int)
        filtered_df = temp_df
    
    if binary_var:
        x_vars = st.multiselect("Variables independientes", 
                                [col for col in numeric_cols if col != binary_var],
                                default=[numeric_cols[0]] if numeric_cols and numeric_cols[0] != binary_var else [])
        
        if not x_vars:
            st.warning("Selecciona al menos una variable independiente.")
        elif compare_mode == "Comparar ciudades":
            # Comparar métricas entre ciudades
            accuracy_values = {}
            precision_values = {}
            recall_values = {}
            
            for city in selected_cities:
                city_df = filtered_df[filtered_df['city'] == city]
                model_df = city_df[[binary_var] + x_vars].dropna()
                
                if len(model_df) > 0:
                    try:
                        # Asegurar que binary_var sea numérico
                        model_df[binary_var] = model_df[binary_var].astype(float)
                        
                        X = sm.add_constant(model_df[x_vars])
                        y = model_df[binary_var]
                        
                        logit_model = sm.Logit(y, X).fit(disp=0)
                        predictions = logit_model.predict(X)
                        pred_classes = (predictions > 0.5).astype(int)
                        
                        accuracy_values[city] = accuracy_score(y, pred_classes)
                        precision_values[city] = precision_score(y, pred_classes, zero_division=0)
                        recall_values[city] = recall_score(y, pred_classes, zero_division=0)
                    except Exception as e:
                        st.warning(f"No se pudo ajustar el modelo para {city}: {str(e)}")
            
            # Mostrar comparación de métricas
            if accuracy_values:
                metrics_df = pd.DataFrame({
                    'Ciudad': list(accuracy_values.keys()),
                    'Exactitud': list(accuracy_values.values()),
                    'Precisión': list(precision_values.values()),
                    'Sensibilidad': list(recall_values.values())
                })
                
                # Gráfico comparativo de métricas
                fig = px.bar(metrics_df.melt(id_vars=['Ciudad'], var_name='Métrica', value_name='Valor'),
                            x='Ciudad', y='Valor', color='Métrica', barmode='group',
                            title="Comparación de métricas por ciudad",
                            text_auto='.3f')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se pudo ajustar el modelo para ninguna ciudad. Intenta con diferentes variables.")
            
        else:  # Análisis individual
            city = st.selectbox("Selecciona una ciudad", selected_cities, key="city_logistic")
            city_df = filtered_df[filtered_df['city'] == city]
            
            # Análisis para la ciudad seleccionada
            model_df = city_df[[binary_var] + x_vars].dropna()
            
            if len(model_df) == 0:
                st.error(f"No hay datos suficientes para el análisis en {city}.")
            else:
                try:
                    # Asegurar que binary_var sea numérico
                    model_df[binary_var] = model_df[binary_var].astype(float)
                    
                    X = sm.add_constant(model_df[x_vars])
                    y = model_df[binary_var]
                    
                    logit_model = sm.Logit(y, X).fit(disp=0)
                    
                    # Métricas y matriz de confusión
                    predictions = logit_model.predict(X)
                    pred_classes = (predictions > 0.5).astype(int)
                    
                    accuracy = accuracy_score(y, pred_classes)
                    precision = precision_score(y, pred_classes, zero_division=0)
                    sensitivity = recall_score(y, pred_classes, zero_division=0)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader(f"Métricas del modelo para {city}")
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        
                        with metrics_col1:
                            st.metric("Exactitud", f"{accuracy:.4f}")
                        with metrics_col2:
                            st.metric("Precisión", f"{precision:.4f}")
                        with metrics_col3:
                            st.metric("Sensibilidad", f"{sensitivity:.4f}")
                    
                    with col2:
                        # Matriz de confusión
                        st.subheader("Matriz de confusión")
                        cm = confusion_matrix(y, pred_classes)
                        
                        cm_labels = ['Negativo', 'Positivo']
                        cm_fig = px.imshow(cm, 
                                        labels=dict(x="Predicción", y="Real", color="Cantidad"),
                                        x=cm_labels, 
                                        y=cm_labels,
                                        text_auto=True,
                                        color_continuous_scale='RdBu_r')
                        
                        st.plotly_chart(cm_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error al ajustar el modelo: {str(e)}")
                    st.info("Sugerencia: Intenta seleccionar diferentes variables independientes.")

def display_map(filtered_df, selected_cities):
    """
    Muestra el mapa de alojamientos para las ciudades seleccionadas
    
    Args:
        filtered_df (DataFrame): DataFrame filtrado para el análisis
        selected_cities (list): Lista de ciudades seleccionadas
    """
    st.markdown('<h2 class="section-header">Mapa de alojamientos</h2>', unsafe_allow_html=True)
    
    if {'latitude', 'longitude'}.issubset(filtered_df.columns):
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Opciones de visualización")
            color_option = st.selectbox(
                "Color por", 
                ['city', 'room_type', 'price', 'review_scores_rating', 'Ninguno'],
                index=0
            )
            
            size_option = st.selectbox(
                "Tamaño por", 
                ['price', 'review_scores_rating', 'Constante'],
                index=0
            )
            
            zoom = st.slider("Zoom", 2, 14, 3)  # Zoom inicial más bajo para ver múltiples ciudades
            
            st.info(f"**Mostrando:** {len(filtered_df)} alojamientos en {len(selected_cities)} ciudades")
        
        with col1:
            with st.spinner('Cargando mapa...'):
                fig = px.scatter_map(
                    filtered_df,
                    lat='latitude', lon='longitude',
                    size=None if size_option == 'Constante' else size_option,
                    size_max=15,
                    color=None if color_option == 'Ninguno' else color_option,
                    hover_name='name' if 'name' in filtered_df.columns else None,
                    hover_data=['city', 'price', 'room_type'],
                    mapbox_style='open-street-map',
                    title="Distribución de alojamientos por ciudad",
                    zoom=zoom,
                    height=700
                )
                
                fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No se encontraron columnas 'latitude' y/o 'longitude' en el dataset.")

# ——— MAIN APP ———
def main():
    # Título principal con estilo personalizado
    st.markdown('<h1 class="main-header">Dashboard de Análisis Comparativo: Airbnb en Múltiples Ciudades</h1>', unsafe_allow_html=True)
    st.markdown("Explora y compara datos de alojamientos en diferentes ciudades utilizando visualizaciones y análisis estadísticos.")

    # Definir archivos de datos para cada ciudad
    city_files = {
        'New York': 'listings - New York_clean.csv',
        'CDMX': 'listings - CDMX_clean.csv',
        'Florencia': 'listings_Florencia_clean.csv',
        'Bangkok': 'listings_bangkok_clean.csv'
    }
    
    # Cargamos los datos
    with st.spinner('Cargando datos de múltiples ciudades...'):
        all_df, numeric_cols, text_cols, unique_room_type, numeric_df, city_dfs = load_data(city_files)
    
    if all_df is None:
        st.error("No se pudieron cargar los datos. Verifica que los archivos existan.")
        return

    # ——— SIDEBAR ———
    st.sidebar.title("Filtros y Navegación")
    st.sidebar.markdown("---")

    # Selector de ciudades
    available_cities = list(city_dfs.keys())
    selected_cities = st.sidebar.multiselect(
        "Seleccionar ciudades",
        available_cities,
        default=available_cities[:4]  # Por defecto seleccionar las 2 primeras ciudades
    )
    
    if not selected_cities:
        st.warning("Por favor selecciona al menos una ciudad para continuar.")
        return
    
    # Filtrar por ciudades seleccionadas
    filtered_by_city = all_df[all_df['city'].isin(selected_cities)]

    # Selector de secciones
    default_page = 'Resumen general'
    page = st.sidebar.radio("Secciones", 
                            ['Resumen general', 'Univariante', 'Bivariante', 'Mapa'],
                            index=['Resumen general', 'Univariante', 'Bivariante', 'Mapa'].index(default_page))

    st.sidebar.markdown("---")
    st.sidebar.subheader("Configuración de Filtros")

    # Filtro room_type
    sel_rooms = st.sidebar.multiselect("Tipo de habitación", 
                                      unique_room_type, 
                                      default=unique_room_type)

    # Filtro precio
    min_price, max_price = int(filtered_by_city['price'].min()), int(filtered_by_city['price'].max())
    sel_price = st.sidebar.slider("Rango de precio ($)", 
                                 min_price, 
                                 max_price, 
                                 (min_price, max_price),
                                 step=10)

    # Aplicar filtros
    filtered_df = filtered_by_city[filtered_by_city['room_type'].isin(sel_rooms)]
    filtered_df = filtered_df[(filtered_df['price'] >= sel_price[0]) & (filtered_df['price'] <= sel_price[1])]

    # Mostrar conteo de resultados
    st.sidebar.markdown(f"**Resultados:** {len(filtered_df)} alojamientos en {len(selected_cities)} ciudades")
    st.sidebar.markdown("---")
    st.sidebar.info("Inteligencia de Negocios")

    # Mostrar mensaje si no hay datos
    if len(filtered_df) == 0:
        st.error("No hay datos que cumplan con los criterios de filtrado. Por favor, ajusta los filtros.")
    else:
        # Renderizar la página seleccionada
        if page == 'Resumen general':
            display_general_summary(filtered_df, selected_cities)
        elif page == 'Univariante':
            display_univariate_analysis(filtered_df, numeric_cols, selected_cities)
        elif page == 'Bivariante':
            display_bivariate_analysis(filtered_df, numeric_cols, selected_cities)
        else:  # Mapa
            display_map(filtered_df, selected_cities)

if __name__ == "__main__":
    main()
