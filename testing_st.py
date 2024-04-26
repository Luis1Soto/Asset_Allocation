import streamlit as st
from AA import DataDownloader, AssetAllocation

# Estilos personalizados
st.markdown(
    """
    <style>
    .title {
        color: #4B1664;
        font-size: 36px;
    }
    .big-font {
        font-size: 20px;
        color: #4B1664;
    }
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4B1664;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #FFC8DE;
    }
    .sidebar .block-container {
        background-color: #FFC8DE;
        color: #4B1664;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.title('Backtesting de Estrategias')

# Sidebar para la configuración de descarga de datos
st.sidebar.header('Configuración de Datos')
start_date = st.sidebar.text_input('Fecha de Inicio', value='')
end_date = st.sidebar.text_input('Fecha de Fin', value='')
assets = st.sidebar.text_area('Lista de Activos (separados por comas)', 'AAPL, IBM, TSLA, GOOG, NVDA')
benchmark = st.sidebar.text_input('Benchmark', '^GSPC')
rf_rate = st.sidebar.number_input('Tasa Libre de Riesgo', value=0.065, step=0.001)

# Configuración del método de optimización
method = st.sidebar.selectbox('Método de Optimización', ['MonteCarlo', 'SLSQP', 'Genetic', 'Gradient', 'Black-Litterman'])

# Configuraciones específicas para Black-Litterman
if method == 'Black-Litterman':
    num_views = st.sidebar.number_input('Número de opiniones (views)', min_value=1, value=1, step=1)
    P = st.sidebar.text_area('Matrix P (separar valores con comas, filas con punto y coma)', '1,0;0,1')
    Q = st.sidebar.text_area('Vector Q (separar valores con comas)', '0.05,0.05')
    Omega = st.sidebar.text_area('Matrix Omega (separar valores con comas, filas con punto y coma)', '0.01,0;0,0.01')
    tau = st.sidebar.number_input('Tau (confianza en el equilibrio)', value=0.05, step=0.01)

# Parámetros para otros métodos
elif method == 'MonteCarlo':
    n_simulations = st.sidebar.number_input('Número de Simulaciones', min_value=1000, max_value=100000, value=10000, step=1000)
elif method == 'Genetic':
    population_size = st.sidebar.number_input('Tamaño de la Población', min_value=50, max_value=500, value=100, step=50)
    generations = st.sidebar.number_input('Generaciones', min_value=100, max_value=500, value=200, step=50)
    crossover_rate = st.sidebar.slider('Tasa de Cruce', min_value=0.1, max_value=1.0, value=0.7)
    mutation_rate = st.sidebar.slider('Tasa de Mutación', min_value=0.01, max_value=0.1, value=0.1)
elif method == 'Gradient':
    learning_rate = st.sidebar.slider('Tasa de Aprendizaje', min_value=0.001, max_value=0.1, value=0.01)
    max_iters = st.sidebar.number_input('Máximo de Iteraciones', min_value=100, max_value=10000, value=1000)
    tol = st.sidebar.slider('Tolerancia', min_value=1e-8, max_value=1e-4, value=1e-6, format='%e')

# Botón para ejecutar la optimización
if st.sidebar.button('Optimizar'):
    if start_date and end_date:  # Verifica que las fechas estén ingresadas
        assets_list = [asset.strip() for asset in assets.split(',')]
        downloader = DataDownloader()
        asset_prices, benchmark_prices, ff_factors = downloader.download_data(start_date, end_date, assets_list, benchmark)
        asset_allocation = AssetAllocation(asset_prices, benchmark_prices, rf_rate)

        # Ejecutar la optimización dependiendo del método seleccionado
        if method == 'Black-Litterman':
            # Convertir inputs de usuario para Black-Litterman
            P = [[float(num) for num in row.split(',')] for row in P.split(';')]
            Q = [float(num) for num in Q.split(',')]
            Omega = [[float(num) for num in row.split(',')] for row in Omega.split(';')]
            asset_allocation.set_blacklitterman_expectations(P, Q, tau, Omega)
            results = asset_allocation.Optimize_Portfolio(method='Black-Litterman')
        elif method in ['MonteCarlo', 'SLSQP', 'Genetic', 'Gradient']:
            results = asset_allocation.Optimize_Portfolio(method=method, n_simulations=n_simulations)
        
        st.header('Resultados de Optimización')
        st.dataframe(results)
    else:
        st.error('Por favor ingresa las fechas de inicio y fin.')
