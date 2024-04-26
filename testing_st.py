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
    .stButton>button {
        color: #ffffff;
        background-color: #4B1664;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        font-size: 18px;
        margin: 3px;
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

# Mostrar el logo en la parte superior de todas las páginas
st.image("https://www.pilou.io/wp-content/uploads/2023/08/Logo-PILOU-28.png", width=200)

# Configuración de la página activa
if 'page' not in st.session_state:
    st.session_state.page = 'Inicio'

# Barra de botones para la navegación
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("Inicio"):
        st.session_state.page = 'Inicio'
with col2:
    if st.button("Backtesting"):
        st.session_state.page = 'Backtesting'
with col3:
    if st.button("Black Litterman"):
        st.session_state.page = 'Black Litterman'

# Funciones para manejar cada página
def main_page():
    st.title('Bienvenido a la Aplicación de Análisis Financiero')
    st.markdown("""
        Esta aplicación te permite analizar, optimizar y realizar backtesting de carteras financieras.
    """)

def backtesting_page():
    st.title('Backtesting de Estrategias')
    st.header('Configuración de Datos para Backtesting')
    start_date = st.date_input('Fecha de Inicio')
    end_date = st.date_input('Fecha de Fin')
    assets = st.text_area('Lista de Activos (separados por comas)', 'AAPL, IBM, TSLA, GOOG, NVDA')
    benchmark = st.text_input('Benchmark', '^GSPC')
    rf_rate = st.number_input('Tasa Libre de Riesgo', value=0.065, step=0.001)
    method = st.selectbox('Método de Optimización', ['MonteCarlo', 'SLSQP', 'Genetic', 'Gradient'])

    if method == 'MonteCarlo':
        n_simulations = st.number_input('Número de Simulaciones', min_value=1000, max_value=100000, value=10000, step=1000)
    elif method == 'Genetic':
        population_size = st.number_input('Tamaño de la Población', min_value=50, max_value=500, value=100, step=50)
        generations = st.number_input('Generaciones', min_value=100, max_value=500, value=200, step=50)
        crossover_rate = st.slider('Tasa de Cruce', min_value=0.1, max_value=1.0, value=0.7)
        mutation_rate = st.slider('Tasa de Mutación', min_value=0.01, max_value=0.1, value=0.1)
    elif method == 'Gradient':
        learning_rate = st.slider('Tasa de Aprendizaje', min_value=0.001, max_value=0.1, value=0.01)
        max_iters = st.number_input('Máximo de Iteraciones', min_value=100, max_value=10000, value=1000)
        tol = st.slider('Tolerancia', min_value=1e-8, max_value=1e-4, value=1e-6, format='%e')

    if st.button('Optimizar'):
        if start_date and end_date:
            assets_list = [asset.strip() for asset in assets.split(',')]
            downloader = DataDownloader()
            asset_prices, benchmark_prices = downloader.download_data(start_date, end_date, assets_list, benchmark)
            asset_allocation = AssetAllocation(asset_prices, benchmark_prices, rf_rate)

            if method == 'MonteCarlo':
                results = asset_allocation.optimize_portfolio(method=method, n_simulations=n_simulations)
            elif method == 'SLSQP':
                results = asset_allocation.optimize_portfolio(method=method)
            elif method == 'Genetic':
                results = asset_allocation.optimize_portfolio(method=method, population_size=population_size, generations=generations, crossover_rate=crossover_rate, mutation_rate=mutation_rate)
            elif method == 'Gradient':
                results = asset_allocation.optimize_portfolio(method=method, learning_rate=learning_rate, max_iters=max_iters, tol=tol)

            st.header('Resultados de Optimización')
            st.dataframe(results)
        else:
            st.error('Por favor ingresa las fechas de inicio y fin.')

def black_litterman_page():
    st.title('Optimización Black Litterman')
    st.header('Configuración de Datos para Black Litterman')
    start_date = st.date_input('Fecha de Inicio')
    end_date = st.date_input('Fecha de Fin')
    assets = st.text_area('Lista de Activos', 'AAPL, IBM, TSLA, GOOG, NVDA')
    benchmark = st.text_input('Benchmark', '^GSPC')
    rf_rate = st.number_input('Tasa Libre de Riesgo', value=0.065, step=0.001)
    P = st.text_area('Matrix P (separar filas con ";", valores con ",")', '1,0;0,1')
    Q = st.text_area('Vector Q (valores separados por comas)', '0.05,0.05')
    Omega = st.text_area('Matrix Omega (separar filas con ";", valores con ",")', '0.01,0;0,0.01')
    tau = st.number_input('Tau (confianza en el equilibrio)', value=0.05, step=0.01)
    if st.button('Optimizar Black Litterman'):
        if start_date and end_date:
            downloader = DataDownloader()
            asset_prices, benchmark_prices = downloader.download_data(start_date, end_date, assets.split(', '), benchmark)
            asset_allocation = AssetAllocation(asset_prices, benchmark_prices, rf_rate)
            P = [list(map(float, row.split(','))) for row in P.split(';')]
            Q = list(map(float, Q.split(',')))
            Omega = [list(map(float, row.split(','))) for row in Omega.split(';')]
            asset_allocation.set_blacklitterman_expectations(P, Q, tau, Omega)
            results = asset_allocation.optimize_portfolio(method='Black-Litterman')
            st.header('Resultados de Optimización')
            st.dataframe(results)
        else:
            st.error('Por favor ingresa las fechas de inicio y fin.')

# Mostrar la página correspondiente
if st.session_state.page == 'Inicio':
    main_page()
elif st.session_state.page == 'Backtesting':
    backtesting_page()
elif st.session_state.page == 'Black Litterman':
    black_litterman_page()
