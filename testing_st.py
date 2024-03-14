import streamlit as st
from AA import DataDownloader, AssetAllocation  


st.title('Backtesting de Estrategias de AA')

# Sidebar para la configuración de descarga de datos
st.sidebar.header('Configuración de Datos')
start_date = st.sidebar.text_input('Fecha de Inicio', value='')
end_date = st.sidebar.text_input('Fecha de Fin', value='')
assets = st.sidebar.text_area('Lista de Activos (separados por comas)', 'AAPL, IBM, TSLA, GOOG, NVDA')
benchmark = st.sidebar.text_input('Benchmark', '^GSPC')
rf_rate = st.sidebar.number_input('Tasa Libre de Riesgo', value=0.065, step=0.001)

# Configuración del método de optimización
method = st.sidebar.selectbox('Método de Optimización', ('MonteCarlo', 'SLSQP', 'Genetic', 'Gradient'))

if method == 'MonteCarlo':
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

        # Descarga de datos
        downloader = DataDownloader()
        asset_prices, benchmark_prices = downloader.download_data(start_date, end_date, assets_list, benchmark)

        # Instancia de AssetAllocation
        asset_allocation = AssetAllocation(asset_prices, benchmark_prices, rf_rate)

        # Ejecución de la optimización según el método seleccionado
        if method == 'MonteCarlo':
            results = asset_allocation.Optimize_Portfolio(method=method, n_simulations=n_simulations)
        elif method == 'SLSQP':
            results = asset_allocation.Optimize_Portfolio(method=method)
        elif method == 'Genetic':
            results = asset_allocation.Optimize_Portfolio(method=method, population_size=population_size, generations=generations, crossover_rate=crossover_rate, mutation_rate=mutation_rate)
        elif method == 'Gradient':
            results = asset_allocation.Optimize_Portfolio(method=method, learning_rate=learning_rate, max_iters=max_iters, tol=tol)
        
        # Mostrar resultados
        st.header('Resultados de Optimización')
        st.dataframe(results)
    else:
        st.error('Por favor ingresa las fechas de inicio y fin.')
