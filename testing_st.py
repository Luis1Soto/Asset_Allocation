import streamlit as st
from AA import DataDownloader, AssetAllocation, DynamicBacktester
import numpy as np
from datetime import date
import matplotlib.pyplot as plt

st.markdown("""
    <style>
    .title { color: #4B1664; font-size: 36px; text-align: center; }
    .big-font { font-size: 20px; color: black; text-align: center; }
    div.stButton > button:first-child {
        color: #ffffff; background-color: #4B1664; border: none;
        border-radius: 5px; padding: 10px 24px; font-size: 18px; margin: 3px; width: 100%;
    }
    div.stButton > button:hover { background-color: #FFC8DE; }
    .sidebar .block-container { background-color: #FFC8DE; color: #4B1664; }
    .center { display: flex; justify-content: center; align-items: center; }
    /* Selector CSS actualizado basado en tu inspección del elemento */
    .stCheckbox .checkbox label { font-size: 8px !important; } 
    </style>
""", unsafe_allow_html=True)

# Display logo at the top of all pages
st.markdown("<div class='center'><img src='https://www.pilou.io/wp-content/uploads/2023/08/Logo-PILOU-28.png' width='200'></div>", unsafe_allow_html=True)

# Page setup
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Navigation buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Home"):
        st.session_state.page = 'Home'
with col2:
    if st.button("Strategies"):
        st.session_state.page = 'Strategies'

def main_page():
    st.markdown("<h1 class='title'>Welcome to Backtesting AA Strategies!</h1>", unsafe_allow_html=True)
    st.markdown("<div class='big-font'>This application allows you to analyze, optimize, and perform backtesting of financial portfolios.</div>", unsafe_allow_html=True)
    
def process_input_matrix(input_string):
    """Procesa una entrada de string formateada y verifica que sea una matriz válida antes de convertirla a numpy array."""
    try:
        row_lists = [list(map(float, row.split(','))) for row in input_string.strip().split(';')]
        if not all(len(row) == len(row_lists) for row in row_lists):
            st.error("Error: La matriz Omega debe ser cuadrada y todas las filas deben tener la misma cantidad de elementos.")
            return None
        return np.array(row_lists)
    except Exception as e:
        st.error(f"Error al procesar la matriz: {str(e)}")
        return None
    
    
def process_input_vector(input_string):
    """Procesa una entrada de string formateada y devuelve un vector numpy."""
    return np.array(list(map(float, input_string.split(','))))

def strategies_page():
    st.markdown("<h1 class='title'>Optimization and Backtesting of Strategies</h1>", unsafe_allow_html=True)
    start_date = st.date_input('Start Date:', date(2019, 1, 1))
    end_date = st.date_input('End Date:', date(2023, 12, 31))
    assets = st.text_area('List of Assets (comma-separated)', 'AAPL, IBM, TSLA, GOOG, NVDA')
    benchmark = st.text_input('Benchmark', '^GSPC')
    rf_rate = st.number_input('Risk-Free Rate', value=0.065, step=0.001)
    initial_capital = st.number_input('Initial Capital', value=1000000, step=100000)
    method = st.selectbox('Optimization Method', ['MonteCarlo', 'SLSQP', 'Genetic', 'Gradient'])

    # Optimization parameters based on method
    params = {}
    if method == 'MonteCarlo':
        params['n_simulations'] = st.number_input('Number of Simulations', min_value=1000, max_value=100000, value=10000, step=1000)
    elif method == 'Genetic':
        params['population_size'] = st.number_input('Population Size', min_value=50, max_value=500, value=100, step=50)
        params['generations'] = st.number_input('Generations', min_value=100, max_value=500, value=200, step=50)
        params['crossover_rate'] = st.slider('Crossover Rate', min_value=0.1, max_value=1.0, value=0.7)
        params['mutation_rate'] = st.slider('Mutation Rate', min_value=0.01, max_value=0.1, value=0.1)
    elif method == 'Gradient':
        params['learning_rate'] = st.slider('Learning Rate', min_value=0.001, max_value=0.1, value=0.01)
        params['max_iters'] = st.number_input('Maximum Iterations', min_value=100, max_value=10000, value=1000)
        params['tol'] = st.slider('Tolerance', min_value=1e-8, max_value=1e-4, value=1e-6, format='%e')

    # Strategy selection
    strategies = [
        "Max Sharpe", "Max (Smart) Sharpe", "Max Omega", "Max (Smart) Omega",
        "Min VaR (Empirical)", "Min VaR (Parametric)", "Semivariance", "Safety-First",
        "Max Sortino", "Risk Parity", "CVaR", "Max Sharpe FF", "HRP", "Black-Litterman"
]
    
    st.markdown("""
        <style>
        .stCheckbox label { font-size: 5px !important; }
        </style>
    """, unsafe_allow_html=True)

    if 'select_all' not in st.session_state:
        st.session_state['select_all'] = False

    selected_strategies = st.multiselect(
        'Select optimization strategies:',
        strategies,
        default=strategies if st.session_state['select_all'] else []
    )

    if st.checkbox('Select/Deselect All', value=st.session_state['select_all'], on_change=lambda: st.session_state.update({'select_all': not st.session_state['select_all']})):
        st.session_state['selected_strategies'] = strategies if st.session_state['select_all'] else []

    if st.session_state['select_all']:
        st.session_state['selected_strategies'] = strategies
    else:
        st.session_state['selected_strategies'] = selected_strategies

    # Fama-French Factors setup if relevant strategies are selected
    ff_factors_expectations = {}
    if any(x in selected_strategies for x in ["Max Sharpe FF"]):
        ff_factors_setup = st.expander("Fama-French Factors Setup")
        with ff_factors_setup:
            ff_factors_expectations = {
                'Mkt-RF': st.number_input('Market Minus Risk-Free (Mkt-RF)', value=0.05, step=0.01),
                'SMB': st.number_input('Small Minus Big (SMB)', value=0.02, step=0.01),
                'HML': st.number_input('High Minus Low (HML)', value=0.03, step=0.01),
                'RF': st.number_input('Risk-Free Rate (RF)', value=0.02, step=0.01)
            }

    def process_input_matrix(input_string):
        """Procesa una entrada de string formateada y devuelve una matriz numpy."""
        matrix = np.array([list(map(float, row.split(','))) for row in input_string.split(';')])
        if matrix.shape[0] != matrix.shape[1]:
            st.error('Error: La matriz Omega debe ser cuadrada.')
            return None
        return matrix

    # En tu función de Streamlit donde procesas las entradas:
    if "Black-Litterman" in selected_strategies:
        st.header('Black Litterman Setup')
        P_input = st.text_area('Matrix P (separate rows with ";", values with ",")', '1,0;0,1')
        Q_input = st.text_area('Vector Q (comma-separated values)', '0.05,0.05')
        Omega_input = st.text_area('Matrix Omega (enter as a square matrix)', '0.01,0,0;0,0.0225,0;0,0,0.0064')
        Omega = process_input_matrix(Omega_input)
        tau = st.number_input('Tau (confidence in equilibrium)', value=0.05, step=0.01)

        P = process_input_matrix(P_input)
        Q = process_input_vector(Q_input)
        if Omega is None:  # Verificar si Omega es cuadrada
            return  # Detener ejecución si hay un error

    if st.button('Optimize Strategies'):
        assets_list = [asset.strip() for asset in assets.split(',')]
        downloader = DataDownloader()
        asset_prices, benchmark_prices, ff_factors = downloader.download_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), assets_list, benchmark)
        asset_allocation = AssetAllocation(asset_prices, benchmark_prices, rf_rate, ff_factors=ff_factors)
        
        if "Black-Litterman" in selected_strategies:
            asset_allocation.set_blacklitterman_expectations(P, Q, tau, Omega)

        results = asset_allocation.Optimize_Portfolio(method=method)
        st.header('Optimization Results')
        st.dataframe(results)


    if st.button('Dynamic Backtest'):
        if start_date and end_date:
            assets_list = [asset.strip() for asset in assets.split(',')]
            backtest = DynamicBacktester(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'), assets=assets_list,
                benchmark=benchmark, initial_capital=initial_capital, strategies=selected_strategies,
                rf=rf_rate, method=method)
            st.header('Backtesting Results')
            backtest.run_backtest()
            fig = backtest.plot_portfolio()  # This now returns a Plotly figure
            st.plotly_chart(fig, use_container_width=True)  # Display the Plotly
        else:
            st.error('Please enter both start and end dates to run the backtest.')

# Display the corresponding page
if st.session_state.page == 'Home':
    main_page()
elif st.session_state.page == 'Strategies':
    strategies_page()
