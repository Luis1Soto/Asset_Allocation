import streamlit as st
from AA import DataDownloader, AssetAllocation

# Custom Styles
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

# Display logo at the top of all pages
st.image("https://www.pilou.io/wp-content/uploads/2023/08/Logo-PILOU-28.png", width=200)

# Page active setup
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Navigation buttons bar
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Home"):
        st.session_state.page = 'Home'
with col2:
    if st.button("Strategies"):
        st.session_state.page = 'Strategies'

# Functions to handle each page
def main_page():
    st.title('Welcome to Backtesting AA Strategies!')
    st.markdown("""
        This application allows you to analyze, optimize, and perform backtesting of financial portfolios.
    """)

def strategies_page():
    st.title('Optimization and Backtesting of Strategies')
    start_date = st.date_input('Start Date')
    end_date = st.date_input('End Date')
    assets = st.text_area('List of Assets (comma-separated)', 'AAPL, IBM, TSLA, GOOG, NVDA')
    benchmark = st.text_input('Benchmark', '^GSPC')
    rf_rate = st.number_input('Risk-Free Rate', value=0.065, step=0.001)
    method = st.selectbox('Optimization Method', ['MonteCarlo', 'SLSQP', 'Genetic', 'Gradient'])

    strategies = [
        "Max Sharpe", "Max (Smart) Sharpe", "Max Omega", "Max (Smart) Omega",
        "Min VaR (Empirical)", "Min VaR (Parametric)", "Semivariance", "Safety-First",
        "Max Sortino", "Risk Parity", "CVaR", "Max Sharpe FF", "HRP", "Black-Litterman"
    ]
    selected_strategies = st.multiselect('Select optimization strategies:', strategies, default=strategies)

    if "Black-Litterman" in selected_strategies:
        st.header('Black Litterman Setup')
        P = st.text_area('Matrix P (separate rows with ";", values with ",")', '1,0;0,1')
        Q = st.text_area('Vector Q (comma-separated values)', '0.05,0.05')
        Omega = st.text_area('Matrix Omega (separate rows with ";", values with ",")', '0.01,0;0,0.01')
        tau = st.number_input('Tau (confidence in equilibrium)', value=0.05, step=0.01)

    if method == 'MonteCarlo':
        n_simulations = st.number_input('Number of Simulations', min_value=1000, max_value=100000, value=10000, step=1000)
    elif method == 'Genetic':
        population_size = st.number_input('Population Size', min_value=50, max_value=500, value=100, step=50)
        generations = st.number_input('Generations', min_value=100, max_value=500, value=200, step=50)
        crossover_rate = st.slider('Crossover Rate', min_value=0.1, max_value=1.0, value=0.7)
        mutation_rate = st.slider('Mutation Rate', min_value=0.01, max_value=0.1, value=0.1)
    elif method == 'Gradient':
        learning_rate = st.slider('Learning Rate', min_value=0.001, max_value=0.1, value=0.01)
        max_iters = st.number_input('Maximum Iterations', min_value=100, max_value=10000, value=1000)
        tol = st.slider('Tolerance', min_value=1e-8, max_value=1e-4, value=1e-6, format='%e')

    if st.button('Optimize Strategies'):
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

            st.header('Optimization Results')
            st.dataframe(results)
        else:
            st.error('Please enter both start and end dates.')

# Display the corresponding page
if st.session_state.page == 'Home':
    main_page()
elif st.session_state.page == 'Strategies':
    strategies_page()
