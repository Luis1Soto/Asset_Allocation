import streamlit as st
from AA import DataDownloader, AssetAllocation, DynamicBacktester
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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


st.markdown("<div class='center'><img src='https://www.pilou.io/wp-content/uploads/2023/08/Logo-PILOU-28.png' width='200'></div>", unsafe_allow_html=True)

# Page setup
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Navigation buttons
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    if st.button("Home"):
        st.session_state.page = 'Home'
with col2:
    if st.button("Download Data"):
        st.session_state.page = 'Download Data'
with col3:
    if st.button("Strategies"):
        st.session_state.page = 'Strategies'
with col4:
    if st.button("Backtesting"):
        st.session_state.page = 'Backtesting'

def main_page():
    st.markdown("<h1 class='title'>Welcome to Backtesting AA Strategies!</h1>", unsafe_allow_html=True)
    st.markdown("<div class='big-font'>This application allows you to analyze, optimize, and perform backtesting of financial portfolios.</div>", unsafe_allow_html=True)
    

def download_data_page():
    st.markdown("<h1 class='title'>Download and Visualize Financial Data</h1>", unsafe_allow_html=True)
    if 'download_data' not in st.session_state:
        st.session_state.download_data = {
            "start_date": date(2019, 1, 1),
            "end_date": date(2023, 12, 31),
            "assets": "AAPL, IBM, TSLA, GOOG, NVDA",
            "benchmark": "^GSPC"
        }
    
    st.session_state.download_data['start_date'] = st.date_input('Start Date:', st.session_state.download_data['start_date'])
    st.session_state.download_data['end_date'] = st.date_input('End Date:', st.session_state.download_data['end_date'])
    st.session_state.download_data['assets'] = st.text_area('List of Assets (comma-separated)', st.session_state.download_data['assets'])
    st.session_state.download_data['benchmark'] = st.text_input('Benchmark', st.session_state.download_data['benchmark'])
    st.session_state.start_date = st.session_state.download_data['start_date']
    st.session_state.end_date = st.session_state.download_data['end_date']
    
    if st.button('Download and Plot Data'):
        assets_list = [asset.strip() for asset in st.session_state.download_data['assets'].split(',')]
        
        # Inicia la barra de progreso
        progress_text = st.empty()
        progress_bar = st.progress(0)
        downloader = DataDownloader()
        progress_text.text('0%')

        progress_bar.progress(25)  # Avanzamos al 25%
        progress_text.text('⏳ 25%')
        
        asset_prices, benchmark_prices, _ = downloader.download_data(
            st.session_state.download_data['start_date'].strftime('%Y-%m-%d'),
            st.session_state.download_data['end_date'].strftime('%Y-%m-%d'),
            assets_list,
            st.session_state.download_data['benchmark']
        )

        progress_bar.progress(50)  # Avanzamos al 50%
        progress_text.text('⏳ 50%')
        
        fig = go.Figure()
        for asset in assets_list:
            fig.add_trace(go.Scatter(x=asset_prices.index, y=asset_prices[asset], mode='lines', name=asset))
        # Corregido para usar st.session_state
        fig.add_trace(go.Scatter(x=benchmark_prices.index, y=benchmark_prices[st.session_state.download_data['benchmark']], mode='lines', name=st.session_state.download_data['benchmark'], line=dict(color='black', width=4)))
        
        fig.update_layout(title='Asset Prices and Benchmark', xaxis_title='Date', yaxis_title='Price', legend_title='Legend')
        st.plotly_chart(fig, use_container_width=True)

        progress_bar.progress(75)  # Avanzamos al 75%
        progress_text.text('75%')

        progress_bar.progress(100)  # Completa la barra al 100%
        progress_text.text('100% - Complete ')

        progress_bar.empty()
        progress_text.empty()

def plot_pie_chart(weights, strategy_name):
    labels = weights.index
    values = weights.values
    # Definir la paleta de colores
    colors = ['#FFC8DE', '#FF5699', '#4B1865', '#D291BC', '#9055A2']  # Paleta de colores personalizada

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                                 insidetextorientation='radial', marker=dict(colors=colors))])
    fig.update_layout(title_text=f'Portfolio Weights for {strategy_name}', title_x=0.5,
                      legend_title="Assets",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig




def strategies_page():
    st.markdown("<h1 class='title'>Optimization and Backtesting of Strategies</h1>", unsafe_allow_html=True)
    if 'download_data' in st.session_state:
        start_date = st.session_state.download_data['start_date']
        end_date = st.session_state.download_data['end_date']
        assets = st.session_state.download_data['assets']
        benchmark = st.session_state.download_data['benchmark']

        st.write("Using data from:")
        st.write(f"**Start Date:** {start_date}")
        st.write(f"**End Date:** {end_date}")
        st.write(f"**Assets:** {assets}")
        st.write(f"**Benchmark:** {benchmark}")
    else:
        st.error("Please download data first on the 'Download Data' page.")
        return  # Salimos de la función si no hay datos descargados

    # Entrada adicional necesaria solo en esta página
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
        "Min VaR (Empirical)", "Min VaR (Parametric)", "Semivariance",
        "Semivariance (Smart)", "Safety-First","Safety-First (Smart)",
        "Max Sortino","Max Sortino (Smart)", "Risk Parity", "CVaR", "Max Sharpe FF","Max Sharpe FF (Smart)", "HRP", "Black-Litterman"
]
    strategy_descriptions = {
    "Max Sharpe": "Optimizes portfolio to maximize the Sharpe ratio.",
    "Max (Smart) Sharpe": "Maximizes a version of Sharpe ratio penalizing autocorrelation in the portfolio.",
    "Max Omega": "Maximizes the Omega ratio, focusing on capturing more gains than losses beyond a threshold.",
    "Max (Smart) Omega": "Maximizes a version of the Omega ratio adjusted for serial correlation in returns.",
    "Min VaR (Empirical)": "Minimizes the portfolio's Value at Risk using historical data.",
    "Min VaR (Parametric)": "Minimizes the portfolio's Value at Risk using a parametric approach.",
    "Semivariance": "Minimizes the semivariance to reduce the portfolio's downside risk.",
    "Semivariance (Smart)": "Minimizes the semivariance to reduce the portfolio's downside risk and penalizing autocorrelation in the portfolio.",
    "Safety-First": "Aims to minimize the probability that portfolio returns fall below a threshold.",
    "Safety-First (Smart)": "Aims to minimize the probability that portfolio returns fall below a threshold and penalizes autocorrelation in the portfolio.",
    "Max Sortino": "Seeks to maximize the Sortino ratio, focusing on downside deviation.",
    "Max Sortino (Smart)": "Seeks to maximize the Sortino ratio, focusing on downside deviation and penalizing autocorrelation in the portfolio.",,
    "Risk Parity": "Aims for equal risk contribution from all portfolio assets.",
    "CVaR": "Minimizes the Conditional Value at Risk, focusing on worst-case scenario losses.",
    "Max Sharpe FF": "Maximizes the Sharpe ratio considering Fama-French factor models.",
    "Max Sharpe FF (Smart)": "Maximizes the Sharpe ratio considering Fama-French factor models and penalizing autocorrelation in the portfolio.",
    "HRP": "Hierarchical Risk Parity approach to diversify risk.",
    "Black-Litterman": "Combines market equilibrium and subjective views for portfolio optimization."
}

    
    st.markdown("""
    <style>
    .stCheckbox label { font-size: 16px !important; }
    </style>
""", unsafe_allow_html=True)

    if 'select_all' not in st.session_state:
        st.session_state['select_all'] = False

    selected_strategies = st.multiselect(
        'Select optimization strategies:',
        strategies,
        default=strategies if st.session_state['select_all'] else [],
        help="Select one or more strategies to view their descriptions below."
    )

    if st.checkbox('Select/Deselect All', value=st.session_state['select_all'], on_change=lambda: st.session_state.update({'select_all': not st.session_state['select_all']})):
        if st.session_state['select_all']:
            selected_strategies = strategies
        else:
            selected_strategies = []

            
            
    if selected_strategies:
        st.write("### Selected Strategy Descriptions:")
        for strategy in selected_strategies:
            st.write(f"**{strategy}:** {strategy_descriptions[strategy]}")
    progress_text = st.empty()
    progress_bar = st.progress(0)
    if st.button('Optimize Strategies'):
        progress_text.text('0%')
        st.session_state['rf_rate'] = rf_rate
        st.session_state['initial_capital'] = initial_capital
        print("##############################################################################")
        print("Saving strategies", selected_strategies)
        st.session_state['selected_strategies'] = selected_strategies
        st.session_state['method'] = method
        
        
        
        assets_list = [asset.strip() for asset in assets.split(',')]
        downloader = DataDownloader()
        asset_prices, benchmark_prices, ff_factors = downloader.download_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), assets_list, benchmark)
        progress_bar.progress(20)
        progress_text.text('⏳ 20%')
        asset_allocation = AssetAllocation(asset_prices, benchmark_prices, rf_rate, ff_factors=ff_factors)

        if "Black-Litterman" in selected_strategies:
            asset_allocation.set_blacklitterman_expectations(P, Q, tau, Omega)

        results = asset_allocation.Optimize_Portfolio(selected_strategies, method=method)
        progress_bar.progress(60)
        progress_text.text('⏳ 60%')

        st.header('Optimization Results')
        # Aplicamos formato al dataframe antes de mostrarlo
        # Formatear todas las celdas excepto las de 'Optimized Value'
        formatted_results = results.style.format({
            col: "{:.2%}" for col in results.columns if col != 'Optimized Value'  # Aplica formato de porcentaje solo a las columnas que no son 'Optimized Value'
        }).set_properties(**{
            'text-align': 'right',
            'color': 'black',
            'font-weight': 'bold',
            'background-color': 'white'
        }).set_table_styles([{
            'selector': 'th',
            'props': [('font-size', '16px'), ('text-align', 'center'), ('background-color', 'purple'), ('color', 'white')]
        }])
        st.dataframe(formatted_results)

        # Plot para cada estrategia, incluyendo HRP si está presente
        for strategy in results.index:
            weights = results.loc[strategy, results.columns != 'Optimized Value']  # Excluir la columna de valores optimizados
            if not weights.empty:
                fig = plot_pie_chart(weights, strategy)
                st.plotly_chart(fig, use_container_width=True)
        progress_bar.progress(80)
        progress_text.text('⏳ 80%')

            
                  
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
    
    progress_bar.progress(100)
    progress_text.text('100% ⌛️')
    progress_bar.empty()
    progress_text.empty()
            
            
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
        
    # Botón Dynamic Backtest modificado
    #if st.button('Dynamic Backtest'):
        st.session_state.page = 'Backtesting'

def backtesting_page():
    if 'selected_strategies' in st.session_state and 'start_date' in st.session_state and 'end_date' in st.session_state:
        rf_rate = st.session_state.get('rf_rate', 0.065)
        initial_capital = st.session_state.get('initial_capital', 1000000)
        selected_strategies = st.session_state.get('selected_strategies', [])
        method = st.session_state.get('method', 'MonteCarlo')
        
        assets = st.session_state.download_data['assets']
        benchmark = st.session_state.download_data['benchmark']
        start_date = st.session_state.download_data['start_date']
        end_date = st.session_state.download_data['end_date']
        
        progress_text = st.empty()
        progress_bar = st.progress(0)
        progress_text.text('0%')
        
        assets_list = [asset.strip() for asset in assets.split(',')]
        progress_bar.progress(25)  # Avanzamos al 25%
        progress_text.text('⏳ 25%')

        print("################################################################################################")
        print(st.session_state)

        backtest = DynamicBacktester(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'), assets=assets_list,
            benchmark=benchmark, initial_capital=initial_capital, strategies=selected_strategies,
            rf=rf_rate, method=method)
        progress_bar.progress(50)  # Avanzamos al 25%
        progress_text.text('⏳ 50%')
        st.header('Backtesting Results')
        backtest.run_backtest()
        fig = backtest.plot_portfolio()
        st.plotly_chart(fig, use_container_width=True)
        progress_bar.progress(100)
        progress_text.text('100% ⌛️')
        progress_bar.empty()
        progress_text.empty()
    else:
        st.error('Please ensure all required fields are filled on the "Strategies" page before running the backtest.')
# Display the corresponding page
if st.session_state.page == 'Home':
    main_page()
elif st.session_state.page == 'Download Data':
    download_data_page()

elif st.session_state.page == 'Strategies':
    strategies_page()
    
elif st.session_state.page == 'Backtesting':
    backtesting_page()
