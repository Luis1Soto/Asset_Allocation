import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
from scipy.stats import norm

class AssetDataDownloader:
    
    def __init__(self):
        """
        Initializes the AssetDataDownloader object. This class is designed to download
        adjusted closing prices for specified financial assets and a benchmark over a given period.
        """
        pass

    def download_data(self, start_date: str, end_date: str, assets: list, benchmark: str) -> tuple:
        """
        Downloads adjusted closing prices for a list of assets and a benchmark
        for the specified period.

        :param start_date: Start date for data download, in 'YYYY-MM-DD' format.
        :param end_date: End date for data download, in 'YYYY-MM-DD' format.
        :param assets: List of strings with the tickers of the assets to download.
        :param benchmark: String with the ticker of the benchmark to download.
        :return: A tuple containing two DataFrames: the first with the adjusted closing prices
                 of the assets and the second with the adjusted closing prices of the benchmark.
        """
        
        # Download adjusted closing prices for the assets and the given benchmark
        asset_data = yf.download(assets, start=start_date, end=end_date)['Adj Close']
        benchmark_data = pd.DataFrame(yf.download(benchmark, start=start_date, end=end_date)['Adj Close'])
        benchmark_data = benchmark_data.rename(columns={'Adj Close': benchmark})

        return pd.DataFrame(asset_data), benchmark_data
    

class AssetAllocation:
    
    def __init__(self, asset_prices, benchmark_prices, rf):
        """
        Initializes the AssetAllocation object with asset and benchmark prices.
        :param asset_prices: DataFrame with rows as dates and columns as asset tickers, containing the prices of each asset.
        :param benchmark_prices: DataFrame with rows as dates and a column with the benchmark ticker, containing the Adjusted close price.
        :param rf: Risk Free Rate for the given period
        """
        self.asset_prices= asset_prices
        self.num_assets = len(asset_prices.columns)
        self.asset_returns= asset_prices.pct_change().dropna()
        self.average_asset_returns= self.asset_returns.mean().values
        self.asset_cov_matrix = self.asset_returns.cov()
        self.corr_matrix = self.asset_returns.corr()
        
        self.benchmark_prices= benchmark_prices
        self.benchmark_returns = benchmark_prices.pct_change().dropna()
        self.average_benchmark_returns = self.benchmark_returns.mean().item()
        
        self.start_weights = np.full(self.num_assets, 1 / self.num_assets)
        self.constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # the assets summation is required to be = 1
        self.bounds = tuple((0.1, 1) for _ in range(self.num_assets))  #each asset will represent at least 1% of the portfolio
        
        self.rf = rf
        self.rf_daily = rf / 252
        
        self.portfolio_value = self.asset_prices.dot(self.start_weights) 
        self.portfolio_market_gap = self.asset_returns - self.benchmark_returns.values
        self.downside = self.portfolio_market_gap[self.portfolio_market_gap < 0].fillna(0).std()
        self.upside = self.portfolio_market_gap[self.portfolio_market_gap > 0].fillna(0).std()
        self.assets_omega = self.upside / self.downside
  

    @staticmethod
    def portfolio_volatility(weights, asset_cov_matrix):
        """
        Calculates the portfolio volatility given asset weights and covariance matrix.

        :param weights: Weights of the assets in the portfolio.
        :param cov_matrix: Covariance matrix of asset returns.
        :return: Portfolio volatility.
        """
        return np.sqrt(np.dot(weights.T, np.dot(asset_cov_matrix, weights)))

    @staticmethod
    def greeks():
        """Calculates alpha and beta of the portfolio"""
        matrix = np.cov(self.asset_returns, self.benchmark_returns)
        beta = matrix[0, 1] / matrix[1, 1]
     
        alpha = self.asset_returns.mean() - beta * self.benchmark_returns.mean()
        alpha = alpha * 252
        
        return beta, alpha
    
    
    def neg_sharpe_ratio(self, weights):
        """
        Calculates the negative Sharpe Ratio for a given set of asset weights.

        :param weights: Weights of the assets in the portfolio.
        :param average_asset_returns: Average daily returns of the assets.
        :param cov_matrix: Covariance matrix of asset returns.
        :param rf_daily: Daily risk-free rate.
        :return: Negative Sharpe Ratio.
        """
        sharpe_ratio = (np.dot(weights, self.average_asset_returns) - self.rf_daily) / self.portfolio_volatility(weights, self.asset_cov_matrix)
        return -sharpe_ratio  
    
    def optimize_max_sharpe(self):
        """
        Optimizes the portfolio to maximize the Sharpe Ratio.
        The Sharpe Ratio is maximized by adjusting the asset weights in the portfolio,
        subject to the constraints that all weights sum to 1 and each weight is between the specified bounds. 
        This method utilizes the 'SLSQP' optimization algorithm
        
        Returns:
            optimized_weights (np.ndarray): The asset weights that maximize the Sharpe Ratio.
            optimized_sharpe (float): The maximum Sharpe Ratio achieved by the optimized portfolio.
    
        Method used:
            'SLSQP' - Sequential Least Squares Programming, a gradient-based optimization algorithm
            suitable for constrained optimization problems, including both equality and inequality constraints.
        """       
        result = sco.minimize(self.neg_sharpe_ratio, self.start_weights, method='SLSQP', bounds=self.bounds, constraints=self.constraints)
        
        optimized_weights = result.x
        optimized_sharpe = -result.fun
        
        return optimized_weights, optimized_sharpe


    def neg_omega_ratio(self, weights):
        """
        Returns the negative Omega ratio for optimization purposes.
        :return: Negative Omega ratio of the portfolio.
        """
        portfolio_omega = np.dot(weights, self.assets_omega)
        
        return -portfolio_omega

    def optimize_for_omega(self) :
        """
        Optimizes the portfolio to maximize the Omega Ratio.
        The Omega Ratio is maximized by adjusting the asset weights in the portfolio,
        subject to the constraints that all weights sum to 1 and each weight is between the specified bounds. 
        This method utilizes the 'SLSQP' optimization algorithm
        
        Returns:
            optimized_weights (np.ndarray): The asset weights that maximize the Omega Ratio.
            optimized_sharpe (float): The maximum Omega Ratio achieved by the optimized portfolio.
    
        Method used:
            'SLSQP' - Sequential Least Squares Programming, a gradient-based optimization algorithm
            suitable for constrained optimization problems, including both equality and inequality constraints.
        """  
        result = sco.minimize(self.neg_omega_ratio, self.start_weights, method='SLSQP', bounds=self.bounds, constraints=self.constraints)
        optimized_weights = result.x
        optimized_omega = -result.fun
        
        return optimized_weights, optimized_omega

    def portfolio_var(self, weights):
        """
        Calcula el Value at Risk (VaR) del portafolio utilizando el enfoque de varianza-covarianza.

        :param weights: Pesos de los activos en el portafolio.
        :return: VaR del portafolio.
        """
        assets_average_return = np.dot(weights, self.average_asset_returns)
        portfolio_volatility = self.portfolio_volatility(weights, self.asset_cov_matrix)
        
        confidence_level = 0.05
        
        # VaR as standard deviation multiplied by th value Z of the normal distribution
        VaR = norm.ppf(1 - confidence_level) * portfolio_volatility
        
        return VaR 

#    
#    def portfolio_var_emp(self, weights):
#        """
#        Calcula el Value at Risk (VaR) del portafolio utilizando el enfoque de varianza-covarianza.
#
#        :param weights: Pesos de los activos en el portafolio.
#        :return: VaR del portafolio.
#        """
#        
#        np.dot(weights.T, weights)
#        portfolio_volatility = self.portfolio_volatility(weights, self.asset_cov_matrix)
#        
#        confidence_level = 0.05
#        
#        # VaR as standard deviation multiplied by th value Z of the normal distribution
#        VaR = norm.ppf(1 - confidence_level) * portfolio_volatility
#        
#        return VaR    
#    
#    
    def minimize_var(self):
        """
        Optimiza los pesos de los activos para minimizar el Value at Risk (VaR) del portafolio.
        
        :return: Pesos optimizados del portafolio.
        """
        opts = sco.minimize(self.portfolio_var, self.start_weights, method='SLSQP', bounds=self.bounds, constraints=self.constraints)
        optimized_weights = opts.x
        optimized_VaR = opts.fun
        
        return optimized_weights, optimized_VaR 
    
    
    def run_optimizations(self):
        """
        Executes the class's optimizations and compiles the results into a DataFrame.

        Returns:
            df_optimizations (pd.DataFrame): A DataFrame where columns represent the types of optimization and
            an additional column for the value of the optimization indicator, rows are the asset names, and the values are the optimized weights in percentage.
        """
        # Run optimizations
        optimized_weights_sharpe, max_sharpe_value = self.optimize_max_sharpe()
        optimized_weights_omega, max_omega_value = self.optimize_for_omega()
        optimized_weights_minvar_n, minvar_n_value = self.minimize_var()
        
        

        # Create results DataFrame
        df_optimizations = pd.DataFrame({
            'Max Sharpe Ratio Weights (%)': optimized_weights_sharpe * 100,
            'Min VaR Ratio Weights (%)': optimized_weights_sharpe * 100,
            'Max Omega Ratio Weights (%)': optimized_weights_omega * 100
        }, index=self.asset_prices.columns)  # Assuming asset names are in the columns of the asset_prices DataFrame

        # Adding a row for the optimization indicator values
        df_optimizations.loc['Optimization Value'] = [max_sharpe_value, max_omega_value]

        return df_optimizations    
    
    
   


























































    def get_optimized_weights(self):
        max_sharpe_weights, _ = self.optimize_max_sharpe()
        df_optimized_weights = pd.DataFrame([max_sharpe_weights], columns=self.asset_prices.columns)
        return df_optimized_weights  

      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    