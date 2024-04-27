import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import pandas_datareader.data as web
import datetime as dt
import statsmodels.api as sm
from scipy.stats import norm
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
class DataDownloader:
    
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
        
        # Download Fama-French three factors
        try:
            factors = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start=start_date, end=end_date)[0]
            factors = factors # / 100  # Convert to decimal form
        except Exception as e:
            print("Error downloading Fama-French factors:", e)
            factors = pd.DataFrame()  # Return an empty DataFrame in case of error

        return pd.DataFrame(asset_data), benchmark_data, factors

class HierarchicalRiskParity:
    
    def __init__(self, returns):
        """
        Initialize the HierarchicalRiskParity object with returns data.

        :param returns: Historical returns of assets.
        :type returns: pd.DataFrame
        """
        self.names = returns.columns
        self.returns = returns
        self.cov = returns.cov()
        self.corr = returns.corr()
        
        self.link = None
        self.sort_ix = None
        self.weights = None

    def get_quasi_diagonalization(self, link):
        """
        Perform quasi-diagonalization ordering on a linkage matrix.

        :param link: Linkage matrix from hierarchical clustering.
        :type link: np.ndarray
        :return: Ordered indices based on quasi-diagonalization.
        :rtype: list
        """
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])

        return sort_ix.tolist()
   
    def cluster_variation(self, cov, c_items):
        """
        Calculate the variance of a cluster.

        :param cov: Covariance matrix of returns.
        :type cov: pd.DataFrame
        :param c_items: Indices of items in the cluster.
        :type c_items: list
        :return: Variance of the cluster.
        :rtype: float
        """
        c_cov = cov.iloc[c_items, c_items]
        
        ivp_weights = 1. / np.diag(c_cov)
        ivp_weights /= ivp_weights.sum()
        ivp_weights = ivp_weights.reshape(-1, 1)
        
        cluster_variance = np.dot(np.dot(ivp_weights.T, c_cov), ivp_weights)[0, 0]
        return cluster_variance

    @staticmethod
    def bisection(items):
        """
        Recursively split clusters into subclusters.

        :param items: List of cluster indices.
        :type items: list
        :return: List of subcluster indices after bisection.
        :rtype: list
        """
        new_items = [i[int(j):int(k)] for i in items for j, k in ((0, len(i) / 2), (len(i) / 2, len(i))) if len(i) > 1]
        return new_items

    def hrp_allocation(self, cov, sort_ix):
        """
        Calculate asset weights using hierarchical risk parity.

        :param cov: Covariance matrix of returns.
        :type cov: pd.DataFrame
        :param sort_ix: Ordered indices from quasi-diagonalization.
        :type sort_ix: list
        :return: Asset weights based on HRP.
        :rtype: pd.Series
        """
        w = pd.Series(1, index=sort_ix)
        clusters = [sort_ix]

        while len(clusters) > 0:
            clusters = self.bisection(clusters)

            for i in range(0, len(clusters), 2):
                cluster_0 = clusters[i]
                cluster_1 = clusters[i + 1]

                c_var0 = self.cluster_variation(cov, cluster_0)
                c_var1 = self.cluster_variation(cov, cluster_1)

                alpha = 1 - c_var0 / (c_var0 + c_var1)

                w[cluster_0] *= alpha
                w[cluster_1] *= 1 - alpha

        return w

    def optimize_hrp(self, linkage_method='single'):
        """
        Optimize asset weights using the specified linkage method.

        :param linkage_method: Method used for hierarchical clustering.
        :type linkage_method: str
        :return: Optimized asset weights based on HRP.
        :rtype: pd.Series
        """
        self.link = linkage(squareform(1 - self.corr), method=linkage_method)
        self.sort_ix = self.get_quasi_diagonalization(self.link)
        sorted_weights = self.hrp_allocation(self.cov, self.sort_ix)
        self.weights = pd.Series(index=self.names, dtype=float)

        for i, ix in enumerate(self.sort_ix):
            self.weights.iloc[ix] = sorted_weights.iloc[i]
        self.weights.name = "HRP"
        
        return self.weights

    
class AssetAllocation:
    
    def __init__(self, asset_prices, benchmark_prices, rf, ff_factors_expectations=None, ff_factors=None, bounds=None, RMT_filtering=False):
        """
        Initializes the AssetAllocation object with asset and benchmark prices.
        :param asset_prices: DataFrame with rows as dates and columns as asset tickers, containing the prices of each asset.
        :param benchmark_prices: DataFrame with rows as dates and a column with the benchmark ticker, containing the Adjusted close price.
        :param rf: Risk Free Rate for the given period
        :param bounds: Optional. A tuple of tuples specifying the minimum and maximum weights for each asset. 
        Default is (0.01, 1) for each asset (minimun 1% and maximum 10%).
        """
        self.ff_expected_returns = None
        self.asset_prices= asset_prices
        self.num_assets = len(asset_prices.columns)
        self.asset_returns= asset_prices.pct_change().dropna()
        self.average_asset_returns= self.asset_returns.mean().values
        self.asset_cov_matrix = self.asset_returns.cov()
        if RMT_filtering == True:
            self.RMT_filtering()
        self.corr_matrix = self.asset_returns.corr()
        
        self.benchmark_prices= benchmark_prices
        self.benchmark_returns = benchmark_prices.pct_change().dropna()
        self.average_benchmark_returns = self.benchmark_returns.mean().item()
        
        self.start_weights = np.full(self.num_assets, 1 / self.num_assets)
        self.constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # the assets summation is required to be = 1
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = tuple((0.01, 1) for _ in range(self.num_assets)) # Default limits for every asset (min 1% - max 100%)
        self.rf = rf
        self.rf_daily = rf / 252
        self.ff_factors = ff_factors
        
        self.portfolio_market_gap = self.asset_returns - self.benchmark_returns.values
        self.downside = self.portfolio_market_gap[self.portfolio_market_gap < 0].fillna(0).std()
        self.upside = self.portfolio_market_gap[self.portfolio_market_gap > 0].fillna(0).std()
        self.assets_omega = self.upside / self.downside
        
        self.P = None
        self.Q = None
        self.tau = None
        self.Omega =   None 
        self.bl_expectations_set = False
        
        self.calculate_ff_expected_returns(ff_factors_expectations)
        
    
    def calculate_ff_expected_returns(self, ff_factors_expectations):
        expected_returns = {}
        fallback_return = self.asset_returns.mean().mean()  
    
        for asset in self.asset_prices.columns:
            aligned_returns = self.asset_returns[asset].dropna()
            aligned_factors = self.ff_factors.loc[aligned_returns.index]
            
            if len(aligned_returns) < 20:  
                expected_returns[asset] = fallback_return
                continue
    
            X = sm.add_constant(aligned_factors)
            y = aligned_returns
    
            try:
                model = sm.OLS(y, X).fit()
                betas = model.params
                expected_return = betas['const'] + sum(betas[factor] * ff_factors_expectations[factor] for factor in ['Mkt-RF', 'SMB', 'HML'])
                expected_returns[asset] = expected_return
            except Exception as e:
                print(f"Failed to calculate expected returns for {asset}: {str(e)}")
                expected_returns[asset] = fallback_return  
    
        self.ff_expected_returns = pd.Series(expected_returns)
                
    def RMT_filtering(self):
        """
        Filters the covariance matrix of asset returns using Random Matrix Theory (RMT)
        to reduce noise.
    
        This method reconstructs the covariance matrix using only the eigenvalues that are considered to be signal,
        based on the upper bound of the Marchenko-Pastur distribution.
        """
        
        eigenvalues, eigenvectors = np.linalg.eigh(self.asset_cov_matrix)
        
        T, N = self.asset_returns.shape
        lambda_plus = (1 + np.sqrt(N/T))**2
        filtered_eigenvalues = np.clip(eigenvalues, 0, lambda_plus)
        filtered_cov_matrix = eigenvectors @ np.diag(filtered_eigenvalues) @ eigenvectors.T
        
        self.asset_cov_matrix = filtered_cov_matrix
    
    
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
    def greeks(asset_returns,benchmark_returns):
        """
        Calculates alpha and beta indicators of the portfolio
        :return: tuple containing the (beta, alpha).
        
        """
        matrix = np.cov(asset_returns, benchmark_returns)
        beta = matrix[0, 1] / matrix[1, 1]
     
        alpha = asset_returns.mean() - beta * benchmark_returns.mean()
        alpha = alpha * 252
        
        return beta, alpha
    
    @staticmethod
    def get_optport_value_n_returns(optimized_weights, asset_prices):
        """
        Calculates the portfolio volatility given asset weights and covariance matrix.

        :param weights: Weights of the assets in the portfolio.
        :param cov_matrix: Covariance matrix of asset returns.
        :return: Portfolio volatility.
        """        
        portfolio_value = asset_prices.dot(optimized_weights)
        
        return  portfolio_value, portfolio_value.pct_change().dropna()

    @staticmethod
    def check_constraints(weights):
        """
        Check if the generated weights fulfill the sum-to-one constraint with a limit difference of 0.00005.

        Parameters:
        - weights: Array of portfolio weights to be checked.

        Returns:
        - Boolean: True if the sum of weights is approximately equal to 1, False otherwise.
        """
        return np.abs(1 - np.sum(weights)) < 1e-5

    @staticmethod
    def check_bounds(weights, bounds):
        """
        Check if the portfolio weights are within the specified bounds.

        Parameters:
        - weights: Array of portfolio weights to be checked.
        - bounds: A sequence of (min, max) pairs for each weight, defining the bounds.

        Returns:
        - Boolean: True if all weights are within their respective bounds, False otherwise.
        """
        return all(bounds[i][0] <= weights[i] <= bounds[i][1] for i in range(len(weights)))   
            
            
    @staticmethod
    def optimize_MonteCarlo(objective_function, start_weights, bounds, args=(), n_simulations=10000):
        """
        Optimize portfolio weights using a Monte Carlo simulation approach.

        This method randomly generates portfolio weights and evaluates the objective function
        for each set of weights, aiming to find the set that minimizes the objective function
        while adhering to specified bounds and constraints.

        Parameters:
        - objective_function: The function to be minimized. This could represent various
          portfolio metrics such as volatility, negative of the expected return, or Sharpe ratio.
        - start_weights: Initial guess for the portfolio weights, used to determine the number
          of assets in the portfolio.
        - bounds: A sequence of (min, max) pairs for each weight, defining the bounds on
          the variables.
        - args: Additional arguments to pass to the objective function, used to enable variations on the optimizations, 
        for example: Sharpe/Smart Sharpe, Cov-Var/Empirical Min-VaR.
        - n_simulations: The number of random weight combinations to simulate.

        Returns:
        - best_weights: The optimized weights for the portfolio that minimize the objective
          function within the given bounds and constraints.
        - best_value: The minimum value of the objective function found during the simulations.

        Methodology:
        The Monte Carlo simulation method generates a large number of random portfolios by
        assigning random weights to each asset in the portfolio, subject to the sum of weights
        being equal to 1. Each portfolio's weights are generated using the Dirichlet distribution,
        which naturally ensures that the weights sum up to 1 and can stay within specified bounds.
        The objective function is evaluated for each set of weights, and the set that results in
        the minimum value of the objective function is retained as the best solution. This approach
        does not guarantee finding the global minimum but can provide a good approximation, especially
        when the number of simulations (n_simulations) is large.
        """
        
        num_assets = len(start_weights)
        best_value = float('inf')  #to minimize
        best_weights = np.zeros(num_assets)        
        alpha = np.ones(num_assets)
        #Montecarlo Simulation
        for _ in range(n_simulations):
            weights = np.random.dirichlet(alpha)

            # Check for restriccions fullfillment
            if not AssetAllocation.check_constraints(weights):
                continue
            value = objective_function(weights, *args)
            if AssetAllocation.check_bounds(weights, bounds) and value < best_value:  # minimizing
                best_value = value
                best_weights = weights

        return best_weights, best_value           

    @staticmethod
    def optimize_Genetic(objective_function, start_weights, bounds, constraints, population_size=100, generations=200, crossover_rate=0.7, mutation_rate=0.1, args=()):
        num_assets = len(start_weights)

        # Function to create an individual(Solution)
        def create_individual():
            return np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=num_assets)

        # Function to evaluate Fitness
        def calculate_fitness(individual):
            # ensure the weights sum 1
            individual = individual / np.sum(individual)
            return objective_function(individual, *args)

        # Function to cross 2 fathers to create a son
        def crossover(parent1, parent2):
            if np.random.rand() < crossover_rate:
                crossover_point = np.random.randint(1, num_assets)
                child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            else:
                child = parent1.copy()
            return child

        # Mutation Function
        def mutate(individual):
            for i in range(num_assets):
                if np.random.rand() < mutation_rate:
                    individual[i] = np.random.uniform(bounds[i][0], bounds[i][1])
            return individual

        # Create initial population
        population = [create_individual() for _ in range(population_size)]
        for _ in range(generations):
            # Evaluate fitness
            fitness = np.array([calculate_fitness(ind) for ind in population])
            #selec most fit population
            selection_probs = fitness / fitness.sum()
            selected_indices = np.random.choice(range(population_size), size=population_size, replace=True, p=selection_probs)
            selected_individuals = [population[i] for i in selected_indices]

            # Create new generation
            next_generation = []
            for i in range(0, population_size, 2):
                parent1, parent2 = selected_individuals[i], selected_individuals[i+1]
                child1 = mutate(crossover(parent1, parent2))
                child2 = mutate(crossover(parent1, parent2))
                next_generation.extend([child1, child2])

            population = next_generation

        # Find best individual from last generation
        final_fitness = np.array([calculate_fitness(ind) for ind in population])
        best_index = np.argmin(final_fitness)
        best_individual = population[best_index]

        # Asegurarse de que los pesos del mejor individuo sumen 1
        best_individual = best_individual / np.sum(best_individual)

        return best_individual, objective_function(best_individual)
    
            
    @staticmethod
    def optimize_SLSQP(objective_function, start_weights, bounds, constraints, args=()):
        """
        Optimize portfolio weights using Sequential Least Squares Programming (SLSQP).

        This method solves an optimization problem where the goal is to find the portfolio
        weights that minimize (or maximize) a given objective function, subject to various
        constraints on the weights.

        Parameters:
        - objective_function: The function to be minimized. For portfolio optimization, this
          could be the negative of the expected return, volatility, or Sharpe ratio, etc.
        - start_weights: Initial guess for the portfolio weights. Must be a 1-D array of
          the same length as the number of assets in the portfolio.
        - bounds: A sequence of (min, max) pairs for each weight, defining the bounds on
          the variables. Use None for one of min or max when there is no bound in that direction.
        - constraints: A dictionary or a list of dictionaries specifying the constraints
          the solution must satisfy. Each dictionary contains fields 'type' (equality or inequality)
          and 'fun' that is a function defining the constraint.
        - args: Additional arguments to pass to the objective function, used to enable variations on the optimizations, 
        for example: Sharpe/Smart Sharpe, Cov-Var/Empirical Min-VaR.

        Returns:
        - optimized_weights: The optimized weights for the portfolio that minimize the objective
          function within the given bounds and constraints.
        - optimized_value: The value of the objective function evaluated at the optimized weights.

        Methodology:
        The SLSQP (Sequential Least Squares Programming) optimization method is used, which is
        suitable for solving nonlinear programming problems with both equality and inequality
        constraints. This method is part of the broader category of Sequential Quadratic Programming
        (SQP) methods. The SQP methods, including SLSQP, are designed to solve optimization problems
        by constructing a sequence of quadratic subproblems, each of which approximates the
        original problem but is easier to solve. SLSQP specifically is efficient for problems
        where the objective function and the constraints are smooth functions of the optimization
        variables, making it well-suited for portfolio optimization where such smoothness
        is often present.
        """
        result = sco.minimize(objective_function, start_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        optimized_weights = result.x
        optimized_value = result.fun
       

        return optimized_weights, optimized_value

    @staticmethod
    def optimize_Gradient(objective_function, start_weights, bounds, learning_rate=0.01, max_iters=1000, tol=1e-6, args=()):
        """
        Optimizes portfolio weights using the gradient descent method.

        Parameters:
        - objective_function: The function to be minimized.
        - start_weights: Initial guess for the portfolio weights.
        - bounds: A sequence of (min, max) pairs for each weight, defining the bounds.
        - args: Additional arguments to pass to the objective function.
        - learning_rate: Step size for each iteration.
        - max_iters: Maximum number of iterations to perform.
        - tol: Tolerance for the stopping criterion. The algorithm stops if the improvement between two iterations is less than this value.

        Returns:
        - weights: Optimized portfolio weights.
        """
        weights = np.array(start_weights)
        prev_val = float('inf')

        for _ in range(max_iters):
            # Calculate the gradient of the objective function
            grad = AssetAllocation.calculate_gradient(objective_function, weights, args)

            # Update weights - move in the opposite direction of the gradient
            new_weights = weights - learning_rate * grad

            # Apply bounds
            new_weights = np.clip(new_weights, a_min=[b[0] for b in bounds], a_max=[b[1] for b in bounds])

            # Ensure weights sum to 1 
            new_weights /= np.sum(new_weights)

            # Check for convergence
            new_val = objective_function(new_weights, *args)
            if np.abs(new_val - prev_val) < tol:
                break

            weights = new_weights
            prev_val = new_val

        return weights, objective_function(weights)

    @staticmethod
    def calculate_gradient(objective_function, weights, args=()):
        """
        Calculates the gradient of the objective function with respect to the weights.
        
        Parameters:
        - objective_function: The function for which the gradient is calculated.
        - weights: Current weights at which the gradient is evaluated.
        - args: Additional arguments to the objective function.

        Returns:
        - grad: The gradient of the objective function at the given weights.
        """
        epsilon = 1e-8  # Small change to compute the numerical gradient
        grad = np.zeros_like(weights)

        for i in range(len(weights)):
            weights_plus = np.array(weights)
            weights_minus = np.array(weights)
            weights_plus[i] += epsilon
            weights_minus[i] -= epsilon

            # Evaluate function at neighboring points
            f_plus = objective_function(weights_plus, *args)
            f_minus = objective_function(weights_minus, *args)

            # Use central difference to approximate gradient
            grad[i] = (f_plus - f_minus) / (2 * epsilon)

        return grad    
 
    
    def portfolio_autocorr_penalty(self, weights):
        """
        Calculates the autocorrelation penalty for the portfolio returns based on the given asset weights. 
        This penalization is used to get "Smart" ratios or indicators.
        
        Methodology:
        
        1. Returns of the portfolio are calculated with a dot product among the weights of the assets and their respective returns.
        
        2. The returns are used to calculate the correlation coefficient of the portfolio with itself (Autocorrelation)
        
        3. The 'corr' list contains values representing the autocorrelation of the portfolio returns adjusted by the number of observations. Each value in corr is calculated as:
        
        ((num - x) / num) * coef**x
        
        where:
            'x' is the time lag between observations 
            'num' is the total number of observations, and 
            'coef' is the autocorrelation coefficient. 
            
        This formula adjusts the autocorrelation coefficient for the time lag, giving less weight to autocorrelations with larger lags. 
        
        4. Summing the adjusted autocorrelation values contained in 'corr' provides an aggregated measure of autocorrelation across the portfolio, 
        taking into account how this autocorrelation diminishes over time.
        
        5. After summing the adjusted autocorrelation values, the result is multiplied by 2. This is done because in time series theory,
        specifically in the adjustment of variance for a sum of autocorrelated observations, the autocorrelation term is counted twice for each pair of observations. 
        
        6. Adding 1 at the end ensures that the minimum penalty is 1, even in the absence of autocorrelation, we wouldn't want the penalty to be 0, 
        as that would imply no adjustment to the ratio or indicator being calculated.
        
        7. Finally, the square root of the total is taken. Since the variance of a sum of random variables (in this case, the portfolio returns over time)
        is proportional to the sum of their covariances. By taking the square root, we convert this aggregated measure of adjusted variance
        (which has been inflated by the autocorrelation and by the factor of 2) back to a scale comparable to that of the original returns,
        providing a penalty that is proportional to the level of "risk" introduced by autocorrelation.
    
        :param weights: Weights of the assets in the portfolio.
        :return: Autocorrelation penalty for the portfolio returns.
        """
        # Calculate portfolio returns based on weights and asset returns
        portfolio_returns = np.dot(self.asset_returns, weights)

        # Calculate the autocorrelation coefficient for the portfolio returns
        num = len(portfolio_returns)
        coef = np.abs(np.corrcoef(portfolio_returns[:-1], portfolio_returns[1:])[0, 1])
        corr = [((num - x) / num) * coef**x for x in range(1, num)]

        # Calculate and return the autocorrelation penalty
        return np.sqrt(1 + 2 * np.sum(corr))

    def neg_sharpe_ratio(self, weights, Smart=False):
        """
        Calculates the negative annualized Sharpe Ratio for a given set of asset weights, optionally adjusting for portfolio autocorrelation.
    
        :param weights: Weights of the assets in the portfolio.
        :param Smart: Boolean flag to apply autocorrelation penalty to the Sharpe Ratio.
        :return: Negative annualized Sharpe Ratio.
        """
        trading_days = 252
    
        daily_volatility = self.portfolio_volatility(weights, self.asset_cov_matrix)
    
        if Smart:
            autocorr_penalty = self.portfolio_autocorr_penalty(weights)
            daily_volatility *= autocorr_penalty
    
        annual_volatility = daily_volatility * np.sqrt(trading_days)
    
        # Calcula el exceso de retorno diario del portafolio y luego anualízalo
        daily_excess_return = np.dot(weights, self.average_asset_returns) - self.rf_daily
        annual_excess_return = daily_excess_return * trading_days
    
        # Calcula el ratio de Sharpe anualizado
        annual_sharpe_ratio = annual_excess_return / annual_volatility
    
        return -annual_sharpe_ratio
    
    
    def neg_sharpe_ratio_ff(self, weights, *args):
        if self.ff_expected_returns is None or self.ff_expected_returns.empty:
            return float('inf')
    
        trading_days = 252
    
        daily_excess_return = np.dot(weights, self.ff_expected_returns) - self.rf_daily
        annual_excess_return = daily_excess_return * trading_days
    
        daily_volatility = self.portfolio_volatility(weights, self.asset_cov_matrix)
        annual_volatility = daily_volatility * np.sqrt(trading_days)
    
        annual_sharpe_ratio_ff = annual_excess_return / annual_volatility
    
        return -annual_sharpe_ratio_ff
    

    def neg_omega_ratio(self, weights, Smart=False):
        """
        Calculates the negative Omega Ratio for a given set of asset weights, comparing daily returns to a daily benchmark.
    
        :param weights: Weights of the assets in the portfolio.
        :param Smart: Boolean flag to apply autocorrelation penalty.
        :return: Negative Omega Ratio for optimization.
        """
        portfolio_returns = pd.DataFrame(self.asset_returns.dot(weights))
        excess_returns = portfolio_returns[0] - self.benchmark_returns[self.benchmark_returns.columns[0]] 
        positive_excess = excess_returns[excess_returns > 0].sum()
        negative_excess = -excess_returns[excess_returns < 0].sum()
        omega_ratio = positive_excess / negative_excess
    
       
        if Smart:
            autocorr_penalty = self.portfolio_autocorr_penalty(weights)
            omega_ratio /= (1 + ((autocorr_penalty - 1) * 2))
    
        return -omega_ratio 

       

    def portfolio_var(self, weights, empirical= True):
        """
        Calculates Value at Risk (VaR) of the portfolio using the empirical data if :param empirical: = True. If set to 'False' a variance-covariance approach its taken and normality in the distribution on the returns is assumed.

        :param weights: Weights of the assets in the portfolio.
        :return: Portfolio's Value at Risk.
        """
        confidence_level = 0.05
        
        if empirical == False:
            portfolio_volatility = self.portfolio_volatility(weights, self.asset_cov_matrix)
            # VaR as standard deviation multiplied by th value Z of the normal distribution
            VaR = norm.ppf(1 - confidence_level) * portfolio_volatility
                                                                                                     
        else:
            #get percentile associated to the confidence level of the portfolios historical returns
            portfolio_value = self.asset_prices.dot(weights)
            portfolio_rend = portfolio_value.pct_change().dropna()
            VaR = portfolio_rend.quantile(confidence_level)
        
               
        return VaR 

    def semivariance_ratio(self, weights, *args):
        """
        Returns the semivariance ratio for optimization purposes.
        :param weights: Weights of the assets in the portfolio.
        :return: Semivariance ratio of the portfolio.
        """

        semivariance = np.dot(weights, np.dot(self.asset_cov_matrix, weights))

        return semivariance

    
    def neg_safety_first_ratio(self, weights, *args):
        """
        Calculates the negative Roy's Safety First Ratio (SFratio) for a given set of asset weights.

        :param weights: Weights of the assets in the portfolio.
        :return: Negative SFratio.
        """
        MAR = self.rf_daily  # Ejemplo de retorno mínimo aceptable
        portfolio_return = np.dot(weights, self.average_asset_returns)
        portfolio_volatility = self.portfolio_volatility(weights, self.asset_cov_matrix)
        
        sf_ratio = (portfolio_return - MAR) / portfolio_volatility
        
        return -sf_ratio
    
    def neg_sortino_ratio(self, weights, *args):
        """  
        Calculate the negative Sortino ratio of a portfolio, focusing on downside risk relative to the risk-free rate. The Sortino ratio is 
        a modification of the Sharpe ratio that distinguishes between harmful volatility and total volatility by using the portfolio's 
        downside deviation instead of the total standard deviation of the portfolio returns.

        Parameters:
        - weights: array-like, the weights of the assets in the portfolio.

        Returns:
        - float, the portfolio's negative Sortino ratio for its minimization in optimization routines.

        1. Calculate the portfolio return: Utilize the dot product between the weights of the portfolio's assets and the average returns of 
        those assets to derive the portfolio return.
        2. Filter for returns below the risk-free rate: Identify only the returns of the portfolio's assets that are below the risk-free 
        rate. This step is crucial for calculating semivariance, which focuses on downside volatility.
        3. Calculate weighted negative excess returns: Apply the portfolio's weights to these filtered returns to obtain a weighted 
        representation of them. These represent the negative excess returns, which are the focus of the Sortino ratio calculation.
        4. Calculate the semivariance: Square the weighted negative excess returns and then average these values to determine the 
        semivariance. The semivariance measures the variability of losses relative to the risk-free rate.
        5. Calculate the Sortino ratio: Subtract the risk-free rate from the portfolio return and divide the result by the square root of 
        the semivariance. This yields the Sortino ratio, a measure of risk-adjusted performance that considers only the negative volatility 
        relative to the risk-free rate.

        Note:
        The calculation is designed to focus on the downside risk by considering only the negative excess returns relative to the risk-free 
        rate.
    
        """ 
        
        portfolio_return = np.dot(weights, self.average_asset_returns)
        excess_returns = self.asset_returns - self.rf_daily
        negative_excess_returns = excess_returns[excess_returns < 0]
        weighted_negative_excess_returns = negative_excess_returns.multiply(weights, axis=1)
        semivariance = np.mean(np.square(weighted_negative_excess_returns.sum(axis=1)))

        sortino_ratio = (portfolio_return - self.rf_daily) / np.sqrt(semivariance)
        
        return -sortino_ratio
    
    
    def neg_risk_parity_ratio(self, weights, *args):
        """
        Calculate the negative Risk Parity ratio for a given set of asset weights, optimizing towards an equitable risk distribution among 
        the portfolio assets. Risk Parity is an investment strategy that aims to distribute risk equitably among the components of a 
        portfolio.

        Parameters:

        - weights: array-like, the weights of the assets in the portfolio.
        - Returns: float, the negative Risk Parity ratio of the portfolio for its minimization in optimization routines.
    

        1. Determine the total volatility of the portfolio using the current weights and the covariance matrix of the asset returns.
        2. Evaluate how each additional weight in an asset contributes to the total risk of the portfolio, using the covariance matrix and 
        current weights.
        3. Multiply the marginal risk contribution of each asset by their respective weights to obtain their total contribution to the 
        portfolio's risk.
        4. Sum the squared differences between the risk contribution of each asset and the average of these contributions, aiming for an 
        equitable risk distribution.
        5. Return the negative value of this sum: By minimizing this negative value, optimization focuses on equalizing each asset's risk 
        contribution, aligning with the principle of Risk Parity.
        """
        
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.asset_cov_matrix, weights)))
        marginal_risk_contribution = self.asset_cov_matrix.dot(weights) / portfolio_volatility
        total_risk_contribution = weights * marginal_risk_contribution
        risk_parity_ratio = np.sum((total_risk_contribution - total_risk_contribution.mean())**2)

        return -risk_parity_ratio
    
    def cvar(self, weights, alpha=0.05, Smart=False):
        """
        Calculate the Conditional Value at Risk (CVaR) for a given set of asset weights.
        This metric focuses on the expected losses in the worst alpha percent of cases, useful for risk management and 
        portfolio optimization, particularly in minimizing potential extreme losses.

        Parameters:
        - weights: array-like, the weights of the assets in the portfolio.
        - alpha: float, confidence level for CVaR calculation (e.g., 0.05 for 5%).

        Returns:
        - float, the CVaR of the portfolio for its minimization in optimization routines.

        Steps:
        1. Calculate the portfolio returns using the current weights.
        2. Determine the Value at Risk (VaR) at the specified alpha quantile.
        3. Calculate CVaR as the average of the returns that are below the VaR threshold.
       
        """
        # Step 1: Calculate the portfolio returns
        portfolio_returns = self.asset_returns.dot(weights)

        # Step 2: Calculate VaR at the specified alpha quantile
        VaR = np.percentile(portfolio_returns, alpha * 100)
    
        # Step 3: Calculate CVaR as the mean of the returns below the VaR threshold
        CVaR = portfolio_returns[portfolio_returns <= VaR].mean()
    
        return CVaR
    
    
    def calculate_hrp_weights(self):
        hrp = HierarchicalRiskParity(self.asset_returns)
        hrp_weights = hrp.optimize_hrp()
        return hrp_weights

    def set_blacklitterman_expectations(self, P, Q, tau, Omega):
        """
        Sets the Black-Litterman expectations to be used in optimization.
    
        :param P: Selection matrix indicating the assets involved in the views.
        :param Q: Views vector containing the expected returns.
        :param tau: Scalar indicating the uncertainty in the market equilibrium returns.
        :param Omega: Diagonal matrix representing the uncertainty in the investor's views.
        """
        self.P = P
        self.Q = Q
        self.tau = tau
        self.Omega = Omega    
        self.bl_expectations_set = True
        
    
    def neg_BL_returns(self, weights, *args):
        """
        Black-Litterman objective function for asset allocation optimization.
        
        :param weights: Weights of the assets in the portfolio.
        :param P: Matrix that identifies the assets involved in the views.
        :param Q: Vector of the investor's views on the assets' excess returns.
        :param tau: Scalar indicating the uncertainty in the equilibrium excess returns.
        :param omega: Diagonal matrix of the uncertainty in the investor's views.
        
        :return: Negative of the adjusted portfolio returns based on the Black-Litterman model.
        """
        
        pi = self.tau * np.dot(self.asset_cov_matrix, weights)
        M = np.linalg.inv(np.linalg.inv(self.tau * self.asset_cov_matrix) + np.dot(np.dot(self.P.T, np.linalg.inv(self.Omega)), self.P))
        adjusted_returns = np.dot(M, np.dot(np.linalg.inv(self.tau * self.asset_cov_matrix), pi) + np.dot(np.dot(self.P.T, np.linalg.inv(self.Omega)), self.Q))
        portfolio_return = np.dot(weights, adjusted_returns)
        
        return -portfolio_return    
     

    def optimize_generic(self, optimize_function, objective_function, is_smart=False, **kwargs):
        args = (is_smart,)
        if 'args' in kwargs:
            del kwargs['args']

        if optimize_function.__name__ in ['optimize_SLSQP', 'optimize_Genetic']:
            weights, value = optimize_function(
                objective_function,
                self.start_weights,
                self.bounds,
                self.constraints,  
                args=args,
                **kwargs
            )
        else:
            weights, value = optimize_function(
                objective_function,
                self.start_weights,
                self.bounds,
                args=args,
                **kwargs
            )

        
        value = abs(value)

        return weights, value
    
    
    def Optimize_Portfolio(self, method="MonteCarlo", **kwargs):
        optimization_names = ["Max Sharpe", "Max (Smart) Sharpe", "Max Omega", "Max (Smart) Omega", "Min VaR (Empirical)", "Min VaR (Parametric)", "Semivariance", "Safety-First","Max Sortino","Risk Parity","CVaR", "Max Sharpe FF"]
        
        if self.bl_expectations_set:
            optimization_names.append("Black-Litterman")
        
        optimized_weights = []
        optimized_values = []

        optimize_function = getattr(self, f"optimize_{method}", None)

        if not optimize_function:
            raise ValueError(f"Optimization method '{method}' not recognized.")
        optimizations = [
            (self.neg_sharpe_ratio, False),
            (self.neg_sharpe_ratio, True),  # 'True' = Smart Sharpe
            (self.neg_omega_ratio, False),
            (self.neg_omega_ratio, True),   # 'True' = Smart Omega
            (self.portfolio_var, True),     # Empirical VaR
            (self.portfolio_var, False),    # Parametric VaR
            (self.semivariance_ratio, False),
            (self.neg_safety_first_ratio, False),
            (self.neg_sortino_ratio, False),
            (self.neg_risk_parity_ratio, False),
            (self.cvar, False),
            (self.neg_sharpe_ratio_ff, False)
        ]
        
        
        if self.bl_expectations_set:
            optimizations.append((self.neg_BL_returns, False))
        

        
        
        for i, (objective_function, is_smart) in enumerate(optimizations):
            weights, value = self.optimize_generic(
                optimize_function,
                objective_function,
                is_smart=is_smart,
                **kwargs
            )
            if optimization_names[i] in ["Min VaR (Empirical)", "Min VaR (Parametric)"]:
                value *= -1
            optimized_weights.append(weights)
            optimized_values.append(value)

        results_df = pd.DataFrame(optimized_weights, index=optimization_names, columns=self.asset_prices.columns)
        results_df['Optimized Value'] = optimized_values
        # HRP
        hrp_weights = self.calculate_hrp_weights()
        #results_df.loc['HRP'] = hrp_weights.append(pd.Series({"Optimized Value": np.nan}))
        # Asumiendo que hrp_weights es una pd.Series y deseas agregar un solo ítem:
        hrp_weights['Optimized Value'] = np.nan
        results_df.loc['HRP'] = hrp_weights
        return results_df


    
    
class DynamicBacktester:
    """
    A class to perform dynamic backtesting of portfolio strategies over a specified period using asset prices and
    benchmarks.

    Attributes:
        start_date (datetime): The start date of the backtesting period.
        end_date (datetime): The end date of the backtesting period.
        assets (list): A list of asset identifiers used in the portfolio.
        benchmark (str): The identifier for the benchmark asset.
        rf (float): The risk-free rate to be used in optimizations.
        initial_capital (float, optional): The initial capital amount for the backtest. Default is $1,000,000.
        strategies (list, optional): A list of strategy names to be tested. Default is ['Max Sharpe'].
        ff_factors_expectations (dict, optional): Expected returns for different Fama-French factors.
        method (str, optional): The optimization method to be used. If not specified, the class defaults to internal settings.

    Methods:
        run_backtest(): Executes the backtesting process over the specified date range and for the specified strategies.
        plot_portfolio(): Plots the portfolio values over time for each strategy.
    """

    def __init__(self, start_date, end_date, assets, benchmark, rf, initial_capital=1_000_000, strategies=None, ff_factors_expectations=None, method=None):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.assets = assets
        self.benchmark = benchmark
        self.initial_capital = initial_capital
        self.strategies = strategies if strategies is not None else ['Max Sharpe']
        self.portfolio_values = {strategy: [] for strategy in self.strategies}
        self.daily_values = {strategy: [] for strategy in self.strategies}
        self.data_downloader = DataDownloader()  
        self.ff_factors_expectations = ff_factors_expectations
        self.rf = rf
        self.rf_daily = self.rf / 252
        self.method = method

    def run_backtest(self):
        """
        Executes the backtesting of selected investment strategies by rebalancing at defined intervals and tracking portfolio values.
        
        This method iteratively updates portfolio allocations using the specified optimization method, recalculates portfolio values based on asset returns, and adjusts the capital allocated to each strategy. It continues until the end date of the backtest is reached.
        """
        current_date = self.start_date
        
        capital_per_strategy = {strategy: self.initial_capital for strategy in self.strategies}
    
        while current_date <= self.end_date:
            next_rebalance_date = min(current_date + timedelta(days=6*30), self.end_date)
            
            asset_prices, benchmark_prices, ff_factors = self.data_downloader.download_data(
                start_date=current_date.strftime("%Y-%m-%d"),
                end_date=next_rebalance_date.strftime("%Y-%m-%d"),
                assets=self.assets, benchmark=self.benchmark
            )
            
            daily_returns = asset_prices.pct_change().dropna()
            
            asset_allocation = AssetAllocation(
                asset_prices=asset_prices,
                benchmark_prices=benchmark_prices,
                rf=self.rf,
                ff_factors=ff_factors,
                ff_factors_expectations=self.ff_factors_expectations,
                RMT_filtering=True
            )
            optimized_portfolios = asset_allocation.Optimize_Portfolio(method=self.method)
           
            for strategy in self.strategies:
                weights = optimized_portfolios.loc[strategy, self.assets]
                
                daily_capital = capital_per_strategy[strategy]
                for date in daily_returns.index:
                    daily_return = (daily_returns.loc[date] * weights).sum()
                    daily_capital *= (1 + daily_return)
                    self.daily_values[strategy].append(daily_capital)
    
            capital_per_strategy = {strategy: self.daily_values[strategy][-1] for strategy in self.strategies}
            
            current_date = next_rebalance_date + timedelta(days=1)
    
    def plot_portfolio(self):
        """
        Plots the daily values of the portfolio for each strategy over the course of the backtest.

        This method generates a line plot displaying the growth of the portfolio value over time for each strategy, aiding in visual comparison of strategy performance.
        """
        max_length = max(len(values) for values in self.daily_values.values())
        dates = pd.date_range(start=self.start_date, periods=max_length, freq='D')
        plt.figure(figsize=(14, 8))
        for strategy, values in self.daily_values.items():
            if len(values) < max_length:
                values += [np.nan] * (max_length - len(values))
            plt.plot(dates)
