"""Monte Carlo Option Pricing Engine

Implements Black-Scholes-Merton Monte Carlo simulation for European options.
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class OptionParams:
    """Parameters for option pricing."""
    S0: float  # Initial stock price
    K: float   # Strike price
    T: float   # Time to maturity (years)
    r: float   # Risk-free rate
    sigma: float  # Volatility
    option_type: Literal['call', 'put'] = 'call'


class MonteCarloEngine:
    """Monte Carlo simulation engine for option pricing."""
    
    def __init__(self, n_simulations: int = 10000, seed: int = 42):
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)
    
    def simulate_paths(self, params: OptionParams, n_steps: int = 252) -> np.ndarray:
        """Simulate stock price paths using GBM.
        
        Args:
            params: Option parameters
            n_steps: Number of time steps
            
        Returns:
            Array of shape (n_simulations, n_steps + 1) with price paths
        """
        dt = params.T / n_steps
        
        # Generate random normal samples
        Z = self.rng.standard_normal((self.n_simulations, n_steps))
        
        # Initialize paths array
        paths = np.zeros((self.n_simulations, n_steps + 1))
        paths[:, 0] = params.S0
        
        # Simulate paths using GBM
        for t in range(1, n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (params.r - 0.5 * params.sigma**2) * dt + 
                params.sigma * np.sqrt(dt) * Z[:, t-1]
            )
        
        return paths
    
    def price_option(self, params: OptionParams) -> dict:
        """Price European option using Monte Carlo.
        
        Returns:
            Dictionary with price, std_error, and confidence interval
        """
        # Simulate final prices
        paths = self.simulate_paths(params)
        S_T = paths[:, -1]
        
        # Calculate payoffs
        if params.option_type == 'call':
            payoffs = np.maximum(S_T - params.K, 0)
        else:
            payoffs = np.maximum(params.K - S_T, 0)
        
        # Discount to present value
        option_price = np.exp(-params.r * params.T) * np.mean(payoffs)
        
        # Calculate standard error
        std_error = np.std(payoffs) / np.sqrt(self.n_simulations)
        discounted_std_error = np.exp(-params.r * params.T) * std_error
        
        # 95% confidence interval
        ci_95 = 1.96 * discounted_std_error
        
        return {
            'price': option_price,
            'std_error': discounted_std_error,
            'ci_lower': option_price - ci_95,
            'ci_upper': option_price + ci_95,
            'paths': paths
        }


if __name__ == '__main__':
    # Example usage
    params = OptionParams(
        S0=100.0,
        K=105.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type='call'
    )
    
    engine = MonteCarloEngine(n_simulations=10000)
    result = engine.price_option(params)
    
    print(f"Option Price: ${result['price']:.4f}")
    print(f"Standard Error: ${result['std_error']:.4f}")
    print(f"95% CI: [${result['ci_lower']:.4f}, ${result['ci_upper']:.4f}]")
