"""
Numerical methods for solving the Geometric Brownian Motion SDE.

This module provides implementations of:
- Euler-Maruyama method
- Milstein method

For the SDE: dY(t) = μY(t)dt + σY(t)dB(t)
"""

import numpy as np
from typing import Tuple, Optional

from .base_gbm import BaseGBM


class EulerMaruyamaGBM(BaseGBM):
    """
    Euler-Maruyama numerical method for GBM.
    
    Discretization scheme:
    Y_{n+1} = Y_n + μ Y_n Δt + σ Y_n ΔB_n
    
    where ΔB_n ~ N(0, Δt)
    """
    
    def __init__(self, y0: float, mu: float, sigma: float, seed: Optional[int] = None):
        """
        Initialize the Euler-Maruyama simulator.
        
        Args:
            y0: Initial value Y(0)
            mu: Drift coefficient
            sigma: Diffusion coefficient
            seed: Random seed for reproducibility (optional)
        """
        super().__init__(y0, mu, sigma)
        if seed is not None:
            np.random.seed(seed)
    
    def simulate_path(self, T: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a GBM path using the Euler-Maruyama method.
        
        Args:
            T: Terminal time
            N: Number of time steps
            
        Returns:
            Tuple of (time_values, y_values)
        """
        if T <= 0:
            raise ValueError("Terminal time T must be positive")
        if N <= 0:
            raise ValueError("Number of steps N must be positive")
        
        dt = T / N
        t_values = np.linspace(0, T, N + 1)
        y_values = np.zeros(N + 1)
        y_values[0] = self.y0
        
        # Generate Brownian increments
        dB = np.sqrt(dt) * np.random.randn(N)
        
        # Euler-Maruyama iteration
        for i in range(N):
            y_values[i + 1] = y_values[i] + self.mu * y_values[i] * dt + \
                              self.sigma * y_values[i] * dB[i]
            
            # Ensure positivity (optional safeguard)
            if y_values[i + 1] <= 0:
                y_values[i + 1] = 1e-10
        
        return t_values, y_values


class MilsteinGBM(BaseGBM):
    """
    Milstein numerical method for GBM.
    
    Discretization scheme (order 1.0 strong convergence):
    Y_{n+1} = Y_n + μ Y_n Δt + σ Y_n ΔB_n + (1/2) σ² Y_n ((ΔB_n)² - Δt)
    
    The Milstein method includes the second-order term that accounts for
    the derivative of the diffusion coefficient.
    """
    
    def __init__(self, y0: float, mu: float, sigma: float, seed: Optional[int] = None):
        """
        Initialize the Milstein simulator.
        
        Args:
            y0: Initial value Y(0)
            mu: Drift coefficient
            sigma: Diffusion coefficient
            seed: Random seed for reproducibility (optional)
        """
        super().__init__(y0, mu, sigma)
        if seed is not None:
            np.random.seed(seed)
    
    def simulate_path(self, T: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a GBM path using the Milstein method.
        
        Args:
            T: Terminal time
            N: Number of time steps
            
        Returns:
            Tuple of (time_values, y_values)
        """
        if T <= 0:
            raise ValueError("Terminal time T must be positive")
        if N <= 0:
            raise ValueError("Number of steps N must be positive")
        
        dt = T / N
        t_values = np.linspace(0, T, N + 1)
        y_values = np.zeros(N + 1)
        y_values[0] = self.y0
        
        # Generate Brownian increments
        dB = np.sqrt(dt) * np.random.randn(N)
        
        # Milstein iteration
        for i in range(N):
            # Drift term
            drift = self.mu * y_values[i] * dt
            
            # Diffusion term
            diffusion = self.sigma * y_values[i] * dB[i]
            
            # Milstein correction term
            # For GBM: g(Y) = σY, so g'(Y) = σ
            # Correction = (1/2) * g(Y) * g'(Y) * ((ΔB)² - Δt)
            milstein_correction = 0.5 * self.sigma**2 * y_values[i] * (dB[i]**2 - dt)
            
            y_values[i + 1] = y_values[i] + drift + diffusion + milstein_correction
            
            # Ensure positivity (optional safeguard)
            if y_values[i + 1] <= 0:
                y_values[i + 1] = 1e-10
        
        return t_values, y_values


class NumericalGBMComparison:
    """
    Utility class to compare analytical and numerical solutions.
    """
    
    def __init__(self, y0: float, mu: float, sigma: float, seed: Optional[int] = None):
        """
        Initialize comparison with all three methods.
        
        Args:
            y0: Initial value Y(0)
            mu: Drift coefficient
            sigma: Diffusion coefficient
            seed: Random seed for reproducibility
        """
        from .gbm_simulation import GBMSimulator
        
        self.y0 = y0
        self.mu = mu
        self.sigma = sigma
        
        self.analytical = GBMSimulator(y0, mu, sigma, seed=seed)
        self.euler = EulerMaruyamaGBM(y0, mu, sigma, seed=seed)
        self.milstein = MilsteinGBM(y0, mu, sigma, seed=seed)
    
    def compare_methods(self, T: float, N: int) -> dict:
        """
        Compare all three methods on the same Brownian path.
        
        Args:
            T: Terminal time
            N: Number of time steps
            
        Returns:
            Dictionary with results from all methods
        """
        # Set same seed for fair comparison
        np.random.seed(42)
        t_anal, y_anal = self.analytical.simulate_path(T, N)
        
        np.random.seed(42)
        t_euler, y_euler = self.euler.simulate_path(T, N)
        
        np.random.seed(42)
        t_milstein, y_milstein = self.milstein.simulate_path(T, N)
        
        return {
            'time': t_anal,
            'analytical': y_anal,
            'euler_maruyama': y_euler,
            'milstein': y_milstein
        }
    
    def plot_comparison(self, T: float, N: int, output: Optional[str] = None):
        """
        Plot comparison of all three methods.
        
        Args:
            T: Terminal time
            N: Number of time steps
            output: Output file path (optional)
        """
        import matplotlib.pyplot as plt
        
        results = self.compare_methods(T, N)
        
        plt.figure(figsize=(12, 6))
        plt.plot(results['time'], results['analytical'], 
                label='Analytical', linewidth=2, alpha=0.8)
        plt.plot(results['time'], results['euler_maruyama'], 
                label='Euler-Maruyama', linewidth=1.5, alpha=0.8, linestyle='--')
        plt.plot(results['time'], results['milstein'], 
                label='Milstein', linewidth=1.5, alpha=0.8, linestyle='-.')
        
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Y(t)', fontsize=12)
        plt.title(f'GBM Methods Comparison (μ={self.mu}, σ={self.sigma}, N={N})', 
                 fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if output:
            plt.savefig(output, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {output}")
        else:
            plt.show()
        
        plt.close()
    
    def compute_errors(self, T: float, N_values: list, n_samples: int = 100) -> dict:
        """
        Compute strong convergence errors for numerical methods.
        
        Args:
            T: Terminal time
            N_values: List of step counts to test
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with errors for each method and step count
        """
        from .gbm_simulation import GBMSimulator
        
        euler_errors = []
        milstein_errors = []
        
        for N in N_values:
            euler_err = []
            milstein_err = []
            
            for _ in range(n_samples):
                seed = np.random.randint(0, 1000000)
                
                # Analytical solution
                anal = GBMSimulator(self.y0, self.mu, self.sigma, seed=seed)
                _, y_anal = anal.simulate_path(T, N)
                
                # Euler-Maruyama
                euler = EulerMaruyamaGBM(self.y0, self.mu, self.sigma, seed=seed)
                _, y_euler = euler.simulate_path(T, N)
                
                # Milstein
                milstein = MilsteinGBM(self.y0, self.mu, self.sigma, seed=seed)
                _, y_milstein = milstein.simulate_path(T, N)
                
                # Compute terminal errors
                euler_err.append(abs(y_euler[-1] - y_anal[-1]))
                milstein_err.append(abs(y_milstein[-1] - y_anal[-1]))
            
            euler_errors.append(np.mean(euler_err))
            milstein_errors.append(np.mean(milstein_err))
        
        return {
            'N_values': N_values,
            'euler_errors': euler_errors,
            'milstein_errors': milstein_errors
        }