"""
Geometric Brownian Motion simulator implementation.

This module provides the GBMSimulator class for simulating GBM paths
and a command-line interface for generating plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, Optional
from pathlib import Path

from .base_gbm import BaseGBM


class GBMSimulator(BaseGBM):
    """
    Simulator for Geometric Brownian Motion.
    
    Implements the analytical solution:
    Y(t) = Y_0 * exp((mu - sigma^2/2) * t + sigma * B(t))
    
    where B(t) is a Brownian motion.
    """
    
    def __init__(self, y0: float, mu: float, sigma: float, seed: Optional[int] = None):
        """
        Initialize the GBM simulator.
        
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
        Simulate a path of the Geometric Brownian Motion.
        
        Uses the analytical solution:
        Y(t) = Y_0 * exp((mu - sigma^2/2) * t + sigma * B(t))
        
        Args:
            T: Terminal time
            N: Number of time steps
            
        Returns:
            Tuple of (time_values, y_values) where:
                - time_values: array of time points from 0 to T
                - y_values: array of simulated Y(t) values
        """
        # Generate Brownian motion
        t_values, B_values = self._generate_brownian_motion(T, N)
        
        # Apply GBM formula
        drift_term = (self.mu - 0.5 * self.sigma**2) * t_values
        diffusion_term = self.sigma * B_values
        y_values = self.y0 * np.exp(drift_term + diffusion_term)
        
        return t_values, y_values
    
    def simulate_multiple_paths(self, T: float, N: int, n_paths: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate multiple independent paths of the GBM.
        
        Args:
            T: Terminal time
            N: Number of time steps
            n_paths: Number of independent paths to simulate
            
        Returns:
            Tuple of (time_values, y_values) where:
                - time_values: array of time points (shared across paths)
                - y_values: 2D array of shape (n_paths, N+1) with simulated paths
        """
        if n_paths <= 0:
            raise ValueError("Number of paths must be positive")
        
        t_values = np.linspace(0, T, N + 1)
        y_values = np.zeros((n_paths, N + 1))
        
        for i in range(n_paths):
            _, y_values[i, :] = self.simulate_path(T, N)
        
        return t_values, y_values
    
    def plot_path(self, t_values: np.ndarray, y_values: np.ndarray, 
                  output: Optional[str] = None, 
                  title: str = "Simulated Geometric Brownian Motion Path") -> None:
        """
        Plot a GBM path from given time and value arrays.
        
        Args:
            t_values: Array of time points
            y_values: Array of Y(t) values
            output: Path to save the plot (if None, displays interactively)
            title: Title for the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(t_values, y_values, label=f"GBM Path (μ={self.mu}, σ={self.sigma})", linewidth=2)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Y(t)", fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if output:
            plt.savefig(output, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_multiple_paths(self, T: float, N: int, n_paths: int, 
                           output_path: Optional[str] = None,
                           title: str = "Multiple GBM Paths") -> None:
        """
        Simulate and plot multiple GBM paths.
        
        Args:
            T: Terminal time
            N: Number of time steps
            n_paths: Number of paths to simulate
            output_path: Path to save the plot (if None, displays interactively)
            title: Title for the plot
        """
        t_values, y_values = self.simulate_multiple_paths(T, N, n_paths)
        
        plt.figure(figsize=(10, 6))
        for i in range(n_paths):
            plt.plot(t_values, y_values[i, :], alpha=0.6, linewidth=1)
        
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Y(t)", fontsize=12)
        plt.title(f"{title} (μ={self.mu}, σ={self.sigma}, n={n_paths})", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Command-line interface for GBM simulation."""
    parser = argparse.ArgumentParser(
        description="Simulate Geometric Brownian Motion paths",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--y0", type=float, default=1.0,
                       help="Initial value Y(0)")
    parser.add_argument("--mu", type=float, default=0.05,
                       help="Drift coefficient")
    parser.add_argument("--sigma", type=float, default=0.2,
                       help="Diffusion coefficient")
    parser.add_argument("--T", type=float, default=1.0,
                       help="Terminal time")
    parser.add_argument("--N", type=int, default=100,
                       help="Number of time steps")
    parser.add_argument("--n-paths", type=int, default=1,
                       help="Number of paths to simulate")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path for the plot (e.g., gbm_plot.png)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--title", type=str, default=None,
                       help="Custom title for the plot")
    
    args = parser.parse_args()
    
    # Initialize simulator
    simulator = GBMSimulator(args.y0, args.mu, args.sigma, seed=args.seed)
    
    # Determine title
    if args.title is None:
        if args.n_paths == 1:
            title = "Simulated Geometric Brownian Motion Path"
        else:
            title = "Multiple GBM Paths"
    else:
        title = args.title
    
    # Simulate and plot
    try:
        if args.n_paths == 1:
            simulator.plot_path(args.T, args.N, output_path=args.output, title=title)
        else:
            simulator.plot_multiple_paths(args.T, args.N, args.n_paths, 
                                         output_path=args.output, title=title)
    except Exception as e:
        print(f"Error during simulation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())