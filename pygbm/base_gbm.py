"""
Base class for Geometric Brownian Motion simulations.

This module provides the abstract base class for GBM-related simulations.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class BaseGBM(ABC):
    """
    Abstract base class for Geometric Brownian Motion simulations.
    
    Attributes:
        y0 (float): Initial value Y(0)
        mu (float): Drift coefficient
        sigma (float): Diffusion coefficient
    """
    
    def __init__(self, y0: float, mu: float, sigma: float):
        """
        Initialize the base GBM model.
        
        Args:
            y0: Initial value Y(0)
            mu: Drift coefficient
            sigma: Diffusion coefficient (must be non-negative)
            
        Raises:
            ValueError: If sigma is negative or y0 is non-positive
        """
        if y0 <= 0:
            raise ValueError("Initial value y0 must be positive")
        if sigma < 0:
            raise ValueError("Diffusion coefficient sigma must be non-negative")
            
        self.y0 = y0
        self.mu = mu
        self.sigma = sigma
    
    @abstractmethod
    def simulate_path(self, T: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a path of the stochastic process.
        
        Args:
            T: Terminal time
            N: Number of time steps
            
        Returns:
            Tuple of (time_values, process_values)
        """
        pass
    
    def _generate_brownian_motion(self, T: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a Brownian motion path using incremental simulation.
        
        Args:
            T: Terminal time
            N: Number of time steps
            
        Returns:
            Tuple of (time_values, brownian_values)
        """
        if T <= 0:
            raise ValueError("Terminal time T must be positive")
        if N <= 0:
            raise ValueError("Number of steps N must be positive")
            
        dt = T / N
        t_values = np.linspace(0, T, N + 1)
        
        # Generate increments: dB = sqrt(dt) * Z, where Z ~ N(0,1)
        dB = np.sqrt(dt) * np.random.randn(N)
        
        # Brownian motion is the cumulative sum of increments, starting at 0
        B_values = np.concatenate(([0], np.cumsum(dB)))
        
        return t_values, B_values
    
    def __repr__(self) -> str:
        """String representation of the GBM model."""
        return f"{self.__class__.__name__}(y0={self.y0}, mu={self.mu}, sigma={self.sigma})"