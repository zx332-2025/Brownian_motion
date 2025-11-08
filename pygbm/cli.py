"""
Command-line interface for Geometric Brownian Motion simulation.

This module provides a CLI for simulating and plotting GBM paths.
"""

import argparse
from .gbm_simulation import GBMSimulator


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Simulate Geometric Brownian Motion")
    parser.add_argument("--y0", type=float, required=True, help="Initial value Y(0)")
    parser.add_argument("--mu", type=float, required=True, help="Drift coefficient")
    parser.add_argument("--sigma", type=float, required=True, help="Diffusion coefficient")
    parser.add_argument("--T", type=float, required=True, help="Total time for simulation")
    parser.add_argument("--N", type=int, required=True, help="Number of time steps")
    parser.add_argument("--output", type=str, help="Output file for the plot")
    
    args = parser.parse_args()
    
    # Initialize simulator
    simulator = GBMSimulator(args.y0, args.mu, args.sigma)
    
    # Simulate path
    t_values, y_values = simulator.simulate_path(args.T, args.N)
    
    # Plot the path
    simulator.plot_path(t_values, y_values, output=args.output)


if __name__ == "__main__":
    main()