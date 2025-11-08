"""
Command-line interface for Geometric Brownian Motion simulation.

This module provides a CLI for simulating and plotting GBM paths.
"""

import argparse
from .gbm_simulation  import GBMSimulator
from .numerical_gbm import EulerMaruyamaGBM, MilsteinGBM, NumericalGBMComparison


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Simulate Geometric Brownian Motion")
    parser.add_argument("--y0", type=float, required=True, help="Initial value Y(0)")
    parser.add_argument("--mu", type=float, required=True, help="Drift coefficient")
    parser.add_argument("--sigma", type=float, required=True, help="Diffusion coefficient")
    parser.add_argument("--T", type=float, required=True, help="Total time for simulation")
    parser.add_argument("--N", type=int, required=True, help="Number of time steps")
    parser.add_argument("--output", type=str, help="Output file for the plot")
    parser.add_argument("--method", type=str, default="analytical",
                       choices=["analytical", "euler", "milstein", "compare"],
                       help="Simulation method: analytical, euler, milstein, or compare all")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Select method and simulate
    if args.method == "analytical":
        simulator = GBMSimulator(args.y0, args.mu, args.sigma, seed=args.seed)
        t_values, y_values = simulator.simulate_path(args.T, args.N)
        simulator.plot_path(t_values, y_values, output=args.output)
        
    elif args.method == "euler":
        simulator = EulerMaruyamaGBM(args.y0, args.mu, args.sigma, seed=args.seed)
        t_values, y_values = simulator.simulate_path(args.T, args.N)
        # Use GBMSimulator for plotting
        plotter = GBMSimulator(args.y0, args.mu, args.sigma)
        plotter.plot_path(t_values, y_values, output=args.output, 
                         title="GBM Path (Euler-Maruyama)")
        
    elif args.method == "milstein":
        simulator = MilsteinGBM(args.y0, args.mu, args.sigma, seed=args.seed)
        t_values, y_values = simulator.simulate_path(args.T, args.N)
        # Use GBMSimulator for plotting
        plotter = GBMSimulator(args.y0, args.mu, args.sigma)
        plotter.plot_path(t_values, y_values, output=args.output,
                         title="GBM Path (Milstein)")
        
    elif args.method == "compare":
        comparator = NumericalGBMComparison(args.y0, args.mu, args.sigma, seed=args.seed)
        comparator.plot_comparison(args.T, args.N, output=args.output)


if __name__ == "__main__":
    main()