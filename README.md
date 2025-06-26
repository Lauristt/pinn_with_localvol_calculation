# pinn_with_localvol_calculation
Local Volatility PINN Training

This codebase implements a Physics-Informed Neural Network (PINN) for pricing options and inferring a local volatility surface. It leverages PyTorch for neural network training and uses a Crank-Nicolson finite difference method (FDM) solver to incorporate the Black-Scholes-Merton (BSM) Partial Differential Equation (PDE) as a physical constraint during training.
Project Structure

    Train_Mar29_L2Loss-LocalVol.py: The main script for data preprocessing, model training, and evaluation.

    src/Solver.py: Contains the core components for option pricing, volatility surface modeling, and the FDM solver.

Train_Mar29_L2Loss-LocalVol.py - Main Training Script

This script orchestrates the entire training pipeline, from data loading and preprocessing to model training, evaluation, and result saving.
Key Functions and Components:

    bsm_price(S, K, T, r, sigma, option_type='call'): Calculates the Black-Scholes-Merton option price.

    preprocess(dataset_path, underlying_asset_data, save_path):

        Handles data loading, cleaning, and feature engineering.

        Filters out invalid implied volatility data.

        Calculates time to expiration and standardizes strike prices.

        Merges option data with underlying asset prices.

        Includes a robust caching mechanism with validation to avoid redundant preprocessing.

    prepare_data(df, ref_df, output_path):

        Prepares the preprocessed DataFrame into PyTorch tensors (X_real, u_real, X_exp, u_exp).

        X_real: Input features for the PINN (Time to expiration, forward price, strike price, implied volatility).

        u_real: Actual option prices (linear approximation of bid/ask).

        X_exp, u_exp: Data for boundary conditions (options at expiry or priced by BSM for far-from-expiry).

        Uses parallel processing (ThreadPoolExecutor) for efficiency.

        Also includes caching for generated tensors.

    LocalVolatilityPINN(nn.Module):

        The core neural network model.

        Takes DupireVolatilityModel as an input, which provides local volatility.

        Uses a ResidualBlock for deeper learning.

        Predicts option prices based on input features (time, underlying price, strike, and the volatility from the Dupire model).

    train_step(model, X_real_batch, u_real_batch, X_exp_batch, u_exp_batch, r, lambda_params):

        Performs a single training step (forward pass, loss calculation, backward pass).

        Calculates three types of losses:

            Data Loss: MSE between PINN predictions and actual market prices.

            BSM Loss: MSE between PINN predictions and prices from the Crank-Nicolson solver (which uses the local volatility). This enforces the PDE constraint.

            Expiry Loss: MSE for boundary conditions (PINN predictions at expiry matching intrinsic values).

        Implements a dynamic weighting scheme for the three loss components (lambda_params) to balance their contributions during training, ensuring numerical stability.

    save_loss_records(loss_records, save_dir): Saves training loss history (batch-wise and epoch-wise) to CSV files and generates plots visualizing loss trends.

    main():

        Sets up the device (CUDA or CPU).

        Defines data paths and output directories.

        Calls preprocess_data and prepare_data to get the dataset.

        Splits data into training and testing sets.

        Initializes the DupireVolatilityModel and CrankNicolsonSolver.

        Initializes the LocalVolatilityPINN model, optimizer (AdamW), and learning rate scheduler (OneCycleLR).

        Implements a training loop with mixed-precision training (GradScaler), early stopping, and periodic checkpointing.

        Performs validation steps, reporting MAE, RMSE, and relative error.

        Periodically visualizes training progress (loss curves, weight distribution).

        Saves the best model and final predictions.

src/Solver.py - Core Solver Components

This file defines the mathematical models and numerical methods used in the project.
Key Classes:

    VolatilityPrecomputer:

        (Not directly used in Train_Mar29_L2Loss-LocalVol.py but provides functionality for precomputing a volatility surface using an FDM solver).

        precompute(): Iteratively solves for volatility at various strike/expiry points.

        save(), load(): For persisting and loading precomputed surfaces.

        get_sigma(): Interpolates volatility from the precomputed grid.

    RBFLayer(nn.Module):

        A Radial Basis Function (RBF) interpolation layer.

        Used by DupireVolatilityModel to interpolate implied volatilities from sparse market data to create a smooth surface for initializing the local volatility.

        Designed for memory efficiency with batch processing.

    DupireVolatilityModel(nn.Module):

        Models the local volatility surface based on the Dupire equation, derived from market implied volatilities.

        _init_sigma_grid(): Initializes the local volatility grid by performing RBF interpolation on provided option data and then using a preliminary Dupire calculation.

        _compute_dupire_vol(iv, K, T): The core method to compute the local volatility using automatic differentiation to calculate the necessary derivatives (first and second order with respect to strike, and first order with respect to time to expiry) from a given implied volatility surface. This is where the Dupire equation is enforced.

        plot_vol_surface(): Visualizes the 3D local volatility surface.

        get_sigma(K, T): Performs bilinear interpolation on the sigma_grid (the local volatility surface) to retrieve volatility for arbitrary strike (K) and time (T) inputs.

        save_vol_surface(): Saves the local volatility surface data.

    CrankNicolsonSolver(nn.Module):

        Implements the Crank-Nicolson finite difference method to solve the Black-Scholes PDE for option prices.

        solve(S0, K, T, M, N): Solves for a single option price given initial stock price, strike, and time to maturity. It uses the local volatility obtained from the DupireVolatilityModel.

        batch_solve(S_tensor, K_tensor, T_tensor): A vectorized version for efficiently pricing multiple options.

        batch_tdma_solve(): An optimized, batched implementation of the Thomas algorithm for solving tridiagonal linear systems, which arise in the Crank-Nicolson method.

Usage

To run the training process:

    Ensure you have the required processed_option_data.csv and 行情序列.xlsx files in your working directory.

    Run the Train_Mar29_L2Loss-LocalVol.py script:

    python Train_Mar29_L2Loss-LocalVol.py

The script will perform data preprocessing, train the PINN model, and save various outputs including:

    Preprocessed data cache.

    Tensor data cache.

    Training loss plots (.png).

    CSV files of loss records.

    Saved model checkpoints (.pth).

    A plot of the inferred local volatility surface (vol_surface.png).

    Final predictions (.npy).

Dependencies

The project relies on the following Python libraries:

    numpy

    pandas

    torch

    tqdm

    matplotlib

    scipy

    datetime

    logging

Ensure these are installed in your environment (e.g., using pip install -r requirements.txt if a requirements.txt file is provided, otherwise install them manually).
