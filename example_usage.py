"""
Example usage of the refactored D-TIPO experiment framework
"""

import numpy as np
import tensorflow as tf

# Force TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

# Set TensorFlow to use float32 as default dtype
tf.keras.backend.set_floatx('float32')
from experiment_training_improved import (
    ExperimentConfig, ModelConfig, NetworkConfig, 
    TrainingConfig, TradingConfig, experiment_training,
    plot_experiment_results
)


def example_1_basic_mean_variance():
    """
    Example 1: Basic mean-variance optimization without options
    """
    print("=" * 60)
    print("Example 1: Basic Mean-Variance Optimization")
    print("=" * 60)
    
    # Create configuration
    config = ExperimentConfig(
        model=ModelConfig(
            model_name='GBM_MV',
            T=2.0,
            r=0.06,
            ExT=1.2,
            jumps=False
        ),
        network=NetworkConfig(
            regularization=0.0,
            neurons=[20, 20],
            activation='relu'
        ),
        training=TrainingConfig(
            num_epochs=12,
            learning_rate=0.01
        ),
        trading=TradingConfig(
            curtage=0.000,
            leverage_constraints=False,
            options_in_p=False,
            bankrupcy_constraint=False
        )
    )
    
    # Run experiment
    results = experiment_training(config, save_dir="./results/basic_mv")
    
    # Plot results
    plot_experiment_results(results, save_path="./results/basic_mv/summary.png")
    
    return results


def example_2_with_jumps():
    """
    Example 2: Portfolio optimization with jump processes
    """
    print("\n" + "=" * 60)
    print("Example 2: Portfolio Optimization with Jumps")
    print("=" * 60)
    
    config = ExperimentConfig(
        model=ModelConfig(
            model_name='GBM_MV',
            T=2.0,
            r=0.06,
            ExT=1.2,
            jumps=True,  # Enable jumps
            lamb=0.05,   # Jump intensity
            mu=0.0,      # Jump mean
            s=0.2        # Jump std dev
        ),
        network=NetworkConfig(
            regularization=0.0,
            neurons=[20, 20],
            activation='relu'
        ),
        training=TrainingConfig(
            num_epochs=10,  # More epochs for complex dynamics
            learning_rate=0.01
        ),
        trading=TradingConfig(
            curtage=0.005,
            leverage_constraints=False,
            options_in_p=False,
            bankrupcy_constraint=True  # Enable bankruptcy constraint
        )
    )
    
    results = experiment_training(config, save_dir="./results/with_jumps")
    plot_experiment_results(results, save_path="./results/with_jumps/summary.png")
    
    return results


def example_3_with_options():
    """
    Example 3: D-TIPO with options (full implementation)
    """
    print("\n" + "=" * 60)
    print("Example 3: D-TIPO with Options")
    print("=" * 60)
    
    config = ExperimentConfig(
        model=ModelConfig(
            model_name='GBM_MV',
            T=2.0,
            r=0.06,
            ExT=1.2,
            jumps=True,
            lamb=0.05,
            mu=0.0,
            s=0.2
        ),
        network=NetworkConfig(
            regularization=1e-6,  # Add regularization
            neurons=[20, 20, 20],  # Deeper network
            activation='relu'
        ),
        training=TrainingConfig(
            num_epochs=15,  # More training for options
            learning_rate=0.01
        ),
        trading=TradingConfig(
            curtage=0.005,
            leverage_constraints=True,   # Enable leverage constraints
            options_in_p=True,          # Enable options trading
            bankrupcy_constraint=True
        )
    )
    
    results = experiment_training(config, save_dir="./results/with_options")
    plot_experiment_results(results, save_path="./results/with_options/summary.png")
    
    return results


def example_4_custom_parameters():
    """
    Example 4: Custom drift and volatility parameters
    """
    print("\n" + "=" * 60)
    print("Example 4: Custom Market Parameters")
    print("=" * 60)
    
    # Custom drift vector (declining returns)
    custom_b = tf.reshape(tf.linspace(0.10, 0.02, 5), [5, 1])
    
    # Custom volatility matrix with correlations
    custom_sigma = tf.constant([
        [0.20, 0.05, 0.02, 0.01, 0.01],
        [0.05, 0.18, 0.04, 0.02, 0.01],
        [0.02, 0.04, 0.16, 0.03, 0.02],
        [0.01, 0.02, 0.03, 0.14, 0.04],
        [0.01, 0.01, 0.02, 0.04, 0.12]
    ])
    
    config = ExperimentConfig(
        model=ModelConfig(
            model_name='GBM_MV',
            T=3.0,  # Longer time horizon
            r=0.04,  # Lower risk-free rate
            ExT=1.15,  # Lower expected terminal wealth
            b=custom_b,
            sigma=custom_sigma,
            jumps=False
        ),
        network=NetworkConfig(
            regularization=0.0,
            neurons=[30, 30],  # Larger network
            activation='relu'
        ),
        training=TrainingConfig(
            num_epochs=12,
            learning_rate=0.005  # Lower learning rate
        ),
        trading=TradingConfig(
            curtage=0.002,  # Lower transaction costs
            leverage_constraints=False,
            options_in_p=False,
            bankrupcy_constraint=False
        )
    )
    
    results = experiment_training(config, save_dir="./results/custom_params")
    plot_experiment_results(results, save_path="./results/custom_params/summary.png")
    
    return results


def compare_experiments(results_list, labels):
    """
    Compare multiple experiment results
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    for results, label in zip(results_list, labels):
        train_loss = results['train_loss']
        plt.plot(train_loss, '.-', linewidth=2, label=label)
    
    # Add theoretical optimal line from first experiment
    optimal_loss = results_list[0]['theoretical_values']['optimal_loss']
    plt.axhline(y=optimal_loss, color='k', linestyle='--', 
                label=f'Theoretical Optimal: {optimal_loss:.4f}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Comparison of Different Configurations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./results/comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Run all examples and compare results
    """
    # Create results directory
    import os
    os.makedirs('./results', exist_ok=True)
    
    # Run examples
    results_basic = example_1_basic_mean_variance()
    results_jumps = example_2_with_jumps()
    # results_options = example_3_with_options()  # Uncomment if FullNetwork supports options
    results_custom = example_4_custom_parameters()
    
    # Compare experiments
    compare_experiments(
        [results_basic, results_jumps, results_custom],
        ['Basic MV', 'With Jumps', 'Custom Parameters']
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary of All Experiments")
    print("=" * 60)
    
    experiments = [
        ('Basic MV', results_basic),
        ('With Jumps', results_jumps),
        ('Custom Parameters', results_custom)
    ]
    
    for name, results in experiments:
        final_loss = results['train_loss'][-1]
        optimal_loss = results['theoretical_values']['optimal_loss']
        gap = abs(final_loss - optimal_loss)
        
        print(f"\n{name}:")
        print(f"  Final Loss: {final_loss:.6f}")
        print(f"  Optimal Loss: {optimal_loss:.6f}")
        print(f"  Gap: {gap:.6f} ({gap/abs(optimal_loss)*100:.2f}%)")
        print(f"  Lambda: {results['theoretical_values']['lam']:.6f}")
        print(f"  Training epochs: {len(results['train_loss'])}")


if __name__ == "__main__":
    main()