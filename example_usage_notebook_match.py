#!/usr/bin/env python3
"""
Example usage of the improved D-TIPO training framework
This version matches the parameters from the notebook to achieve better performance
"""

from experiment_training_improved import (
    ExperimentConfig, ModelConfig, NetworkConfig, 
    TrainingConfig, TradingConfig, experiment_training,
    plot_experiment_results
)
import tensorflow as tf
import numpy as np

# Match the notebook's learning rate scheduler
def notebook_scheduler(epoch, lr):
    """Learning rate scheduler from the notebook"""
    if epoch < 2:
        return float(lr)
    else:
        # Exponential decay with factor exp(-0.5 * epoch)
        return float(lr * tf.exp(-0.5 * epoch))

def main():
    print("="*60)
    print("Example: Mean-Variance Optimization (Notebook Parameters)")
    print("="*60)
    
    # Basic Mean-Variance configuration matching the notebook
    config = ExperimentConfig(
        model=ModelConfig(
            model_name='GBM_MV',
            T=2.0,
            r=0.06,
            ExT=1.2,
            jumps=False
        ),
        network=NetworkConfig(
            regularization=0.0,  # Notebook uses 0
            neurons=[20, 20],
            activation='relu'
        ),
        training=TrainingConfig(
            num_epochs=12,  # Match notebook
            learning_rate=0.01
        ),
        trading=TradingConfig(
            curtage=0.0,  # No transaction costs in basic example
            leverage_constraints=False,
            options_in_p=False,
            bankrupcy_constraint=False
        )
    )
    
    # Custom training function to match notebook exactly
    from model_classes_refactored import ModelClass
    from experiment_training_improved import ImprovedFullNetwork
    import json
    from pathlib import Path
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create save directory
    save_dir = "results/notebook_match"
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config.save(save_path / 'config.json')
    print(f"Configuration saved to {save_path / 'config.json'}")
    
    # Initialize model parameters (matching notebook exactly)
    M = 2**22
    d = 5
    N = 20
    batch_size = 2**16
    
    # Set drift and volatility as in notebook
    b = tf.reshape(tf.linspace(0.08, 0.04, d), [d, 1])
    B = tf.transpose(b) - config.model.r
    
    # Volatility matrix from notebook
    sigma = (tf.linalg.diag(tf.linspace(0.18, 0.12, d)) + 
             0.05 * tf.ones([d, d]) - 
             tf.constant([[0., 0., 0.1, 0., 0.], 
                         [0., 0., 0., 0., 0.],
                         [.1, 0., 0., .0, .0], 
                         [0., 0., .0, 0., 0.],
                         [0., 0., 0., 0., 0.]]))
    
    config.model.b = b
    config.model.sigma = sigma
    
    # Calculate theoretical values
    sig_inv = tf.linalg.inv(tf.matmul(sigma, tf.transpose(sigma)))
    rho = tf.matmul(tf.matmul(B, sig_inv), tf.transpose(B))[0, 0].numpy()
    gamma = 1/(1 - np.exp(-rho*config.model.T))*(config.model.ExT - 
                                                  np.exp((config.model.r - rho)*config.model.T))
    varxT = np.exp(-rho*config.model.T)/(1 - np.exp(-rho*config.model.T))*(
        config.model.ExT - np.exp(config.model.r*config.model.T))**2
    lam = 1/(2*(gamma - config.model.ExT))
    
    print(f"Theoretical values:")
    print(f"  ExT: {config.model.ExT:.4f}")
    print(f"  VarxT: {varxT:.6f}")
    print(f"  Lambda: {lam:.6f}")
    print(f"  Optimal loss: {-config.model.ExT + lam*varxT:.6f}")
    
    # Create initial data
    x0 = np.random.uniform(low=1., high=1., size=[M, 1])
    t = np.linspace(0, config.model.T, N + 1)
    h = t[1] - t[0]
    
    # Model parameters
    model_parameters = [
        config.model.model_name, B, sigma, config.model.r, 
        x0, h, config.model.T
    ]
    
    # Create model instances
    model = ModelClass(model_parameters)
    model_2_parameters = model_parameters.copy()
    model_2_parameters[1] = B - B  # Zero drift
    model_2 = ModelClass(model_2_parameters)
    
    # Create network
    full_model = ImprovedFullNetwork(config, model, model_2, h)
    
    # Create optimizer with custom scheduler matching notebook
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)
    
    def dummy_loss(y_target, y_pred):
        return y_pred
    
    loss_list = [dummy_loss] * 7
    loss_weights = [1., float(lam), 0., 0., 0., 0., 0.]
    
    full_model.compile(optimizer=optimizer, loss=loss_list, loss_weights=loss_weights)
    
    # Use the notebook's learning rate scheduler
    lr_callback = tf.keras.callbacks.LearningRateScheduler(notebook_scheduler, verbose=1)
    
    # Train the model
    print(f"\nStarting training for {config.training.num_epochs} epochs...")
    zero_vec = np.zeros(x0.shape)
    
    history = full_model.fit(
        x0, 
        [zero_vec] * 7,
        epochs=config.training.num_epochs,
        batch_size=batch_size,
        callbacks=[lr_callback],
        verbose=1
    )
    
    # Save training history
    history_path = save_path / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    print(f"\nTraining completed!")
    print(f"Final loss: {history.history['loss'][-1]:.6f}")
    print(f"Theoretical optimal loss: {-config.model.ExT + lam*varxT:.6f}")
    
    # Create results dict for plotting
    results = {
        'model': full_model,
        'history': history,
        'theoretical_values': {
            'B': B.numpy(),
            'sig_inv': sig_inv.numpy(),
            'rho': rho,
            'gamma': gamma,
            'varxT': varxT,
            'lam': lam,
            'optimal_loss': -config.model.ExT + lam*varxT
        },
        'model_class': model,
        'model_2_class': model_2,
        'config': config,
        'train_loss': history.history['loss'],
        'h': h,
        'x0': x0
    }
    
    # Plot results
    plot_experiment_results(results, save_path=save_path / "summary.png")

if __name__ == "__main__":
    main()