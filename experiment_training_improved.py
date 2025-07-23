import numpy as np
import tensorflow as tf
from tensorflow.python import tf2
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path

# Force TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

tf.keras.backend.set_floatx('float32')

# Global constants
M = 2**22
d = 5
N = 20
batch_size = 2**16




@dataclass
class ModelConfig:
    """Configuration for the market model"""
    model_name: str = 'GBM_MV'
    T: float = 2.0  # Terminal time
    # d is defined globally as 5
    r: float = 0.06  # Risk-free rate
    ExT: float = 1.2  # Expected terminal wealth
    b: Optional[np.ndarray] = None  # Drift coefficients
    sigma: Optional[np.ndarray] = None  # Volatility matrix
    # Jump parameters
    lamb: float = 0.05
    mu: float = 0.0
    s: float = 0.2
    jumps: bool = False
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        data = self.__dict__.copy()
        # Convert numpy arrays to lists for JSON serialization
        if self.b is not None:
            data['b'] = self.b.tolist() if isinstance(self.b, np.ndarray) else self.b
        if self.sigma is not None:
            data['sigma'] = self.sigma.tolist() if isinstance(self.sigma, np.ndarray) else self.sigma
        return data
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        # Convert lists back to numpy arrays
        if 'b' in data and data['b'] is not None:
            data['b'] = np.array(data['b'])
        if 'sigma' in data and data['sigma'] is not None:
            data['sigma'] = np.array(data['sigma'])
        return cls(**data)


@dataclass
class NetworkConfig:
    """Configuration for neural networks"""
    regularization: float = 0.0
    neurons: List[int] = field(default_factory=lambda: [20, 20])
    activation: str = 'relu'


@dataclass
class TrainingConfig:
    """Configuration for training"""
    num_epochs: int = 12
    learning_rate: float = 0.01
    # M, N, and batch_size are defined globally
    

@dataclass
class TradingConfig:
    """Configuration for trading constraints"""
    curtage: float = 0.005  # Transaction costs
    leverage_constraints: bool = False
    options_in_p: bool = False
    bankrupcy_constraint: bool = False


@dataclass
class ExperimentConfig:
    """Complete configuration for an experiment"""
    model: ModelConfig = field(default_factory=ModelConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        data = {
            'model': self.model.to_dict(),
            'network': self.network.__dict__,
            'training': self.training.__dict__,
            'trading': self.trading.__dict__
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(
            model=ModelConfig.from_dict(data['model']),
            network=NetworkConfig(**data['network']),
            training=TrainingConfig(**data['training']),
            trading=TradingConfig(**data['trading'])
        )


class ImprovedFullNetwork(tf.keras.Model):
    """
    Improved FullNetwork that doesn't rely on global variables
    """
    def __init__(self, config: ExperimentConfig, model_class, model_2_class, h: float):
        super(ImprovedFullNetwork, self).__init__()
        
        # Import the refactored classes
        from model_classes_refactored import SubNetwork
        
        self.config = config
        self.model = model_class
        self.model_2 = model_2_class
        self.h = h
        
        # Network parameters
        self.NN = N
        self.diff = int(self.NN / N)
        
        # Initialize networks
        self.network_u = []
        self.network_u.append(
            SubNetwork(d=d, regularization=None, neurons=[], activation='linear')
        )
        
        for _ in range(N):
            self.network_u.append(
                SubNetwork(d=d, 
                         regularization=config.network.regularization,
                         neurons=config.network.neurons,
                         activation=config.network.activation)
            )
        
        # Networks for options
        self.network_options = SubNetwork(d=2*d + 1, regularization=None,
                                        neurons=[], activation='linear')
        self.network_K = SubNetwork(d=2*d, regularization=None,
                                  neurons=[], activation='linear')

    def call(self, x_in, training=None):
        """Forward pass with all parameters passed explicitly"""
        # Initialize variables
        x = x_in
        alpha_options = tf.nn.softmax(self.network_options(x))[:, :2*d]
        alpha_options_tot = tf.reduce_mean(tf.reduce_sum(alpha_options, 1))
        K = 1. + 0.25*tf.nn.tanh(self.network_K(x))
        rescale = 1. - alpha_options_tot
        alpha = self.network_u[0](x)
        
        # Initial stock prices
        S = tf.ones([batch_size, d])
        S_Q = tf.ones([batch_size, d])
        P = np.ones([M, d, N+1])
        delta_N_acu = 0.
        
        # Time evolution
        for n in range(1, self.NN+1):
            if self.config.trading.leverage_constraints:
                u = self.model.u(alpha, P, x)
            else:
                u = alpha[:, :]
            
            # Track position changes for transaction costs
            if n > 1:
                N_p = tf.multiply(u, 1/S)
                delta_N = N_p - N_m1
                delta_N_acu = delta_N_acu + tf.abs(delta_N)
            N_m1 = tf.multiply(u, 1/S)
            
            # Generate jumps if enabled
            if self.config.model.jumps:
                Z = tf.random.poisson([batch_size, d], self.config.model.lamb)
                Y = tf.random.normal(Z.shape, 
                                   mean=self.config.model.mu*tf.cast(Z, tf.float32), 
                                   stddev=tf.sqrt(tf.cast(Z, tf.float32))*self.config.model.s)
            else:
                Y = tf.zeros([batch_size, d])
            
            # Brownian motion
            dW = tf.random.normal([batch_size, d], mean=0, stddev=np.sqrt(self.h))
            
            # Update wealth and prices
            x = self.model.F_x(x, u, dW, Y, 
                              self.config.trading.bankrupcy_constraint, 
                              self.config.model.jumps)
            
            # Update bond and stock prices
            P[:,0] = self.model.Bond(n)
            S = self.model.F_S(S, dW, Y, self.config.model.jumps, self.h)
            S_Q = self.model_2.F_S(S_Q, dW, Y, self.config.model.jumps, self.h)
            
            # Update allocation at rebalancing dates
            if np.remainder(n, self.diff) == 0 and n < self.NN:
                alpha = self.network_u[int(n/self.diff)](x)
        
        # Calculate final values
        delta_N_acu = tf.reshape(tf.reduce_sum(delta_N_acu, 1), x.shape)
        
        # Option payoffs (Monte Carlo pricing)
        r = self.config.model.r
        T = self.config.model.T
        
        C_0_MC = tf.matmul(tf.ones([batch_size, 1]), 
                          tf.reshape(tf.exp(-r*T)*tf.reduce_mean(
                              tf.maximum(S_Q - K[:, :d], 0), 0), [1, d]))
        P_0_MC = tf.matmul(tf.ones([batch_size, 1]), 
                          tf.reshape(tf.exp(-r*T)*tf.reduce_mean(
                              tf.maximum(K[:, d:] - S_Q, 0), 0), [1, d]))
        
        C_T = tf.maximum(S - K[:, :d], 0) / (C_0_MC + 1e-8)
        P_T = tf.maximum(K[:, d:] - S, 0) / (P_0_MC + 1e-8)
        
        # Final wealth with transaction costs
        if self.config.trading.options_in_p:
            payoff = (tf.reshape(tf.math.reduce_sum(alpha_options[:, :d]*C_T, 1), x.shape) + 
                     tf.reshape(tf.math.reduce_sum(alpha_options[:, d:]*P_T, 1), x.shape))
            x = rescale*x - self.config.trading.curtage*rescale*delta_N_acu + payoff
        else:
            x = x - self.config.trading.curtage*delta_N_acu
        
        # Calculate multi-objective losses
        loss1 = -tf.reduce_mean(x)  # Negative expected wealth
        loss2 = tf.math.reduce_variance(x)  # Variance
        loss3 = -tf.reduce_mean(x[x < tfp.stats.percentile(x, 5)])  # CVaR 5%
        loss4 = -tf.reduce_mean(x[x > tfp.stats.percentile(x, 95)])  # Upper tail
        loss5 = tf.reduce_mean(C_T[:, 0])  # Option cost
        loss6 = tf.math.reduce_mean(K[0, 1])  # Strike price
        loss7 = tf.math.reduce_mean(rescale)  # Allocation balance
        
        return (tf.expand_dims(loss1, 0), tf.expand_dims(loss2, 0),
                tf.expand_dims(loss3, 0), tf.expand_dims(loss4, 0),
                tf.expand_dims(loss5, 0), tf.expand_dims(loss6, 0),
                tf.expand_dims(loss7, 0))


def experiment_training(config: ExperimentConfig, 
                       save_dir: Optional[str] = None,
                       verbose: bool = True) -> Dict[str, Any]:
    """
    Run a complete D-TIPO experiment with improved architecture
    
    Parameters:
    -----------
    config : ExperimentConfig
        Complete configuration for the experiment
    save_dir : Optional[str]
        Directory to save results and checkpoints
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    results : Dict[str, Any]
        Dictionary containing all experiment results
    """
    
    # Import the refactored model class
    from model_classes_refactored import ModelClass
    
    # Create save directory if specified
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config.save(save_path / 'config.json')
        if verbose:
            print(f"Configuration saved to {save_path / 'config.json'}")
    
    # Initialize model parameters
    # d is defined globally
    
    # Set default drift if not provided
    if config.model.b is None:
        config.model.b = tf.reshape(tf.linspace(0.08, 0.04, d), [d, 1])
    
    B = tf.transpose(config.model.b) - config.model.r
    
    # Set default volatility matrix if not provided
    if config.model.sigma is None:
        config.model.sigma = (tf.linalg.diag(tf.linspace(0.18, 0.12, d)) + 
                             0.05 * tf.ones([d, d]) - 
                             tf.constant([[0., 0., 0.1, 0., 0.], 
                                         [0., 0., 0., 0., 0.],
                                         [.1, 0., 0., .0, .0], 
                                         [0., 0., .0, 0., 0.],
                                         [0., 0., 0., 0., 0.]]))
    
    # Calculate theoretical values
    sig_inv = tf.linalg.inv(tf.matmul(config.model.sigma, tf.transpose(config.model.sigma)))
    rho = tf.matmul(tf.matmul(B, sig_inv), tf.transpose(B))[0, 0].numpy()
    gamma = 1/(1 - np.exp(-rho*config.model.T))*(config.model.ExT - 
                                                  np.exp((config.model.r - rho)*config.model.T))
    varxT = np.exp(-rho*config.model.T)/(1 - np.exp(-rho*config.model.T))*(
        config.model.ExT - np.exp(config.model.r*config.model.T))**2
    lam = 1/(2*(gamma - config.model.ExT))
    
    theoretical_values = {
        'B': B.numpy(),
        'sig_inv': sig_inv.numpy(),
        'rho': rho,
        'gamma': gamma,
        'varxT': varxT,
        'lam': lam,
        'optimal_loss': -config.model.ExT + lam*varxT
    }
    
    if verbose:
        print(f"Theoretical values:")
        print(f"  ExT: {config.model.ExT:.4f}")
        print(f"  VarxT: {varxT:.6f}")
        print(f"  Lambda: {lam:.6f}")
        print(f"  Optimal loss: {theoretical_values['optimal_loss']:.6f}")
    
    # Create initial data
    x0 = np.random.uniform(low=1., high=1., size=[M, 1])
    t = np.linspace(0, config.model.T, N + 1)
    h = t[1] - t[0]
    
    # Model parameters
    model_parameters = [
        config.model.model_name, 
        B, 
        config.model.sigma, 
        config.model.r, 
        x0, 
        h, 
        config.model.T
    ]
    
    # Zero drift for second model
    model_2_parameters = model_parameters.copy()
    model_2_parameters[1] = B - B
    
    # Create model instances
    model = ModelClass(model_parameters)
    model_2 = ModelClass(model_2_parameters)
    
    # Create improved network
    full_model = ImprovedFullNetwork(config, model, model_2, h)
    
    # Create optimizer and compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)
    
    def dummy_loss(y_target, y_pred):
        return y_pred
    
    loss_list = [dummy_loss] * 7
    loss_weights = [1., float(lam), 0., 0., 0., 0., 0.]
    
    full_model.compile(optimizer=optimizer, loss=loss_list, loss_weights=loss_weights)
    
    # Learning rate scheduler
    callbacks = []
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.7,  # Reduce LR by half
    patience=2,  # Wait 2 epochs without improvement
    min_lr=0.02,
    verbose=1
)   
    callbacks.append(reduce_lr)

    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    min_delta=0.001,
    patience=2,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=2
)
    callbacks.append(early_stopping)
    
    # Add checkpoint callback if save directory is specified
    if save_dir:
        checkpoint_path = checkpoint_path = save_path / 'checkpoints/weights.{epoch:02d}-{loss:.4f}.weights.h5'
        checkpoint_path.parent.mkdir(exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor='loss',
                save_best_only=True,
                save_weights_only=True
            )
        )
        
    
    # Train the model
    if verbose:
        print(f"\nStarting training for {config.training.num_epochs} epochs...")
    
    zero_vec = np.zeros(x0.shape)
    history = full_model.fit(
        x0, 
        [zero_vec] * 7,
        epochs=config.training.num_epochs,
        batch_size=batch_size,
        callbacks = callbacks,
        verbose=1 if verbose else 0
    )
    
    # Save training history if save directory is specified
    if save_dir:
        history_path = save_path / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(history.history, f, indent=2)
        if verbose:
            print(f"Training history saved to {history_path}")
    
    # Prepare results
    results = {
        'model': full_model,
        'history': history,
        'theoretical_values': theoretical_values,
        'model_class': model,
        'model_2_class': model_2,
        'config': config,
        'train_loss': history.history['loss'],
        'h': h,
        'x0': x0
    }
    
    if verbose:
        print(f"\nTraining completed!")
        print(f"Final loss: {results['train_loss'][-1]:.6f}")
        print(f"Theoretical optimal loss: {theoretical_values['optimal_loss']:.6f}")
    
    return results


def plot_experiment_results(results: Dict[str, Any], 
                           save_path: Optional[str] = None,
                           show_plots: bool = True):
    """
    Create comprehensive plots for experiment results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training loss convergence
    ax = axes[0, 0]
    train_loss = results['train_loss']
    optimal_loss = results['theoretical_values']['optimal_loss']
    
    ax.plot(train_loss, '.-', linewidth=2, markersize=8, label='Training Loss')
    ax.axhline(y=optimal_loss, color='k', linestyle='--', linewidth=2,
               label=f'Optimal Loss: {optimal_loss:.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Convergence')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Loss components over epochs
    ax = axes[0, 1]
    history = results['history'].history
    
    # Extract individual loss components if available
    loss_names = ['loss_1', 'loss_2', 'loss_3', 'loss_4']
    loss_labels = ['Mean', 'Variance', 'CVaR 5%', 'Upper 95%']
    
    for i, (name, label) in enumerate(zip(loss_names, loss_labels)):
        if name in history:
            ax.plot(history[name], label=label, linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Component')
    ax.set_title('Individual Loss Components')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Theoretical vs achieved metrics
    ax = axes[1, 0]
    metrics = ['ExT', 'VarxT', 'Lambda']
    theoretical = [results['config'].model.ExT, 
                  results['theoretical_values']['varxT'],
                  results['theoretical_values']['lam']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars = ax.bar(x, theoretical, width, label='Theoretical', alpha=0.8)
    
    ax.set_ylabel('Value')
    ax.set_title('Key Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Configuration summary
    ax = axes[1, 1]
    ax.axis('off')
    
    config_text = f"""Configuration Summary:
    
Model: {results['config'].model.model_name}
Assets: {d}
Time horizon: {results['config'].model.T} years
Risk-free rate: {results['config'].model.r:.2%}
Jumps: {'Yes' if results['config'].model.jumps else 'No'}

Network: {results['config'].network.neurons}
Activation: {results['config'].network.activation}
Regularization: {results['config'].network.regularization}

Training:
Epochs: {results['config'].training.num_epochs}
Batch size: {batch_size:,}
Learning rate: {results['config'].training.learning_rate}

Trading:
Transaction costs: {results['config'].trading.curtage:.2%}
Leverage constraints: {'Yes' if results['config'].trading.leverage_constraints else 'No'}
Options: {'Yes' if results['config'].trading.options_in_p else 'No'}
    """
    
    ax.text(0.1, 0.9, config_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show_plots:
        plt.show()
    
    return fig


# Example usage
if __name__ == "__main__":
    # Create a default configuration
    config = ExperimentConfig()
    
    # Run experiment
    results = experiment_training(config, save_dir="./experiment_results", verbose=True)
    
    # Plot results
    plot_experiment_results(results, save_path="./experiment_results/summary.png")