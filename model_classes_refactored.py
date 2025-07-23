import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import norm
from typing import Optional, Tuple, Dict, Any
tf.keras.backend.set_floatx('float32')

# Global dimension parameter
d = 5


class ModelClass:
    """
    Refactored ModelClass that encapsulates market model dynamics
    """
    def __init__(self, model_parameters: list):
        self.model_name = model_parameters[0]
        print(f"Initializing model: {self.model_name}")
        
        if self.model_name == 'GBM_MV':
            self.B = model_parameters[1]
            self.sigma = model_parameters[2]
            self.r = model_parameters[3]
            self.x0 = model_parameters[4]
            self.h = model_parameters[5]
            self.T = model_parameters[6]
            self.b = self.B + self.r

    def F_x(self, x: tf.Tensor, u: tf.Tensor, dW: tf.Tensor, Y: tf.Tensor, 
            bankrupcy_constraint: bool, jumps: bool) -> tf.Tensor:
        """
        Update rule for wealth process
        """
        M = dW.shape[0]
        
        if self.model_name == 'GBM_MV':
            if bankrupcy_constraint:
                bc = tf.where(x > 0, 1., 0.)
                x_kp1 = x + bc * ((self.r * x + tf.matmul(u, tf.transpose(self.B))) * self.h + 
                                  tf.reshape(tf.reduce_sum(tf.matmul(u, self.sigma) * dW, axis=1), [M, 1]))
            else:
                x_kp1 = x + ((self.r * x + tf.matmul(u, tf.transpose(self.B))) * self.h + 
                            tf.reshape(tf.reduce_sum(tf.matmul(u, self.sigma) * dW, axis=1), [M, 1]))
            
            if jumps:
                sum_Y = tf.reshape(tf.reduce_sum(u * Y, 1), x_kp1.shape)
                if bankrupcy_constraint:
                    bc = tf.where(x > 0, 1., 0.)
                    x_kp1 = x_kp1 + bc * sum_Y
                else:
                    x_kp1 = x_kp1 + sum_Y
                    
        return x_kp1

    def F_x_np(self, x: np.ndarray, u: np.ndarray, dW: np.ndarray, Y: np.ndarray,
               bankrupcy_constraint: bool, jumps: bool) -> np.ndarray:
        """
        Update rule for wealth process (numpy version)
        """
        M = dW.shape[0]
        
        if self.model_name == 'GBM_MV':
            B_np = self.B.numpy() if hasattr(self.B, 'numpy') else self.B
            sigma_np = self.sigma.numpy() if hasattr(self.sigma, 'numpy') else self.sigma
            
            if bankrupcy_constraint:
                bc = np.where(x > 0, 1., 0.)
                x_kp1 = x + bc * ((self.r * x + np.matmul(u, np.transpose(B_np))) * self.h + 
                                  np.reshape(np.sum(np.matmul(u, sigma_np) * dW, axis=1), [M, 1]))
            else:
                x_kp1 = x + ((self.r * x + np.matmul(u, np.transpose(B_np))) * self.h + 
                            np.reshape(np.sum(np.matmul(u, sigma_np) * dW, axis=1), [M, 1]))
            
            if jumps:
                sum_Y = np.reshape(np.sum(u * Y, 1), x_kp1.shape)
                if bankrupcy_constraint:
                    bc = np.where(x > 0, 1., 0.)
                    x_kp1 = x_kp1 + bc * sum_Y
                else:
                    x_kp1 = x_kp1 + sum_Y
                    
        return x_kp1

    def u(self, alpha: tf.Tensor, P: np.ndarray, x: tf.Tensor) -> tf.Tensor:
        """
        Compute portfolio allocation with leverage constraints
        """
        # Convert numpy P to TensorFlow tensor for computation
        P_tf = tf.convert_to_tensor(P, dtype=tf.float32)
        a = tf.multiply(tf.nn.softmax(alpha * P_tf[:, 1:]), x)
        bb = tf.concat([tf.ones([tf.shape(P_tf)[0], 1]), tf.zeros([tf.shape(P_tf)[0], tf.shape(P_tf)[1]-1])], axis=1)
        u = tf.multiply(alpha, bb)
        return u

    def Bond(self, n: int) -> float:
        """
        Calculate bond value at time step n
        """
        B_kp1 = np.exp(self.r * self.h * n)
        return B_kp1

    def F_S(self, S: tf.Tensor, dW: tf.Tensor, Y: tf.Tensor, jumps: bool, h: float) -> tf.Tensor:
        """
        Update rule for stock prices
        """
        tr_sig2 = tf.linalg.diag(tf.reduce_sum(self.sigma * self.sigma, 1))
        tr_sig2_rd = tf.matmul(tf.ones(S.shape), tr_sig2)
        S_kp1 = S * tf.exp((self.b - tr_sig2_rd/2) * h + tf.matmul(dW, tf.transpose(self.sigma)))
        
        if jumps:
            S_kp1 = S_kp1 + Y * S_kp1
            
        return S_kp1

    def F_S_np(self, S: np.ndarray, dW: np.ndarray, Y: np.ndarray, jumps: bool, h: float) -> np.ndarray:
        """
        Update rule for stock prices (numpy version)
        """
        sigma_np = self.sigma.numpy() if hasattr(self.sigma, 'numpy') else self.sigma
        b_np = self.b.numpy() if hasattr(self.b, 'numpy') else self.b
        
        tr_sig2 = np.diag(np.sum(sigma_np * sigma_np, 1))
        tr_sig2_rd = np.matmul(np.ones(S.shape), tr_sig2)
        S_kp1 = S * np.exp((b_np - tr_sig2_rd/2) * h + np.matmul(dW, np.transpose(sigma_np)))
        
        if jumps:
            S_kp1 = S_kp1 + Y * S_kp1
            
        return S_kp1

    def CP(self, S: tf.Tensor, K: tf.Tensor, T: float, n: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Calculate call and put option prices using Black-Scholes
        """
        tfd = tfp.distributions
        normal = tfd.Normal(loc=0.0, scale=1.0)
        
        sig = tf.sqrt(tf.reduce_sum(self.sigma**2, 1))
        sqrt = tf.cast(tf.sqrt(T - n*self.h), tf.float32)
        disc = tf.cast(tf.exp(-self.r*(T-n*self.h)), tf.float32)
        
        d1 = 1/(sig*sqrt) * (tf.math.log(S/K) + (self.r + sig**2/2)*(T - n*self.h))
        d2 = d1 - sig*sqrt
        
        C = normal.cdf(d1)*S - normal.cdf(d2)*K*disc
        P = K*disc - S + C
        
        return C, P


class SubNetwork(tf.keras.Model):
    """
    Refactored SubNetwork for portfolio allocation
    """
    def __init__(self, d: int, regularization: float = 1e-6, 
                 neurons: list = None, activation: str = 'relu'):
        super(SubNetwork, self).__init__()
        
        if neurons is None:
            neurons = [16, 16, 16]
            
        self.neurons = neurons
        self.activation = activation
        self.d = d
        self.regularization = regularization
        
        regu = tf.keras.regularizers.l2(self.regularization) if self.regularization and self.regularization > 0 else None
        
        # Define layers
        self.dense_layers = []
        self.batch_norm = []
        
        for i in range(len(self.neurons)):
            self.dense_layers.append(
                tf.keras.layers.Dense(self.neurons[i],
                                    activation=self.activation,
                                    use_bias=True,
                                    kernel_regularizer=regu)
            )
            self.batch_norm.append(tf.keras.layers.BatchNormalization())
        
        # Define output layer
        self.dense_out = tf.keras.layers.Dense(self.d,
                                             activation='linear',
                                             use_bias=True,
                                             kernel_regularizer=None)

    def call(self, x_in, training=None):
        """Forward propagation"""
        x = x_in
        for i in range(len(self.neurons)):
            x = self.dense_layers[i](x)
            x = self.batch_norm[i](x, training=training)
        x = self.dense_out(x)
        return x


class FullNetwork(tf.keras.Model):
    """
    Refactored FullNetwork that takes configuration as input
    """
    def __init__(self, d: int, l: int, regularization: float, neurons: list,
                 N: int, NN: int, h: float, activation: str, config: Dict[str, Any]):
        super(FullNetwork, self).__init__()
        
        self.network_u = []
        self.NN = NN
        self.N = N
        self.diff = int(NN/N)
        self.h = h
        self.d = d
        self.dim = d
        self.config = config  # Store configuration
        
        # Initialize networks for portfolio allocation
        self.network_u.append(
            SubNetwork(d=self.dim, regularization=None, neurons=[], activation='linear')
        )
        
        for _ in range(self.N):
            self.network_u.append(
                SubNetwork(d=self.d, regularization=regularization,
                         neurons=neurons, activation=activation)
            )
        
        # Networks for options
        self.network_options = SubNetwork(d=2*self.dim + 1, regularization=None,
                                        neurons=[], activation='linear')
        self.network_K = SubNetwork(d=2*self.dim, regularization=None,
                                  neurons=[], activation='linear')

    def call(self, inputs):
        """
        Forward propagation with configuration-based parameters
        
        Parameters:
        -----------
        inputs : dict
            Dictionary containing:
            - 'x_in': Initial wealth
            - 'model': ModelClass instance
            - 'model_2': Second ModelClass instance
            - 'S0': Initial stock prices
            - 'batch_size': Batch size
            - 'config': Trading configuration
        """
        # Extract inputs
        x_in = inputs['x_in'] if isinstance(inputs, dict) else inputs
        
        # Get configuration from stored config or inputs
        if isinstance(inputs, dict):
            model = inputs['model']
            model_2 = inputs['model_2']
            S0 = inputs['S0']
            batch_size = inputs['batch_size']
            trading_config = inputs.get('config', self.config)
        else:
            # Fallback to global variables (for backward compatibility)
            model = globals().get('model')
            model_2 = globals().get('model_2')
            S0 = globals().get('S0')
            batch_size = globals().get('batch_size')
            trading_config = self.config
        
        # Extract trading parameters
        curtage = trading_config.get('curtage', 0.005)
        leverage_constraints = trading_config.get('leverage_constraints', False)
        options_in_p = trading_config.get('options_in_p', False)
        bankrupcy_constraint = trading_config.get('bankrupcy_constraint', False)
        jumps = trading_config.get('jumps', False)
        
        # Initialize variables
        x = x_in
        K_options = tf.nn.softmax(self.network_options(x))[:, :2*self.d]
        alpha_options_tot = tf.reduce_mean(tf.reduce_sum(K_options, 1))
        K = 1. + 0.25*tf.nn.tanh(self.network_K(x))
        rescale = 1. - alpha_options_tot
        alpha = self.network_u[0](x)
        x0 = x[:, :]
        
        S = S0
        S_Q = tf.ones([batch_size, self.d])
        P = np.ones([batch_size, self.d, self.N+1])
        delta_N_acu = 0.
        
        # Evolution through time
        for n in range(1, self.NN+1):
            if leverage_constraints:
                u = model.u(alpha, P[:, :, n], x)
            else:
                u = alpha[:, :]
            
            if n > 1:
                N_p = tf.multiply(u, 1/S)
                delta_N = N_p - N_m1
                delta_N_acu = delta_N_acu + tf.abs(delta_N)
            N_m1 = tf.multiply(u, 1/S)
            
            # Jump process
            if jumps:
                lamb = trading_config.get('lamb', 0.05)
                mu = trading_config.get('mu', 0.0)
                s = trading_config.get('s', 0.2)
                Z = tf.random.poisson([batch_size, self.d], lamb)
                Y = tf.random.normal(Z.shape, mean=mu*tf.cast(Z, tf.float32), 
                                   stddev=tf.sqrt(tf.cast(Z, tf.float32))*s)
            else:
                Y = np.zeros([batch_size, self.d])
            
            # Brownian motion
            dW = tf.random.normal(S.shape, mean=0, stddev=tf.sqrt(tf.cast(self.h, tf.float32)))
            
            # Update wealth and stock prices
            x = model.F_x(x, u, dW, Y, bankrupcy_constraint, jumps)
            # Update bond prices in P array
            P[:, 0, n] = model.Bond(n)
            S = model.F_S(S, dW, Y, jumps, self.h)
            S_Q = model_2.F_S(S_Q, dW, Y, jumps, self.h)
            
            # Update portfolio allocation
            if np.remainder(n, self.diff) == 0:
                if n < self.NN:
                    alpha = self.network_u[int(n/self.diff)](x)
        
        # Calculate terminal values and losses
        delta_N_acu = tf.reshape(tf.reduce_sum(delta_N_acu, 1), x.shape)
        
        # Option payoffs
        C_0_MC = tf.matmul(tf.ones([batch_size, 1]), 
                          tf.reshape(tf.exp(-model.r*model.T)*tf.reduce_mean(
                              tf.maximum(S_Q - K[:, :self.d], 0), 0), [1, self.d]))
        P_0_MC = tf.matmul(tf.ones([batch_size, 1]), 
                          tf.reshape(tf.exp(-model.r*model.T)*tf.reduce_mean(
                              tf.maximum(K[:, self.d:] - S_Q, 0), 0), [1, self.d]))
        
        C_T = tf.maximum(S - K[:, :self.d], 0) / C_0_MC
        P_T = tf.maximum(K[:, self.d:] - S, 0) / P_0_MC
        
        # Final wealth calculation
        if options_in_p:
            payoff = (tf.reshape(tf.math.reduce_sum(K_options[:, :self.d]*C_T, 1), x.shape) + 
                     tf.reshape(tf.math.reduce_sum(K_options[:, self.d:]*P_T, 1), x.shape))
            x = rescale*x - curtage*rescale*delta_N_acu + payoff
        else:
            x = x - curtage*delta_N_acu
        
        # Calculate losses
        loss1 = -tf.reduce_mean(x)
        loss2 = tf.math.reduce_variance(x)
        loss3 = -tf.reduce_mean(x[x < tfp.stats.percentile(x, 5)])
        loss4 = -tf.reduce_mean(x[x > tfp.stats.percentile(x, 95)])
        loss5 = tf.reduce_mean(C_T[:, 0])
        loss6 = tf.math.reduce_mean(K[0, 1])
        loss7 = tf.math.reduce_mean(rescale)
        
        return (tf.expand_dims(loss1, 0), tf.expand_dims(loss2, 0),
                tf.expand_dims(loss3, 0), tf.expand_dims(loss4, 0),
                tf.expand_dims(loss5, 0), tf.expand_dims(loss6, 0),
                tf.expand_dims(loss7, 0))