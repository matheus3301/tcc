import tensorflow as tf
import keras

class BiLSTM:
    def __init__(self, input_shape, n_neurons, learning_rate, regularizer) -> None:
        
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(shape=input_shape))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_neurons, return_sequences=True, kernel_regularizer=regularizer)))
        self.model.add(tf.keras.layers.Dense(1))
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_function = "mean_squared_error"

    def compile(self):
        self.model.compile(self.optimizer, self.loss_function, metrics=[keras.metrics.RootMeanSquaredError(name="rmse")])
    
    def get_model(self):
        return self.model