import os
import tensorflow as tf
import numpy as np


class model:
    def __init__(self, path):
        # Model-related paths inside the 'path' folder
        model_paths = [
            os.path.join(path, 'EfficientNet'),
            os.path.join(path, 'ConvNext'),
            os.path.join(path, 'Resnet')
        ]

        # Load all three models
        self.models = [tf.keras.models.load_model(model_path) for model_path in model_paths]

    def predict(self, X):
        # Collect predictions from all models
        predictions = [model.predict(X) for model in self.models]

        # Convert probabilities to binary votes
        binary_votes = [np.where(pred >= 0.5, 1, 0) for pred in predictions]

        # Somma i voti per ogni classe
        sum_votes = np.sum(binary_votes, axis=0)

        
        # Determine the winning class for each input
        final_predictions = np.where(sum_votes > len(self.models) / 2, 1, 0)
        # Convert the result to a 1D array
        final_predictions = np.squeeze(final_predictions)

        output = tf.convert_to_tensor(final_predictions, dtype=tf.int32)
        return output
    
