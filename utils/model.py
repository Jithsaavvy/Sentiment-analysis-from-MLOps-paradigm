"""
@author: Jithin Sasikumar

Model to define deep neural network for training.

Bi-directional LSTM (biLSTM) network is used for this project encompassing an
embedding layer, stack of biLSTM layers followed by fully connected dense layers
with dropout. This module provides the flexibility to add any other models
by inheriting Models(ABC).

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

class Models(ABC):
    """
    Abstract base class that defines and creates model.
    """
    @abstractmethod
    def define_model(self):
        pass

    @abstractmethod
    def create_model(self):
        pass


@dataclass
class BiLSTM_Model(Models):
    """
    Dataclass to create biLSTM model inheriting Models class.
    """
    vocab_size: int
    num_classes: int
    embedding_dim: int = 64
    input_length: int = 128

    def define_model(self) -> Sequential:
        """
        Method to define model that can be used for training and inference.
        The existing model can also be tweaked by changing parameters,
        based on the requirements.

        Parameters
        ----------
            None

        Returns
        -------
        keras.models.Sequential
        """
        return Sequential(
                    [

                    # Embedding layer that expects the following:
                    # Size of vocabulary, Output embedding vectors & Size of each input sequence
                    Embedding(self.vocab_size, self.embedding_dim, input_length = self.input_length),

                    #Bidirectional LSTM layers
                    Bidirectional(LSTM(self.embedding_dim, return_sequences=True)),
                    Bidirectional(LSTM(64, return_sequences = True)),
                    Bidirectional(LSTM(32)),
                    
                    #Dense layers
                    Dense(self.embedding_dim, activation = 'relu'),
                    Dense(64, activation = 'relu'),
                    Dropout(0.25),
                    Dense(self.num_classes, activation = 'softmax')
                    ]
                )
        
    def create_model(self) -> Sequential:
        """
        Method to create the model defined by define_model() method
        and prints the model summary.

        Parameters
        ----------
            None

        Returns
        -------
        model: keras.models.Sequential
            Created model
        """

        model: Sequential = self.define_model()
        model.summary()
        return model