"""
@author: Jithin Sasikumar

Module to transform preprocessed dataframe (parquet or csv) into tf.data.Dataset format
which creates an efficient input pipeline that in turn be fed into the tensorflow model.
BERT tokenizer is used instead of normal tokenizer for better embeddings.

"""
import math
import pandas as pd
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from transformers import BertTokenizer

@dataclass
class Dataset:
    """
    Dataclass that encodes and transforms dataframe into tensorflow dataset.
    """
    tokenizer: BertTokenizer
    dataframe: pd.DataFrame = field(default_factory = pd.DataFrame())
    labels: np.ndarray = None
    batch_size: int = 64
    max_length: int = 256
    train: bool = False
    col_name: str = "cleaned_tweets"

    @property
    def list_of_texts(self) -> list[str]:
        """
        Class property to convert text column of dataframe to list of strings
        for processing.

        Parameters
        ----------
            None
        
        Returns
        -------
        list[str]
            List of texts
        """
        return self.dataframe[self.col_name].tolist()

    @property
    def shuffle_size(self) -> int:
        """
        Class property to calculate the shuffle size for dataset.

        Parameters
        ----------
            None
        
        Returns
        -------
        shuffle_size: int
        """
        return math.ceil(len(self.list_of_texts) / self.batch_size)

    def encode_bert_tokens_to_tf_dataset(self) -> tf.data.Dataset.zip:
        """
        Transform tokens into tensorflow dataset. The dataset is batched and
        shuffled.
        
        BERT tokenizer is used => (i.e.) The texts are tokenized and each token
        is encoded into unique IDs referred as input_ids by means of vocabulary.

        Parameters
        ----------
            None
        
        Returns
        -------
        dataset: tf.data.Dataset.zip
            Tensorflow dataset after batching and shuffling.
        """
        tokenized: BertTokenizer = self.tokenizer(
                                                text = self.list_of_texts,
                                                add_special_tokens = True,
                                                max_length = self.max_length,
                                                padding = "max_length",
                                                return_tensors = "tf",
                                                return_attention_mask = False,
                                                return_token_type_ids = False,
                                                verbose = True
                                            )

        input_ids = tf.data.Dataset.from_tensor_slices(np.array(tokenized['input_ids']))
        labels = tf.data.Dataset.from_tensor_slices(self.labels)
        # Zipping input_ids and labels as a single dataset object
        dataset = tf.data.Dataset.zip((input_ids, labels))

        if self.train:
            return dataset.shuffle(self.shuffle_size).batch(self.batch_size)

        return dataset.batch(self.batch_size)