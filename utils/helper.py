"""
@author: Jithin Sasikumar

Module consisting of helper functions which is generic across the project.
"""

import re
import os
import nltk
import pandas as pd
from textblob import TextBlob
from nltk.probability import FreqDist
import tomli as tomlib
from typing import Any
from dataclasses import dataclass
from airflow import settings
from airflow.exceptions import AirflowFailException
from airflow.models.connection import Connection

class Config:
    """
    Loads all configurations from `config.toml` for the project.
    """
    def __new__(cls) -> dict[str, Any]:
        """
        Dunder method to load config.

        Parameters
        ----------
        cls
            Class to be instantiated.

        Returns
        -------
        config: dict[str, Any]
            Loaded configurations as dict.
        """

        with open("./config/config.toml", mode="rb") as config_file:
            config = tomlib.load(config_file)
        return config

def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Helper function to load any parquet file as pandas dataframe.

    Parameters
    ----------
    file_path: str
        Path to input parquet file.
    
    Returns
    -------
    dataframe: pd.DataFrame
    """
    this_dir = os.getcwd()
    dataframe_path = os.path.join(this_dir, file_path)
    dataframe = pd.read_parquet(path = dataframe_path, engine = "pyarrow")
    return dataframe

@dataclass
class Connections:
    """
    Dataclass to configure and set Airflow connections.
    """
    new_connection: Connection

    def create_connections(self) -> bool:
        """
        Method to create a new airflow connection

        Parameters
        ----------
            None

        Returns
        -------
        bool
            True if connection is created, else False.

        Raises
        ------
        AirflowFailException: Exception
            If connection cannot be created or invalid.
        """
        try:
            session = settings.Session()
            connection_name = session.query(Connection).filter(
                                                        Connection.conn_id == self.new_connection.conn_id
                                                        ).first()

            if str(connection_name) != str(self.new_connection.conn_id):
                session.add(self.new_connection)
                session.commit()

        except Exception as exc:
            raise AirflowFailException( f"Error when creating new connection:{exc}") from exc

        else:
            return True
        
        finally:
            session.close()

def remove_noise(text: str) -> str:
    """
    Helper function to remove noise from text as part of text cleaning
    using regular expressions (regex).

    Parameters
    ----------
    text: str
        Input text
    
    Returns
    -------
    Cleaned text
    """

    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
            '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', text)
    text = re.sub("(@[A-Za-z0-9_]+)","", text)
    text = re.sub('\n',' ', text)
    text = re.sub('#','', text)
    
    return text

def calculate_polarity(text: str) -> float:
    """
    Helper function to calculate text polarity.

    Parameters
    ----------
    text: str
        Input text
    
    Returns
    -------
    polarity: float
    """
    return TextBlob(text).sentiment.polarity

def remove_stopwords(tokens: list[str],
                    stopwords_: nltk.corpus.stopwords) -> list[str]:
    """
    Helper function to remove stopwords from given input tokens.

    Parameters
    ----------
    tokens: list[str]
        List of tokens pertaining to each text.
    stopwords_: nltk.corpus.stopwords
        List of stopwords defined in NLTK.

    Returns
    -------
    list[str]
        Resultant list of text with no stopwords.
    """
    return [token for token in tokens if token not in stopwords_]

def remove_less_frequent_words(dataframe) -> pd.DataFrame:
    """
    Helper function to remove the words that are less frequent (< 2 times).

    Parameters
    ----------
    dataframe: pd.DataFrame
        Input dataframe

    Returns
    -------
    Resultant dataframe with less frequent words removed.
    """

    dataframe['tokenized_strings'] = dataframe['tokenized_tweets'].apply(
                                                                    lambda tokens: ' '.join(
                                                                                    [token for token in tokens if len(token) > 2]
                                                                                    )
                                                                        )
    tokenized_words = nltk.tokenize.word_tokenize(' '.join(
                                                            [word
                                                            for word in dataframe['tokenized_strings']
                                                            ]
                                                        )
                                                )
    frequency_distribution = FreqDist(tokenized_words)
    dataframe['tokenized_strings'] = dataframe['tokenized_tweets'].apply(
                                                                        lambda tweets: ' '.join(
                                                                                [tweet for tweet in tweets
                                                                                if frequency_distribution[tweet] > 2
                                                                                ]
                                                                            )
                                                                        )
    return dataframe

def assign_sentiment_labels(score: float) -> str:
    """
    Helper function to assign sentiment labels to polarity scores.

    Parameters
    ----------
    score: float
        Polarity score of each text.

    Returns
    -------
    sentiment_label: str
    """
    if score > 0.25:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"