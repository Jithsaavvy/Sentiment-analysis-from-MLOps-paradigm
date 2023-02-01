#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Script to productionalize the best model. The models (latest, production) from the
MLflow model registry in EC2 instance are pulled and benchmarked by means of
behavioral testing and evaluation. As a result, the best performing model is
pushed to production and other is archived, so that the production model can be
packaged as a deployable artifact and deployed to AWS Sagemaker instance.
"""

import os
import mlflow
import sys
import pandas as pd
import tensorflow as tf
import behavioral_test
from dataclasses import dataclass, field
from keras.utils import to_categorical
from transformers import BertTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.helper import Config, load_dataframe
from utils.prepare_data import Dataset

config = Config()

@dataclass
class Productionalize:
    """
    Benchmark and push latest model to production based on testing and evaluation.
    """
    tracking_uri: str
    test_data: str = "./test_data.parquet"
    client: mlflow.MlflowClient = None
    test_dataframe: pd.DataFrame = None
    model_name: str = ""
    batch_size: int = 64
    sequence_length: int = 256
    num_classes: int = 3
    latest_version: int = 3
    filter_string = "name LIKE 'sentiment%'"

    def __post_init__(self) -> None:
        """
        Dunder method to set mlflow_tracking_uri and values to some instance variables.

        Returns
        -------
            None

        Raises
        ------
        ConnectionError: Exception
            If mlflow_tracking_uri is invalid.
        """
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
        
        except ConnectionError:
            print(f"Cannot connect to {self.tracking_uri}. Please check and try again!!!")

        else:
            self.client = mlflow.MlflowClient()
            self.latest_version = self.client.get_latest_versions(name = self.model_name)[0].version
            self.test_dataframe = load_dataframe(self.test_data)

    def get_all_registered_models(self) -> None:
        """
        Method to search and display all registered models from model registry in EC2 instance based on
        given filter.

        Parameters
        ----------
            None
        
        Returns
        -------
            None
        """
        # Searching all models with names starting with sentiment
        for model in self.client.search_registered_models(filter_string = self.filter_string):
            for model_version in model.latest_versions:
                print(f"name = {model_version.name}, version = {model_version.version}, stage = {model_version.current_stage}, run_id = {model_version.run_id}")

    def load_models(self) -> tf.function:
        """
        Method to pull and load tensorflow models from model registry to be used for benchmarking.
        It loads two models namely:
                - Latest model => Trained model added to the model registry with latest version.
                - Production model => Model which is already in production stage.

        Parameters
        ----------
            None

        Returns
        -------
        latest_model, production_model: tf.function
            Callable TensorFlow graph that takes inputs and returns inferences.
        """

        latest_model: tf.function = mlflow.tensorflow.load_model(
                                                            model_uri = f"models:/{self.model_name}/{self.latest_version}"
                                                            )

        production_model: tf.function = mlflow.tensorflow.load_model(
                                                            model_uri = f"models:/{self.model_name}/production"
                                                            )

        return latest_model, production_model

    def transform_data(self, dataframe: pd.DataFrame,
                      col_name: str = "cleaned_tweets") -> tf.data.Dataset.zip:
        """
        Method that transform dataframe into tensorflow dataset using BERT tokenizer. It wraps
        Dataset class from `prepare_data.py` module.

        Parameters
        ----------
        dataframe: pd.DataFrame
            Input dataframe
        col_name: str = "cleaned_tweets"
            Name of column containing input texts. Defaults to "cleaned_tweets".

        Returns
        -------
        dataset: tf.data.Dataset.zip
            Tensorflow dataset after batching.
        """

        y_test = to_categorical(dataframe['labels'], self.num_classes)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        dataset = Dataset(tokenizer = tokenizer, dataframe = dataframe,
                          labels = y_test, batch_size = self.batch_size,
                          max_length = self.sequence_length,
                          col_name = col_name).encode_bert_tokens_to_tf_dataset()

        return dataset

    def benchmark_models(self) -> tuple[tuple[float], tuple[float]]:
        """
        Method to benchmark the loaded models from model registry to productionalize them.
        The benchmarking is done by performing behavioral testing of loaded models and
        evaluating them.

        Parameters
        ----------
            None

        Returns
        -------
        latest_model_accuracies, production_model_accuracies: tuple(tuple[float], tuple[float])
            Resulting accuracies from testing and evaluation with perturbed and test data
            respectively.
        """

        latest_model, production_model = self.load_models()

        # Minimum Functionality test
        sample_mft_dataframe = load_dataframe("./scripts/test_data/sample_test_data_for_mft.parquet")
        negated_dataframe = behavioral_test.min_functionality_test(sample_mft_dataframe)
        perturbed_dataset_mft = self.transform_data(dataframe = negated_dataframe, col_name = "negated_text")
        accuracy_latest_model_mft = behavioral_test.run(test_name = "MFT_latest", model = latest_model,
                                                        test_dataset = perturbed_dataset_mft, dataframe = negated_dataframe)
        accuracy_production_model_mft = behavioral_test.run(test_name = "MFT_production", model = production_model,
                                                        test_dataset = perturbed_dataset_mft, dataframe = negated_dataframe)

        # Invariance test (Inv)
        perturbed_dataframe_inv = self.test_dataframe.tail(100)
        perturbed_dataframe_inv["cleaned_tweets"] = perturbed_dataframe_inv["cleaned_tweets"].apply(
                                                                            lambda text: behavioral_test.invariance_test(text)
                                                                            )
        perturbed_dataset_inv = self.transform_data(dataframe = perturbed_dataframe_inv)
        accuracy_latest_model_inv = behavioral_test.run(test_name = "Invariance_latest", model = latest_model,
                                                        test_dataset = perturbed_dataset_inv, dataframe = perturbed_dataframe_inv)
        accuracy_production_model_inv = behavioral_test.run(test_name = "Invariance_production", model = production_model,
                                                        test_dataset = perturbed_dataset_inv, dataframe = perturbed_dataframe_inv)

        # Model evaluation using full test data
        test_dataset = self.transform_data(dataframe = self.test_dataframe)
        latest_model_score = latest_model.evaluate(test_dataset)
        production_model_score = production_model.evaluate(test_dataset)

        # Wrap results in the tuple
        latest_model_accuracies = (accuracy_latest_model_mft, accuracy_latest_model_inv, latest_model_score[1])
        production_model_accuracies = (accuracy_production_model_mft, accuracy_production_model_inv, production_model_score[1])

        return latest_model_accuracies, production_model_accuracies

    def push_new_model_to_production(self, latest_model_accuracies: tuple[float],
                                    production_model_accuracies: tuple[float]) -> bool:
        """
        Method to push the latest-best model to production stage based on
        testing and evaluation metrics.

        Parameters
        ----------
        latest_model_accuracies: tuple[float]
            Resulting accuracies from testing and evaluation of latest model.
        production_model_accuracies: tuple[float]
            Resulting accuracies from testing and evaluation of production model.

        Returns
        -------
        success: bool
            True if latest model is pushed to production, else False.
        """

        print(f"Latest model accuracies: {latest_model_accuracies},\nProduction model accuracies: {production_model_accuracies}")

        if latest_model_accuracies > production_model_accuracies:
            self.client.transition_model_version_stage(
                                                    name = self.model_name,
                                                    version = self.latest_version,
                                                    stage = "Production")

            print("Transitioned latest model to production!!")
            success = True

        else:
            print("Cannot transition the model stage. Latest model cannot outperform production model in all conducted tests!!!")
            success = False

        return success

def main() -> None:
    productionalize_ = Productionalize(tracking_uri = config["model-tracking"]["mlflow_tracking_uri"],
                                    test_data = config["files"]["test_data"],
                                    model_name = config["model-registry"]["model_name"],
                                    batch_size = config["train-parameters"]["batch_size"],
                                    sequence_length = config["train-parameters"]["sequence_length"]
                                    )

    accuracy_latest_model, accuracy_production_model = productionalize_.benchmark_models()

    success_ = productionalize_.push_new_model_to_production(accuracy_latest_model, accuracy_production_model)

    if success_:
        productionalize_.get_all_registered_models()

if __name__ == "__main__":
    main()