"""
@author: Jithin Sasikumar

Module to define and perform behavioral testing of sentiment analysis model. It is based on
the paper [1] that proposes three different types of tests but only two tests are performed
in this project namely -
    - Minimum Functionality test (MFT)
    - Invariance test (INV)

Note
----
    Model testing differs from model evaluation.

References
----------
[1] Beyond Accuracy: Behavioral Testing of NLP models with CheckList
[2] https://github.com/marcotcr/checklist
"""

import os
import spacy
import numpy as np
import pandas as pd
import tensorflow as tf
from checklist.perturb import Perturb
from keras.models import Sequential
from sklearn.metrics import accuracy_score
nlp = spacy.load('en_core_web_sm')


def min_functionality_test(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perturb test data which is suitable to perform MFT. A specific behavior (or)
    capability of the model is tested. In this case, the specific behavior to be tested
    is `negation` (i.e.) how well the model handles negated inputs.

    More detailed information can be found in the README.md

    Parameters
    ----------
    dataframe: pd.DataFrame
        Test dataframe consisting of original text.

    Returns
    -------
    negated_dataframe: pd.DataFrame
        Dataframe after negating original texts with their corresponding labels.
    """

    original_text: list = dataframe["sample_text"].tolist()
    true_labels: list = dataframe["labels"].tolist()
    piped_text  = list(nlp.pipe(original_text))

    # Adding negation to original text using `checklist` package
    perturbed_data = Perturb.perturb(piped_text, Perturb.add_negation)
    negated_texts: list = [text[1] for text in perturbed_data.data]

    negated_dataframe = pd.DataFrame(
                                    list(zip(negated_texts, true_labels)),
                                    columns = ["negated_text", "labels"]
                                    )

    return negated_dataframe

def invariance_test(text: str) -> str:
    """
    Function to perturb test data which is suitable to perform invariance test.
    The test data is perturbed in a way that their context are preserved. Despite
    perturbing the data, the model is expected to generalize well and predict the
    same labels pertaining to the actual test data.

    Two perturbations are added namely:
        - Adding typos to the actual test data.
        - Expanding contractions to the same.

    Parameters
    ----------
    text: str
        Input text from actual test data.

    Returns
    -------
    perturbed_text: str
        Resulting text after applying two perturbations.
    """

    text_with_typo = str(Perturb.add_typos(text))
    perturbed_text = Perturb.expand_contractions(text_with_typo)
    return perturbed_text


def run(test_name: str, model: Sequential,
        test_dataset: tf.data.Dataset.zip,
        dataframe: pd.DataFrame) -> float:
    """
    Function to perform specified behavioral test using perturbed data.

    Parameters
    ----------
    test_name: str
        Name of test (MFT or invariance).
    model: Sequential
        Trained (or) productionalized model pulled from model registry
        in EC2 instance.
    test_dataset: tf.data.Dataset.zip
        Perturbed dataset transformed to tensorflow dataset format.
    dataframe: pd.DataFrame
        Dataframe where test results will be written and saved at the
        end as CSV for analysis and benchmarking.

    Returns
    -------
    test_accuracy: float
    """
    try:
        for text, _ in test_dataset.take(1):
            text_ = text.numpy()

    except Exception:
        print(f"Exception occurred when trying to access {test_dataset}. Please check!!")
    
    else:
        predicted_probabilities = model.predict(text_)
        predicted_labels = np.argmax(
                                    np.array(predicted_probabilities),
                                    axis = 1
                                    )

        dataframe["predicted_labels"] = predicted_labels
        dataframe["predicted_probabilities"] = predicted_probabilities.tolist()

        # Save test results as CSv
        dataframe_path = os.path.join(os.getcwd(), "test_results")
        dataframe.to_csv(f"{dataframe_path}/{test_name}_test_results.csv", index = False)

        test_accuracy = accuracy_score(
                                        y_true = dataframe['labels'].tolist(),
                                        y_pred = dataframe['predicted_labels'].tolist()
                                    )

        return test_accuracy