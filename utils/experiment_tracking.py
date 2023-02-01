"""
@author: Jithin Sasikumar

Module to track model training and log the model artifacts, resulting metrics
and parameters. For that purpose, `MLFlow` is used. This module has the flexibility
to extend its functionality to support other tracking mechanism like tensorboard etc.
It is facilitated via `ExperimentTracker protocol` which is similar to interface.
"""

import mlflow
from typing import Protocol
from dataclasses import dataclass

class ExperimentTracker(Protocol):
    """
    Interface to track experiments by inherting from Protocol class.
    """
    def __start__(self):
        ...

    def log(self):
        ...

    def end(self):
        ...

@dataclass
class MLFlowTracker:
    """
    Dataclass to track experiment via MLFlow.

    Instance variables
    ------------------
    experiment_name: str
        Name of the experiment to be activated or created.
    tracking_uri: str
        URI of EC2 instance where MLflow server is hosted.
    run_name: str
        Name of training run pertaining to an experiment.
    experiment: bool
        Boolean to create a new experiment, else False.
    """

    experiment_name: str
    tracking_uri: str
    run_name: str
    experiment: bool
    
    def __start__(self) -> None:
        """
        Dunder method to start a new mlflow run in MLFlow server and set
        model tracking URI and create experiment.

        Parameters
        ----------
            None
        
        Returns
        -------
            None

        Raises
        ------
        ConnectionError: Exception
            If mlflow tracking URI doesn't exist or invalid.
        """
        try:
            mlflow.set_tracking_uri(self.tracking_uri)

        except ConnectionError:
            print(f"Cannot connect to {self.tracking_uri}. Please check and validate the URI!!")

        else:
            if self.experiment:
                exp_id = mlflow.create_experiment(self.experiment_name)
                experiment = mlflow.get_experiment(exp_id)

            else:
                experiment = mlflow.set_experiment(self.experiment_name)

            mlflow.start_run(run_name = self.run_name,
                            experiment_id = experiment.experiment_id)
    
    def log(self) -> None:
        """
        Initialize auto-logging for tracking. This will log model
        artifacts in S3 bucket, parameters and metrics in the EC2 instance.
        """
        self.__start__()
        mlflow.tensorflow.autolog()

    def end(self) -> None:
        """
        End an active MLflow run.
        """
        mlflow.end_run()