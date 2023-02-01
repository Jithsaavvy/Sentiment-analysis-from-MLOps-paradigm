#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Script to define model training pipeline as Airflow DAG that trains Bi-LSTM model with the
processed data from data warehouse. In the DAG, in order to improve the training time and
efficiency, the model training is done within an external (user-build) docker container with
tensorflow-gpu base image and it is not included in airflow docker compose.
It is a GPU accelerated training.

"""

import os
import sys
import pandas as pd
from datetime import datetime
from airflow.decorators import task, dag
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.helper import Config

# Load all configurations from config.toml
config = Config()

@dag(dag_id = "model_training", start_date = datetime(2023,1,1), schedule_interval = "@monthly", catchup = False)
def model_training_pipeline_dag() -> None:
    """
    Pipeline to perform the GPU accelerated model training within the user-build docker image

    Returns
    -------
            None
    """

    @task(task_id = "load_data_from_warehouse")
    def pull_snowflake_data_as_df(query: str) -> pd.DataFrame:
        """
        Task 1 => Loaded data as a result of ETL pipeline is fetched from the database
        table of snowflake data warehouse as a dataframe. This will be used for
        model training.

        Parameters
        ----------
        query: str
            Database query

        Returns
        -------
        dataframe: pd.DataFrame
            Fetched data
        """
        try:
            snowflake_conn = SnowflakeHook(
                                        snowflake_conn_id = "snowflake_conn",
                                        account = os.environ["ACCOUNT"],
                                        warehouse = os.environ["WAREHOUSE"],
                                        database = os.environ["DATABASE"],
                                        schema = os.environ["SCHEMA"],
                                        role = os.environ["ROLE"]
                                    )

            cursor = snowflake_conn.cursor().execute(query)
            dataframe = cursor.fetch_pandas_all()

            return dataframe

        except Exception as exc:
            raise ConnectionError("Snowflake connection error. Please check and try again!!") from exc

        finally:
            cursor.close()
            snowflake_conn.close()

    
    # Task 2 => Refer /task_definitions/model_training.py for documentation
    train_model = DockerOperator(
                                task_id = "train_model_task",
                                image = "model_training_tf:latest",
                                auto_remove = True,
                                docker_url = "unix://var/run/docker.sock",
                                api_version = "auto",
                                command = "python3 model_training.py"
                                )

    pull_snowflake_data_as_df(config["misc"]["query"]) >> train_model

model_train_dag = model_training_pipeline_dag()