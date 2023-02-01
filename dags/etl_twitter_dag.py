#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Script to define the data pipeline as Airflow DAG that performs ETL (Extract Load Transform) tasks such as
scraping tweets from twitter, labelling, cleaning, normalizing and preprocessing the raw data to be used
for analysis and model training on scheduled interval.
"""

import os
import json
import sys
from datetime import datetime
from airflow.decorators import task, dag
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from snowflake.connector.pandas_tools import write_pandas
from airflow.models.connection import Connection
from task_definitions.etl_task_definitions import scrap_raw_tweets_from_web, preprocess_tweets
from task_definitions.etl_task_definitions import add_sentiment_labels_to_tweets

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.helper import Config, Connections
from utils.helper import load_dataframe


# Load all configurations from config.toml
config = Config()

@dag(dag_id = "etl", start_date = datetime(2023,1,1), schedule_interval = "@monthly", catchup = False)
def twitter_data_pipeline_dag_etl() -> None:
    """
    Data pipeline for performing ETL task that has to be used for training.

    Returns
    -------
            None
    """

    @task(task_id = "configure_connections")
    def set_connections() -> None:
        """
        Task 1 => Configure and establish respective connections for external services like
        AWS S3 buckets and Snowflake data warehouse. The credentials are stored as docker secrets
        in respective containers and accessed as environment variables for secure usage which
        restricts them from getting leaked in the docker image or repository.

        Note:
            AWS credentials are generated using specific IAM users and roles.

        Returns
        -------
            None
        """
        
        # AWS S3 connection
        aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        aws_region_name = os.environ["REGION"]
        s3_credentials = json.dumps(
                                dict(
                                    aws_access_key_id = aws_access_key_id,
                                    aws_secret_access_key = aws_secret_access_key,
                                    aws_region_name = aws_region_name,
                                    )
                                )

        s3_connection = Connection(conn_id = "s3_connection",
                                   conn_type = "S3",
                                   extra = s3_credentials
                                  )
        s3_conn_response = Connections(s3_connection).create_connections()

        # Snowflake connection
        login = os.environ["LOGIN"]
        password = os.environ["PASSWORD"]
        host_name = os.environ["HOST"]

        snowflake_connection = Connection(conn_id = "snowflake_conn",
                                          conn_type = "Snowflake",
                                          host = host_name,
                                          login = login,
                                          password = password
                                        )

        snowflake_conn_response = Connections(snowflake_connection).create_connections()


        if not s3_conn_response and snowflake_conn_response:
            print("Connection not established!!")

    #Instantiating S3 hook for respective tasks
    s3_hook = S3Hook(aws_conn_id = config["aws"]["connection_id"])

    # Task 2 => Refer respective task definition for documentation
    scrap_raw_tweets_from_web_ = PythonOperator(
                                    task_id = "scrap_raw_tweets_from_web",
                                    python_callable = scrap_raw_tweets_from_web,
                                    op_kwargs = {
                                        's3_hook': s3_hook,
                                        'bucket_name': config["aws"]["s3_bucket_name"],
                                        'search_query': config["tweets-scraping"]["search_query"],
                                        'tweet_limit': config["tweets-scraping"]["tweet_limit"],
                                        'raw_file_name': config["files"]["raw_file_name"]
                                        }
                                    )

    @task(task_id = "download_from_s3")
    def download_data_from_s3_bucket(temp_data_path: str, file_name: str) -> None:
        """
        Task 3 => Download data stored in S3 buckets for usage.

        Parameters
        ----------
        temp_data_path: str
            Path to save downloaded file.
        file_name: str
            Name of the downloaded file.

        Returns
        -------
            None
        """

        # Creating a S3 hook using the connection created via task 1.
        downloaded_file = s3_hook.download_file(
                                            key = file_name,
                                            bucket_name = config["aws"]["s3_bucket_name"],
                                            local_path = temp_data_path
                                            )
        os.rename(src = downloaded_file, destination = f"{temp_data_path}/{file_name}")

    with TaskGroup(group_id = "sentiment_labelling") as group1:
        #Task 4 => Refer respective task definition for documentation
        add_sentiment_labels_to_scrapped_tweets_ = PythonOperator(
                                                task_id = "add_sentiment_labels_to_scrapped_tweets",
                                                python_callable = add_sentiment_labels_to_tweets,
                                                op_kwargs = {
                                                    's3_hook': s3_hook,
                                                    'bucket_name': config["aws"]["s3_bucket_name"],
                                                    'temp_data_path': config["aws"]["temp_data_path"],
                                                    'raw_file_name': config["files"]["raw_file_name"],
                                                    'labelled_file_name': config["files"]["labelled_file_name"],
                                                }
                                            )

        # Prioritizing every downstream tasks pertaining to task group 1
        download_data_from_s3_bucket(config["aws"]["temp_data_path"], config["files"]["raw_file_name"]) >> add_sentiment_labels_to_scrapped_tweets_


    with TaskGroup(group_id = "preprocess_tweets_using_NLP") as group2:
        #Task 5 => Refer respective task definition for documentation
        preprocess_tweets_ = PythonOperator(
                                task_id = "preprocess_labelled_tweets_using_nlp_techniques",
                                python_callable = preprocess_tweets,
                                op_kwargs = {
                                    's3_hook': s3_hook,
                                    'bucket_name': config["aws"]["s3_bucket_name"],
                                    'temp_data_path': config["aws"]["temp_data_path"],
                                    'labelled_file_name': config["files"]["labelled_file_name"],
                                    'preprocessed_file_name': config["files"]["preprocessed_file_name"]
                                }
                            )
        
        # Prioritizing every downstream tasks pertaining to task group 2
        download_data_from_s3_bucket(config["aws"]["temp_data_path"], config["files"]["labelled_file_name"]) >> preprocess_tweets_

    @task(task_id = "load_processed_data_to_datawarehouse")
    def load_processed_data_to_snowflake(processed_file: str, table_name: str) -> None:
        """
        Task 6 => Load and write final processed data into snowflake data warehouse. It loads the processed parquet
        file as dataframe and loads it as a database table into the data warehouse.

        Parameters
        ----------
        processed_file: str
            Name of preprocessed parquet file.
        table_name: str
            Name of the database table in snowflake data warehouse.

        Returns
        -------
            None
        """
        try:
            # Similar to S3 hook, snowflake hook is used accordingly
            snowflake_conn = SnowflakeHook(
                                        snowflake_conn_id = "snowflake_conn",
                                        account = os.environ["ACCOUNT"],
                                        warehouse = os.environ["WAREHOUSE"],
                                        database = os.environ["DATABASE"],
                                        schema = os.environ["SCHEMA"],
                                        role = os.environ["ROLE"]
                                        )

            dataframe = load_dataframe(processed_file)

            # Functionality to write any pandas dataframe into snowflake
            write_pandas(
                        conn = snowflake_conn,
                        df = dataframe,
                        table_name = table_name,
                        quote_identifiers = False
                        )
        
        except Exception as exc:
            raise ConnectionError("Something went wrong with the snowflake connection. Please check them!!") from exc

        finally:
            snowflake_conn.close()

    # Prioritizing every downstream tasks pertaining to the entire DAG
    set_connections() >> scrap_raw_tweets_from_web_>> group1 >> group2 >> load_processed_data_to_snowflake(config["files"]["preprocessed_file_name"], config["misc"]["table_name"])


etl_dag = twitter_data_pipeline_dag_etl()