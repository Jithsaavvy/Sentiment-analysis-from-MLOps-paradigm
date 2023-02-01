# Makefile to run Airflow DAG in docker container with external dependencies

include .env

run_dag:
#	Build extended airflow docker image with required pip dependencies
	docker build . -f ./dependencies/Dockerfile --tag extending_airflow:latest
#	Rebuild airflow webserver and scheduler with our newly build image
	docker-compose up -d --no-deps --build airflow-webserver airflow-scheduler

#	Start all required containers to run all airflow services 
	docker-compose -f docker-compose.yaml up -d
	docker ps
	sleep 15

#	Triggering DAG for the first time by accessing the webserver container
	docker exec -it  twitter_bot_airflow-webserver_1 bash -c "airflow dags trigger twitter_data_pipeline_dag_etl

stop_dag:
	docker-compose -f docker-compose.yaml down