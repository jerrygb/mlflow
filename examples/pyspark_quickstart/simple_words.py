"""
Trains an Alternating Least Squares (ALS) model for user/movie ratings.
The input is a Parquet ratings dataset (see etl_data.py), and we output
an mlflow artifact called 'als-model'.
"""
import click

import mlflow
import mlflow.spark

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer

@click.command()
@click.option("--rank", default=12, type=int)
def train_simple_words(rank):

    spark = pyspark.sql.SparkSession.builder.getOrCreate()

    training = spark.createDataFrame([
        (0, "a b c d e spark", 1.0),
        (1, "b d", 0.0),
        (2, "spark f g h", 1.0),
        (3, "hadoop mapreduce", 0.0) ], ["id", "text", "label"])
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=10, regParam=0.001)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
    model = pipeline.fit(training)
    mlflow.log_metric("training_df_count", training.count())
    mlflow.spark.log_model(model, "simple-words")

if __name__ == '__main__':
    train_simple_words()
