import findspark

findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel

from utils import normalizeContentDf, tokenizeDf

# Load model
persistedModel = PipelineModel.load("./model_lr")

# Create Spark Session
spark = SparkSession.builder.appName('comment_nlp').getOrCreate()

# Create test data
testData = spark.read.csv('./input/check.csv', inferSchema=True, header=True)

# Tokenize and normalize data
processedData = tokenizeDf(normalizeContentDf(testData))

# Use model
transform_df = persistedModel.transform(processedData).select("content", "prediction")

# Show some result
transform_df.show()
