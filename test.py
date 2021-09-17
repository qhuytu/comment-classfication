import findspark

findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import PipelineModel

from utils import normalizeContentDf, tokenizeDf

# Load model
persistedModel = PipelineModel.load("./model")

# Create Spark Session
spark = SparkSession.builder.appName('comment_nlp').getOrCreate()

# Create test data
columns = ["content", "abcd"]
data = [
    ("https://google.com.vn các bạn nhân viên ơi hỗ trợ mình với", 0),
    ("#helloworld các bạn ơi", 0),
    ("@#$! cho tôi thấy cánh tay của các bạn đi", 0),
    ("   Anh em đâu hết rồi", 0),
    ("  Bài hát này hay quá", 0),
    ("Cái thằng chó này", 0)
]
testData = spark.createDataFrame(data=data, schema = columns)

# Tokenize and normalize data
processedData = tokenizeDf(normalizeContentDf(testData))

# Use model
transform_df = persistedModel.transform(processedData);

# Show some result
transform_df.show()
