import findspark

findspark.init()

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import IDF, HashingTF, StopWordsRemover

from utils import normalizeContentDf, tokenizeDf

# Create Spark Session
spark = SparkSession.builder.appName('comment_nlp').config("spark.driver.memory", "6g").getOrCreate()

# Read raw data
rawData = spark.read.csv('./input/comment_input.csv', inferSchema=True, header=True)

# Split raw data
SPLITTED_PATH = './splitted'
rawData.repartition(20).write.partitionBy("status").csv(SPLITTED_PATH, mode="overwrite")

# Read splitted data
schemaDdl = "content STRING, status INTEGER"
sparkReader = spark.read.schema(schemaDdl)
splittedData = sparkReader.csv(SPLITTED_PATH)

# Normalize content
processedData = normalizeContentDf(splittedData).filter("content != ''").select("content", "status").coalesce(3)

# Tokenize Vietnamese before process
processedData = tokenizeDf(processedData)

# Split data to evaluation
(trainingData, validationData, testData) = processedData.randomSplit([0.90, 0.05, 0.05], seed=2021)

# Read Vietnamese Stopwords
stopwords_file = open("./vietnamese-stopwords.txt", "r")
stopwords_list = stopwords_file.read().split('\n')

# Define pipeline
stopwords_remover = StopWordsRemover(
    inputCol="content1",
    outputCol="content2",
    stopWords=stopwords_list,
)
hashing_tf = HashingTF(
    inputCol="content2",
    outputCol="term_frequency",
)
idf = IDF(
    inputCol="term_frequency",
    outputCol="features",
    minDocFreq=5
)
lr = LogisticRegression(labelCol="status")
sematic_analysis_pipeline = Pipeline(
    stages=[stopwords_remover, hashing_tf, idf, lr]
)

# Create model
model = sematic_analysis_pipeline.fit(trainingData)

# Evaluate model
trained_df = model.transform(trainingData)
val_df = model.transform(validationData)
test_df = model.transform(testData)

trained_df.show()
val_df.show()
test_df.show()

evaluator = MulticlassClassificationEvaluator(labelCol="status", metricName="accuracy")
accuracy_val = evaluator.evaluate(val_df)
accuracy_test = evaluator.evaluate(test_df)

print(f"Validation accuracy: {accuracy_val*100:.5f}%")
print(f"Test accuracy: {accuracy_test*100:.5f}%")

# Save model
model.write().overwrite().save('./model/')
