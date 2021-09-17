import html
from pyspark.sql.functions import col, udf, regexp_replace, trim
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql import DataFrame
from pyvi import ViTokenizer, ViPosTagger

hashtag_regex = r"#(\w{1,})"
url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
not_char_regex = "[^a-zA-ZaAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ]"

@udf
def html_unescape(s: str):
    return html.unescape(s)

def normalizeContentDf(df : DataFrame) -> DataFrame:
    return (
        df
        .withColumn("content", regexp_replace(col("content"), hashtag_regex, "$1"))
        .withColumn("content", regexp_replace(col("content"), url_regex, ""))
        .withColumn("content", regexp_replace(col("content"), email_regex, ""))
        .withColumn("content", html_unescape("content"))
        .withColumn("content", regexp_replace(col("content"), not_char_regex, " "))
        .withColumn("content", regexp_replace(col("content"), " +", " "))
        .withColumn("content", trim(col("content")))
    )

tokenize_text_udf = udf(lambda s: ViPosTagger.postagging(ViTokenizer.tokenize(s))[0], ArrayType(StringType()))
def lower_arr_str(x):
    res = []
    for x_ in x:
        res.append(x_.lower())
    return res
to_lower_str_arr = udf(lower_arr_str, ArrayType(StringType()))

def tokenizeDf(df : DataFrame) -> DataFrame:
    return df.withColumn("content1", to_lower_str_arr(tokenize_text_udf(col("content"))))
