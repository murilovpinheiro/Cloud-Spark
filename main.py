from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, substring, hour, to_date, col, when, udf, explode, size, rank
from pyspark.sql.types import ArrayType, StringType, IntegerType
from unicodedata import normalize
from pyspark.ml.clustering import LDA
import re

print("teste")

# Inicialize a sessão do Spark
spark = SparkSession.builder \
    .appName("Leitura de Arquivo TSV") \
    .config("spark.sql.debug.maxToStringFields", 1000)\
    .getOrCreate()

# Caminho para o arquivo TSV
file_path = "./data/debate-tweets.tsv"


# Ler o arquivo TSV como um DataFrame do Spark
df = spark.read.option("header", "false").option("delimiter", "\t").csv(file_path)

# Para começar eu vou selecionar apenas as colunas que eu tenho interesse
# Como as operações desejadas são:
# 1 - Quais foram as hashtags mais usadas pela manhã, tarde e noite
# 2 - Quais as hashtags mais usadas em cada dia
# 3 - Qual o número de tweets por hora a cada dia
# 4 - Quais as principais sentenças relacionadas à palavra “Dilma”
# 5 - Quais as principais sentenças relacionadas à palavra “Aécio”

# É fácil observar que só vamos precisar de um pequeno conjunto de colunas,
# sendo elas "hora", "data" e "conteúdo", respectivamente _c7, _c8 e _c1
# De começo acho tranquilo observar que as 3 primeiras atividades são mais simples
# Diferentemente da Atividade 4 e 5 que parecem um pouco mais complicadas

df = df.select("_c0", "_c1", "_c7", "_c8")
df = df.withColumnRenamed("_c0", "id") \
       .withColumnRenamed("_c1", "content") \
       .withColumnRenamed("_c7", "day_and_hour_string") \
       .withColumnRenamed("_c8", "date")

unique_values = df.select("date").distinct()
unique_values.show(truncate=False)

# tem 15 a 20 de outubro

# Mon May 09 00:12:02 +0000 2011
# Como não consegui converter diretamente para timestamp vou pegar por substring e dps converter
"""
df = df.withColumn("hour_of_day_string", substring(df["day_and_hour_string"], 12, 8))

df = df.withColumn("timestamp_col", to_timestamp(df["hour_of_day_string"], "HH:mm:ss"))
# Extraia apenas a hora
df = df.withColumn("hour_of_day", hour(df["timestamp_col"]))

df = df.withColumn("date_", to_date(df["date"], "yyyy-mm-dd"))

df = df.select("id", "content", "date_", "hour_of_day")

df = df.withColumn("period", when((df.hour_of_day >= 5) & (df.hour_of_day < 12), "morning").otherwise(
                             when((df.hour_of_day >= 12) & (df.hour_of_day < 18), "afternoon").otherwise("night")))

df_filtred = df.select("date_", "hour_of_day")

df_filtred = df_filtred.groupBy("date_", "hour_of_day").count()

df_filtred.show()
"""

def extract_name_AECIO(text):
    text_sem_acento = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    if re.findall(r'\bA[eé]cio\b', text_sem_acento, re.IGNORECASE):
        return 1
    else:
        return 0

def extract_name_DILMA(text):
    if re.search(r'\bDilma\b', text, re.IGNORECASE):
        return 1
    else:
        return 0

# Registrar a função UDF
extract_dilma = udf(extract_name_AECIO, IntegerType())

# Adicionar a coluna "temDilma" ao DataFrame com valores 1 ou 0
df = df.withColumn("temDilma", extract_dilma(df["content"]))

# Filtrar linhas com a palavra "Dilma"
df_com_dilma = df.filter(col("temDilma") == 1)

dataset = df.select("content")
dataset.show()
# Loads data.
# precisa fazer um vectorizer

# Trains a LDA model.
lda = LDA(k=10, maxIter=10)
model = lda.fit(dataset)

ll = model.logLikelihood(dataset)
lp = model.logPerplexity(dataset)
print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
print("The upper bound on perplexity: " + str(lp))

# Describe topics.
topics = model.describeTopics(3)
print("The topics described by their top-weighted terms:")
topics.show(truncate=False)

# Shows the result
transformed = model.transform(dataset)
transformed.show(truncate=False)

# Mostrar DataFrame resultante
pandas_df = dataset.toPandas()

pandas_df.to_csv("./teste.csv", header=True, index=False)

"""
def extract_hashtags(text):
    return re.findall(r'#(\w+)', text)

extract_hashtags_udf = udf(extract_hashtags, ArrayType(StringType()))

# count = df.groupBy("period").count()

df = df.withColumn("hashtags", extract_hashtags_udf(df["content"]))

df = df.filter(size(col("hashtags")) > 0)

df = df.select("hashtags", "date_")

df = df.withColumn("hashtag", explode("hashtags"))

df = df.groupBy("date_", "hashtag").count()

df.show()

window_spec = Window.partitionBy("date_").orderBy(col("count").desc())

# Adicionar uma coluna de classificação com base na contagem das hashtags
df_ranked = df.withColumn("rank", rank().over(window_spec))

# Selecionar as maiores hashtags de cada dia (classificação igual a 1)
top_hashtags_by_day = df_ranked.filter(col("rank") == 1)

top_hashtags_by_day.show()

"""

"""
hashtags_periods = df.select("hashtags", "period")

df_filtered = hashtags_periods.filter(size(col("hashtags")) > 0)

df_exploded = df_filtered.withColumn("hashtag", explode("hashtags"))

hashtags_by_period = df_exploded.groupBy("period", "hashtag").count()

hashtags_by_period = hashtags_by_period.orderBy(col("count").desc())

hashtags_by_period.show()

pandas_df = hashtags_by_period.toPandas()

pandas_df.to_csv("./teste.csv", header=True, index=False)

# OK
"""

# df.show()


# Parar a sessão do Spark
spark.stop()
