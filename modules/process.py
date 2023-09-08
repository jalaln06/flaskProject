import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from modules.utils import (
    extract_data,
    fetch_n_records,
    get_dataframe,
    get_category,
    filter_features,
    similarities_search,
    get_urls
)

BASE_PATH = "./"
FILE_NAME = "arxiv-metadata-oai-snapshot.json"
FILE_PATH = os.path.join(BASE_PATH, FILE_NAME)
CHUNK_SIZE = 1000000
USEFUL_FEATURES = [
    "id",
    "authors",
    "title",
    "categories",
    "abstract",
    "update_date",
    "doi",
    "license",
]
MAX_FEATURES = 10000
CATEGORY = "q-bio"

data_gen = extract_data(FILE_PATH)
data_records = fetch_n_records(data_gen, CHUNK_SIZE)
df = get_dataframe(data_records)

# отбираем только q-bio, оставляем только нужные поля
df = get_category(df, CATEGORY)
df = filter_features(df, USEFUL_FEATURES)

# создаем корпус абстрактов, получаем tf-idf вектора
corpus = df['abstract']
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=MAX_FEATURES)
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)


def data_process(question):
    # ищем ближайшие 5(по умолчанию) по запросу
    ids = np.array([*df.index])
    query_tf = tfidf_vectorizer.transform([question])
    similar_ids = similarities_search(ids, tfidf_matrix.todense(), query_tf.todense(), 5)

    # берем данные только по найденным статьям
    sim_df = df[df.index.isin(similar_ids)]

    # вытаскиваем ссылки
    urls = get_urls(sim_df)
    return urls


if __name__ == "__main__":
    query = 'show me topics about signatures of bayesian inference emerge from energy efficient synapses'
    print(data_process(query))
