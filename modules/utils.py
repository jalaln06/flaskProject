import numpy as np
import pandas as pd
import json
from itertools import islice
from typing import Tuple, List
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import arxiv


def extract_data(datapath):
    with open(datapath, "r") as datafile:
        for line in datafile:
            yield line


def fetch_n_records(data_gen, chunksize=500):
    return [json.loads(record) for record in islice(data_gen, chunksize)]


def get_dataframe(list_of_dicts, columns=None):
    data = pd.DataFrame(list_of_dicts)
    if columns:
        data.columns = columns
    return data


def get_category(df, category):
    df[f"{category}"] = df.categories.apply(
        lambda x: 1 if x.startswith(f"{category}") else 0
    )
    df = df[df[f"{category}"] == 1].reset_index()
    return df


def filter_features(data, features):
    return data[features]


def similarities_search(db_ids, db_vectors, query, k_nearest=3):
    dim = db_vectors.shape[1]
    assert dim == query.shape[1]
    index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(index)
    index.add_with_ids(db_vectors, db_ids)
    similarities, similarities_ids = index.search(query, k=k_nearest)
    return similarities_ids[0].tolist()


def get_urls(df):
    papers = arxiv.Search(id_list=df.id.values)

    return [{"title": res.title, "entry_id": res.pdf_url} for res in papers.get()]
