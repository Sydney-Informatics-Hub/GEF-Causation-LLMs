""" Sampler

Sampler tries to downsize a classification textual dataset.

How about this,
group the annotated data by biases,
then within each group,
run a clustering based on the similarity score between sentences,
say we want to 50 sentences per bias, then try to get 20 clusters,
then sample at least one sentence per cluster,
and proportionally increase the number of sentences per cluster,
so you get 50 sentences at the end.

Embeddings:
    + spacy (en_core_web_sm, en_core_web_md, en_core_web_lg)
    + bert encoding (bert-base, distilbert)
    + openai (text-davinci-003)     -> persist these since it's expensive.

Clustering:
    + k-means
    - hierarchical
    - density-based spatial clustering (DBSCAN, HDBScan)
    - gaussian mixture models (GMMs)
    - spectral clustering
"""
import time
from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Callable, Any
from collections import namedtuple
from functools import partial
from collections.abc import Hashable
import spacy

__all__ = ['Sampler']

INDEX = int
CLUSTER_NAME = Hashable
CLUSTER_RESULTS = dict[CLUSTER_NAME, list[INDEX]]

EMBEDDINGS = np.ndarray

Clustering_Strategies = ('kmeans',)
Embedding_Types = ('spacy_en_core_web_sm',)  # 'bert'


# Cluster for one class
def kmeans_cluster(embs: np.ndarray, n_clusters: int, seed: int = 42, **kwargs) -> CLUSTER_RESULTS:
    """ Cluster the embeddings using kmeans and return a dictionary of cluster -> list of indices."""
    if not isinstance(embs, EMBEDDINGS): raise TypeError("embs must be np.ndarray.")
    if not isinstance(n_clusters, int): raise TypeError("n_clusters must be an integer.")
    if not n_clusters > 0: raise ValueError("n_clusters must be > 0.")
    if not isinstance(seed, int): raise TypeError("seed must be an integer.")

    kmeans = KMeans(n_clusters=3, random_state=seed, **kwargs)

    from sklearn import preprocessing
    embs = preprocessing.normalize(embs)  # normalise for cosine similarity

    kmeans.fit(embs)

    series = pd.Series(kmeans.labels_)

    results = {}
    for gid, gseries in series.groupby(by=series):
        cluster_name: CLUSTER_NAME = gid
        indices: list[INDEX] = gseries.index.to_list()
        results[cluster_name] = indices
    return results


def embed_spacy_en_core_web_sm(texts: list[str]) -> EMBEDDINGS:
    nlp = spacy.load('en_core_web_sm')
    embeddings = np.array(list(map(lambda doc: doc.vector, nlp.pipe(texts))))
    assert len(texts) == embeddings.shape[0], "Mismatched number of texts and embeddings."
    return embeddings


ARTIFACTS = namedtuple('ARTIFACT', ['dataframe', 'embeddings', 'clusters'])


class Sampler(object):
    Clustering_Strategies = Clustering_Strategies
    Embedding_Types = Embedding_Types

    def __init__(self, clustering_strategy: str, embedding_type: str):
        if not clustering_strategy in Clustering_Strategies:
            raise ValueError(f"clustering_strategy must be one of {', '.join(Clustering_Strategies)}")
        if not embedding_type in Embedding_Types:
            raise ValueError(f"embedding_type must be one of {', '.join(Embedding_Types)}")

        self.clustering_strategy = clustering_strategy
        self.embedding_type = embedding_type

        self.artifacts: ARTIFACTS = None

    def initialise(self, df: pd.DataFrame, col_clazz: str, col_text: str, verbose: bool = False) -> ARTIFACTS:
        """ Instantiates the Sampler with your text dataset by running clustering per class. """
        if col_clazz not in df.columns: raise ValueError(f"{col_clazz=} is not one of the columns")
        if col_text not in df.columns: raise ValueError(f"{col_text=} is not one of the columns")

        if verbose:
            print(f"Clustering strategy: {self.clustering_strategy}")
            print(f"Embedding type: {self.embedding_type}")

        emb_fn: Callable[[list[str]], EMBEDDINGS] = embed_spacy_en_core_web_sm
        cluster_fn: Callable[[EMBEDDINGS, Any], CLUSTER_RESULTS] = kmeans_cluster

        clazzes = list(df.loc[:, col_clazz].unique())
        if verbose:
            print(f"Classes: {', '.join(clazzes)}")

        embeddings: EMBEDDINGS = emb_fn(df.loc[:, col_text].tolist())

        # Generate clusters
        clazz_cresults = dict()
        if verbose: print("Generating clusters...")
        # noinspection PyTypeChecker
        for clazz, group in tqdm(df.loc[:, [col_clazz, col_text]].groupby(by=col_clazz),
                                 colour='orange',
                                 total=len(clazzes)):
            cresults: CLUSTER_RESULTS = cluster_fn(embeddings, n_clusters=5)  # todo: infer cluster size
            clazz_cresults[clazz] = cresults

        self.artifacts = ARTIFACTS(dataframe=df, embeddings=embeddings, clusters=clazz_cresults)
        if verbose: print("Done.")
        return ARTIFACTS(dataframe=df, embeddings=embeddings, clusters=clazz_cresults)  # separate reference

    def sample(self, n: int) -> pd.DataFrame:
        if not isinstance(n, int): raise TypeError("n must be an integer.")
        if not n > 0: raise ValueError("n must be > 0.")
        assert self.artifacts is not None, "Please first call the initialise() method to initialise the sampler."

        # obtain class weights for representative sample
        clazzes = list(self.artifacts.clusters.keys())
        clazz_weights = np.zeros(shape=(len(clazzes),))
        for i, clazz in enumerate(clazzes):
            weight = 0
            for indices in self.artifacts.clusters.get(clazz).values():
                weight += sum(indices)
            clazz_weights[i] = weight

        sample_pool_size = sum(clazz_weights)
        if not n < sample_pool_size:
            raise ValueError(f"n must be smaller than sampling pool. Pool size: {sample_pool_size}")

        # convert to probability weights
        clazz_weights = clazz_weights / sum(clazz_weights)
        assert sum(clazz_weights), "Class weights do not sum to 1. Nuh way... This is a bug. Sorry! hahaha contact me."

        # sample a set of indices per class.
        sample_indices = list()
        for i in range(n):
            clazz = np.random.choice(clazzes, p=clazz_weights)
            clusters = self.artifacts.clusters.get(clazz)
            # uniformly sample from one of the clusters.
            cluster_name = np.random.choice(len(clusters.keys()))
            indices = clusters.get(cluster_name)
            idx = indices.pop(np.random.choice(len(indices)))
            sample_indices.append(idx)

        # align sampled indices with dataframe's index.
        remapped_indices = [self.artifacts.dataframe.iloc[i].name for i in sample_indices]
        return self.artifacts.dataframe.loc[remapped_indices]


if __name__ == '__main__':
    sampler = Sampler(clustering_strategy='kmeans', embedding_type='spacy_en_core_web_sm')

    clazzes = [0, 1, 1, 1, 0, 0, 1, 0, 1, 1]
    texts = [
        "The cat is sleeping on the mat.",
        "I enjoy listening to music.",
        "She is running in the park.",
        "The book is on the table.",
        "He likes to play basketball.",
        "We went to the beach yesterday.",
        "They are cooking dinner together.",
        "The sun is shining brightly.",
        "I can't wait for the weekend.",
        "She won the first prize in the competition."
    ]
    df = pd.DataFrame(zip(texts, clazzes), columns=['text', 'clazz'])
    sampler.initialise(df, col_text='text', col_clazz='clazz')

    samples = sampler.sample(3)
    print(samples)
