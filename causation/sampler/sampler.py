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
import sys
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from typing import Callable, Any, Optional, Union
from collections import namedtuple
from copy import deepcopy
from collections.abc import Hashable
import spacy
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import umap
import plotly.graph_objects as go
import plotly.subplots as sp
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

__all__ = ['Sampler']

INDEX = int
CLUSTER_NAME = Hashable
CLUSTER_RESULTS = dict[CLUSTER_NAME, list[INDEX]]

EMBEDDINGS = np.ndarray


# Cluster for one class
def kmeans_cluster(embs: np.ndarray, n_clusters: int, seed: int = 42, **kwargs) -> CLUSTER_RESULTS:
    """ Cluster the embeddings using kmeans and return a dictionary of cluster -> list of indices."""
    if not isinstance(embs, EMBEDDINGS): raise TypeError("req: embs must be np.ndarray.")
    if not isinstance(n_clusters, int): raise TypeError("req: n_clusters must be an integer.")
    if not n_clusters > 0: raise ValueError("req: n_clusters must be > 0.")
    if not isinstance(seed, int): raise TypeError("opt: seed must be an integer.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, **kwargs)

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


def embed_spacy_en_core_web_lg(texts: list[str]) -> EMBEDDINGS:
    nlp = spacy.load('en_core_web_lg')
    embeddings = np.array(list(map(lambda doc: doc.vector, nlp.pipe(texts))))
    assert len(texts) == embeddings.shape[0], "Mismatched number of texts and embeddings."
    return embeddings


def embed_hf(model_name: str) -> Callable[[list[str]], EMBEDDINGS]:
    model = AutoModel.from_pretrained(model_name)
    tokeniser = AutoTokenizer.from_pretrained(model_name)

    def embed(texts: list[str]) -> EMBEDDINGS:
        embeddings = list()
        for i in tqdm(range(len(texts)), total=len(texts)):
            outputs = model(tokeniser.encode(texts[i], add_special_tokens=True, return_tensors='pt'))
            embedding = outputs.last_hidden_state
            embedding = embedding.sum(axis=1).detach().numpy() / embedding.shape[1]
            embeddings.append(embedding.squeeze(0))
        return np.array(embeddings)

    return embed


ARTIFACTS = namedtuple('ARTIFACT', ['dataframe', 'col_clazz', 'col_text', 'embeddings'])

Clustering_Strategies = {
    'kmeans': kmeans_cluster,
}
Embedding_Types = {
    'spacy_en_core_web_sm': embed_spacy_en_core_web_sm,
    'spacy_en_core_web_lg': embed_spacy_en_core_web_lg,
    'bert-base-uncased': embed_hf('bert-base-uncased')
}


class Sampler(object):
    Clustering_Strategies = Clustering_Strategies
    Embedding_Types = Embedding_Types

    def __init__(self, embedding_type: str, clustering_strategy: str, n_clusters: Optional[int] = None):
        if not embedding_type in Embedding_Types.keys():
            raise ValueError(f"embedding_type must be one of {', '.join(Embedding_Types)}")
        if not clustering_strategy in Clustering_Strategies.keys():
            raise ValueError(f"clustering_strategy must be one of {', '.join(Clustering_Strategies)}")
        if n_clusters is not None and not isinstance(n_clusters, int):
            raise TypeError("n_clusters must be an integer.")

        self.clustering_strategy = clustering_strategy
        self.n_clusters = n_clusters
        self.embedding_type = embedding_type
        self._col_cluster = 'cluster'

        self.artifacts: Optional[ARTIFACTS] = None

    def initialise(self, df: pd.DataFrame, col_clazz: str, col_text: str, verbose: bool = False) -> ARTIFACTS:
        """ Instantiates the Sampler with your text dataset by running clustering per class. """
        if col_clazz not in df.columns: raise ValueError(f"{col_clazz=} is not one of the columns.")
        if col_text not in df.columns: raise ValueError(f"{col_text=} is not one of the columns.")

        if verbose:
            print(f"Clustering strategy: {self.clustering_strategy}")
            print(f"Number of clusters: {self.n_clusters}")
            print(f"Embedding type: {self.embedding_type}")

        emb_fn: Callable[[list[str]], EMBEDDINGS] = \
            self.Embedding_Types.get(self.embedding_type, None)
        assert emb_fn is not None, f"No embedding function associated with {self.embedding_type}."
        cluster_fn: Callable[[EMBEDDINGS, Any], CLUSTER_RESULTS] = \
            self.Clustering_Strategies.get(self.clustering_strategy, None)
        assert cluster_fn is not None, f"No clustering function associated with {self.clustering_strategy}."

        clazzes = list(df.loc[:, col_clazz].unique())
        if verbose: print(f"Classes: {', '.join(clazzes)}")

        if verbose: print("Generating embeddings...", end='')
        embeddings: EMBEDDINGS = emb_fn(df.loc[:, col_text].tolist())
        if verbose: print("Done.", end='\n')

        # Generate clusters
        if verbose: print("Generating clusters...", end='')
        # noinspection PyTypeChecker
        clusters: CLUSTER_RESULTS = cluster_fn(embeddings, n_clusters=self.n_clusters)
        if verbose: print("Done.", end='\n')

        remapped_clusters: list[tuple[int, Hashable]] = [(df.iloc[idx].name, cluster)
                                                         for cluster, indices in clusters.items()
                                                         for idx in indices]

        cdf = pd.DataFrame(remapped_clusters, columns=['idx', self._col_cluster]).set_index(keys='idx')
        df = pd.concat([df, cdf], axis=1)

        # note: return class information, number of clusters, indices in each cluster, embeddings, dataframe.
        self.artifacts = ARTIFACTS(dataframe=df, embeddings=embeddings,
                                   col_clazz=col_clazz, col_text=col_text)
        return deepcopy(self.artifacts)

    def sample(self, n: int, clazz_weights: Optional[list[Union[int, float]]] = None,
               with_replacement: bool = False,
               verbose: bool = False) -> pd.DataFrame:
        """ Sample from the dataset and return a subset of the dataset as a dataframe."""
        if not isinstance(n, int): raise TypeError("n must be an integer.")
        if n <= 0: raise ValueError("n must be > 0.")
        assert self.artifacts is not None, "Please first call the initialise() method to initialise the sampler."
        if n > len(self.artifacts.dataframe):
            raise ValueError(f"n must be <= {len(self.artifacts.dataframe)} (size of dataset)")

        # obtain class weights for representative sample
        clazz_dist = self.artifacts.dataframe.loc[:, self.artifacts.col_clazz].value_counts()
        clazzes = clazz_dist.index.tolist()
        if verbose: print(f"Classes: {', '.join(clazzes)}")

        if clazz_weights is None:
            # retain class distribution of the dataset.
            clazz_weights = clazz_dist.values
        else:
            if not isinstance(clazz_weights, list): raise TypeError("clazz_weights must be a list.")
            if not len(clazz_weights) > 0: raise ValueError("clazz_weights must not be empty.")
            if len(clazz_weights) != len(clazzes):
                raise ValueError("Mismatched number of classes in clazz_weights than in dataset.")

        # convert to probability weights
        clazz_weights = np.array(clazz_weights) / sum(clazz_weights)
        assert sum(clazz_weights) == 1.0, \
            "Class weights do not sum to 1. Nuh way... This is a bug. Sorry! hahaha contact me."
        if verbose: print(f"Normalised class weights: {clazz_weights}")

        # calculate number of examples per class
        num_examples_per_clazz: dict[str, int] = {
            c: clazz_weights[i] * n for (i, c) in enumerate(clazzes)
        }
        assert sum(num_examples_per_clazz.values()) == n, \
            "Number of target examples to accumulate per class should equal sample size. "
        num_examples_per_clazz = dict(zip(
            num_examples_per_clazz.keys(),
            map(lambda n: int(n), num_examples_per_clazz.values()))
        )
        if verbose: print(f"Target number of examples per class:\n{num_examples_per_clazz}")

        clazz_indices = {clazz: set() for clazz in clazzes}
        df = pd.DataFrame(self.artifacts.dataframe, index=self.artifacts.dataframe.index)
        for clazz, group in df.groupby(by=self.artifacts.col_clazz):
            groups = list(group.groupby(by='cluster'))
            num_sampled = 0
            target_num_sampled = num_examples_per_clazz.get(clazz)
            while num_sampled < target_num_sampled:
                exhausted_cluster_counter = 0
                for cluster, group in groups:
                    if num_sampled >= num_examples_per_clazz.get(clazz):
                        continue
                    sampled_indices = clazz_indices.get(clazz)
                    indices = group.index.to_list()
                    if not with_replacement:
                        while True:
                            if len(indices) <= 0:
                                exhausted_cluster_counter += 1
                                break
                            idx = indices.pop(np.random.choice(len(indices)))
                            if idx not in sampled_indices:
                                sampled_indices.add(idx)
                                break
                    else:
                        idx = np.random.choice(indices)
                        sampled_indices.add(idx)

                    num_sampled = len(sampled_indices)
                if exhausted_cluster_counter >= len(groups):
                    print(f"Exhausted all clusters when sampling for class: {clazz}. "
                          f"Capped at {num_sampled}/{target_num_sampled} samples.",
                          file=sys.stderr)
                    break
        sampled_indices = [idx for indices_for_clazz in clazz_indices.values() for idx in indices_for_clazz]
        return self.artifacts.dataframe.loc[sampled_indices]

    def visualise(self, kind: str = 'umap'):
        assert self.artifacts is not None, "Please first call the initialise() method to initialise the sampler."
        if kind.upper() == 'UMAP':
            compressed_2d = self.umap(self.artifacts.embeddings)
        elif kind.upper() == 'TSNE':
            compressed_2d = self.tsne(self.artifacts.embeddings)
        else: raise NotImplementedError(f"kind = {kind} is not supported. Only umap, tsne are.")
        # set up visualisation plot
        cluster_labels = self.artifacts.dataframe.loc[:, self._col_cluster].to_list()
        labels = self.artifacts.dataframe.loc[:, self.artifacts.col_clazz].to_list()
        clazz_labels = LabelEncoder().fit_transform(labels)

        texts = self.artifacts.dataframe.loc[:, [self.artifacts.col_clazz, self.artifacts.col_text]].apply(
            lambda row: f"[{row.clazz}] {row.sentence}", axis=1
        )

        scatter_clazz = go.Scatter(
            x=compressed_2d[:, 0],
            y=compressed_2d[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=clazz_labels,  # assign color to each label
                colorscale='Viridis',  # choose a colorscale
                opacity=0.8
            ),
            text=texts,
            name='class'
        )
        scatter_cluster = go.Scatter(
            x=compressed_2d[:, 0],
            y=compressed_2d[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=cluster_labels,  # assign color to each label
                colorscale='solar',  # choose a colorscale
                opacity=0.8
            ),
            text=texts,
            name='clusters',
        )
        fig = sp.make_subplots(rows=1, cols=2)
        # fig = go.Figure(data=[scatter_class, scatter_cluster])
        fig.add_trace(scatter_clazz, row=1, col=1)
        fig.add_trace(scatter_cluster, row=1, col=2)
        fig.update_layout(autosize=False, width=1600, height=800)
        fig.show()

    @staticmethod
    def umap(embeddings):
        reducer = umap.UMAP(random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        return embeddings_2d

    @staticmethod
    def tsne(embeddings):
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        return embeddings_2d


if __name__ == '__main__':
    # sampler = Sampler(clustering_strategy='kmeans', n_clusters=5, embedding_type='spacy_en_core_web_sm')
    sampler = Sampler(clustering_strategy='kmeans', n_clusters=5, embedding_type='bert-base-uncased')

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
