import random
import numpy as np

from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from reranking.config import *

def get_similar_questions(all_questions):
    """Cluster questions by semantic similarity and return lists of similar question indices with varying similarity levels"""
    model = SentenceTransformer(SENTENCE_TRANSFORMER_PRETRAINED_MODEL_MINILM)

    q_embeddings = model.encode(all_questions, convert_to_numpy=True ,show_progress_bar=True) #?? as numpy or tensor
    clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
    clustering.fit(q_embeddings)
    labels = clustering.labels_

    distance_matrix = cosine_distances(q_embeddings)

    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        if label != -1: #-1 is noise
            clusters[label].append(i)

    results = []
    for i, q_embedding in enumerate(q_embeddings):
        label = labels[i]
        distances = distance_matrix[i]

        # --- Similar / Same Cluster --- #
        same_cluster = [idx for idx in clusters.get(label, []) if idx != i]
        similar = random.sample(same_cluster, min(AMOUNT_SAME_SIMILAR, len(same_cluster)))

        sorted_indices = np.argsort(distances)
        # if not enough questions in cluster:
        if len(similar) < AMOUNT_SAME_SIMILAR:
            # get next nearest

            for ni in sorted_indices:
                if ni != i and ni not in similar:
                    similar.append(ni)
                    if len(similar) == AMOUNT_SAME_SIMILAR:
                        break

        n = len(sorted_indices)

        # --- Medium Far - 40–60 percentile of distance ---
        medium_far = []
        for k in range(AMOUNT_MID_SIMILAR):
            mid_range = sorted_indices[int(n * 0.4):int(n * 0.6)]
            medium_far.append(random.choice([j for j in mid_range if j != i]))

        # --- Super Far - 90 percentile of distance ---
        super_far = []
        for k in range(AMOUNT_FAR_SIMILAR):
            super_far_candidates = sorted_indices[int(n * 0.9):]
            super_far.append(random.choice([j for j in super_far_candidates if j != i]))

        results.append([*similar, *medium_far, *super_far])
    return results
