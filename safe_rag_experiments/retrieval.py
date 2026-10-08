# retrieval.py
"""
Builds a lightweight retrieval module:
 - BM25 using rank_bm25
 - Embedding index using sentence-transformers + faiss (small demo)
Provides retrieve(query, topk) -> list of (doc_id, snippet, poisoned_flag)
"""

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from typing import List, Tuple

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, local

class Retriever:
    def __init__(self, corpus_df: pd.DataFrame):
        self.df = corpus_df.reset_index(drop=True)
        tokenized = [d.split() for d in self.df["text"].tolist()]
        self.bm25 = BM25Okapi(tokenized)
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.embeddings = self.embedder.encode(self.df["text"].tolist(), show_progress_bar=False)
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.embeddings.astype("float32"))

    def retrieve(self, query: str, topk: int = 5) -> List[dict]:
        # BM25 scores
        q_tokens = query.split()
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_top = np.argsort(bm25_scores)[::-1][:topk*2]

        # embedding similarity
        q_emb = self.embedder.encode([query], show_progress_bar=False).astype("float32")
        D, I = self.index.search(q_emb, topk*2)
        emb_top = I[0]

        # merge top candidates by index order preference
        candidate_idxs = list(dict.fromkeys(list(bm25_top) + list(emb_top)))[:topk]
        results = []
        for idx in candidate_idxs:
            results.append({
                "doc_id": self.df.at[idx, "doc_id"],
                "snippet": self.df.at[idx, "text"][:400],
                "poisoned_flag": bool(self.df.at[idx, "poisoned_flag"]),
                "poison_type": self.df.at[idx, "poison_type"]
            })
        return results
