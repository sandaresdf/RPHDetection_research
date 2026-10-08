# experiment.py
"""
Top-level experiment runner:
 - Prepares corpus (clean + poisons)
 - Builds retriever
 - Runs agent on each query
 - Runs detector + judge
 - Logs results to CSV for later analysis
"""

import pandas as pd
import json
import os
from tqdm import tqdm
from data_prep import load_sample_corpus, inject_poisoned_variants, prepare_queries
from retrieval import Retriever
from agent import build_prompt, mocked_generate_cot_and_answer
from detector import SimpleDetector, extract_features
from judge import mocked_entailment_check, judge_policy
import numpy as np

LOG_PATH = "experiment_logs.csv"

def run_experiment(poison_ratio=0.05, poison_type="cot_step", insertion_pos="end"):
    # 1. data
    df_clean = load_sample_corpus()
    df_corpus = inject_poisoned_variants(df_clean, ratio=poison_ratio, poison_type=poison_type, strength=0.8, insertion_pos=insertion_pos)

    # 2. retriever
    retriever = Retriever(df_corpus)

    # 3. prepare queries
    queries = prepare_queries()

    # 4. detector: train a toy detector on synthetic examples (for demo)
    # Create training X/y: here we make simple synthetic features to let the detector function
    det = SimpleDetector()
    X_train = np.array([
        [0, 50, 16.0, 0.0],
        [1, 80, 12.0, 0.5],
        [1, 90, 10.0, 0.8],
        [0, 40, 13.0, 0.0]
    ])
    y_train = np.array([0, 1, 1, 0])
    det.train(X_train, y_train)

    # 5. run queries
    rows = []
    for qid, query_text, gt in tqdm(queries):
        retrieved = retriever.retrieve(query_text, topk=5)
        prompt = build_prompt(query_text, retrieved)
        agent_out = mocked_generate_cot_and_answer(prompt, retrieved)
        cot = agent_out["cot"]
        answer = agent_out["answer"]
        feat = extract_features(cot, retrieved)
        detector_score = det.predict_proba(feat)
        entailment_score = mocked_entailment_check(cot, retrieved)
        decision = judge_policy(cot, retrieved, detector_score, entailment_score)

        rows.append({
            "qid": qid,
            "query": query_text,
            "retrieved_docs": json.dumps(retrieved),
            "cot": json.dumps(cot),
            "final_answer": answer,
            "detector_score": detector_score,
            "entailment_score": entailment_score,
            "judge_decision": decision,
            "poison_ratio": poison_ratio,
            "poison_type": poison_type
        })

    # 6. save logs
    df_log = pd.DataFrame(rows)
    df_log.to_csv(LOG_PATH, index=False)
    print(f"Saved logs to {LOG_PATH}")
    return df_log

if __name__ == "__main__":
    df_log = run_experiment(poison_ratio=0.05, poison_type="cot_step", insertion_pos="end")
    print(df_log.to_dict(orient="records"))
