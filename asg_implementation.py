import os
import json
import time
import zlib
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics import roc_auc_score, brier_score_loss, auc
from scipy.stats import bootstrap, entropy
from scipy.spatial.distance import pdist

def predict_factuality(responses: List[str]) -> float:
    """
    Algorithmic Spectral Gap (ASG)
    Target Architecture: Dense RLHF
    """
    n = len(responses)
    if n <= 1: return 1.0

    # 1. Pairwise Algorithmic Distance (NCD)
    W = np.zeros((n, n))
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            b1, b2 = responses[i].encode('utf-8'), responses[j].encode('utf-8')
            c1, c2 = len(zlib.compress(b1)), len(zlib.compress(b2))
            c12 = len(zlib.compress(b1 + b2))
            ncd = (c12 - min(c1, c2)) / max(c1, c2)
            dists.append(ncd)

    # Use mean distance for local scaling
    avg_d = np.mean(dists) if dists else 1.0
    idx = 0
    for i in range(n):
        W[i, i] = 1.0
        for j in range(i + 1, n):
            sim = np.exp(-dists[idx] / (avg_d + 1e-9))
            W[i, j] = W[j, i] = sim
            idx += 1

    # 2. Spectral Gap Analysis
    try:
        s_eigs = np.linalg.eigvalsh(W)
        s_eigs = np.sort(s_eigs)[::-1]

        # Spectral Gap: Difference between top two eigenvalues
        # Normalized by n to keep it in [0, 1]
        # In a perfect consensus (all ones matrix), eigs are [n, 0, ..., 0], Gap = n
        # In a random set (Identity), eigs are [1, ..., 1], Gap = 0
        gap = (s_eigs[0] - s_eigs[1]) / n
        return float(np.clip(gap, 0.0, 1.0))
    except:
        return 0.5

def main():
    parquet_path = "combine/embeddings/combined_data.parquet"
    df = pd.read_parquet(parquet_path)
    models = ['llama-3.3-70b-versatile', 'mistral_7b']
    for model_name in models:
        print(f"\n--- Testing ASG on Model: {model_name} ---")
        model_df = df[df['model_name'] == model_name].copy()
        model_df['target'] = model_df['difficulty_type'].map({'Factual': 1, 'Adversarial': 0})
        eval_df = model_df.dropna(subset=['target'])
        probs = []
        for idx, row in eval_df.iterrows():
            res_text = row['response_text']
            if isinstance(res_text, str):
                try: responses = json.loads(res_text)
                except: responses = [res_text]
            else: responses = res_text
            responses = [str(r) for r in responses if r is not None][:10]
            probs.append(predict_factuality(responses))
        metrics = evaluate_metrics(probs, eval_df['target'].values)
        print(f"AUROC: {metrics['auroc']:.4f}")
        print(f"ECE: {metrics['ece']:.4f}")

def evaluate_metrics(probs, labels):
    probs = np.array(probs)
    labels = np.array(labels)
    if np.max(probs) > np.min(probs):
        probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs) + 1e-9)
    metrics = {}
    if len(np.unique(labels)) < 2:
        metrics.update({'auroc': 0.5})
    else:
        metrics['auroc'] = roc_auc_score(labels, probs)
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0
    for i in range(10):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i+1])
        if i == 9: mask = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
        if np.any(mask):
            ece += np.abs(np.mean(labels[mask]) - np.mean(probs[mask])) * np.sum(mask) / len(probs)
    metrics['ece'] = ece
    return metrics

if __name__ == "__main__":
    main()
