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
from sklearn.metrics.pairwise import cosine_similarity

# Setup Seeds for Reproducibility
np.random.seed(42)

class UQCalculatorFast:
    def __init__(self):
        print("Initialized Fast UQ Calculator (Embedding-based)")

    def get_cosine_entail_matrix(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        # Use cosine similarity as a proxy for entailment
        # embeddings shape (10, 384)
        sim_matrix = cosine_similarity(embeddings)
        # We'll use the raw similarity as entailment score
        # For contradiction, we don't have a good proxy from embeddings alone,
        # but we can use 1 - sim_matrix as a rough proxy for distance/contradiction
        return {"entail": sim_matrix, "contra": 1.0 - sim_matrix}

    def get_semantic_classes(self, entail_matrix, threshold=0.8):
        n = entail_matrix.shape[0]
        sample_to_class = {}
        class_to_sample = []
        is_entail = entail_matrix > threshold
        for i in range(n):
            found = False
            for class_id, members in enumerate(class_to_sample):
                representative = members[0]
                if is_entail[i, representative] and is_entail[representative, i]:
                    class_to_sample[class_id].append(i)
                    sample_to_class[i] = class_id
                    found = True
                    break
            if not found:
                sample_to_class[i] = len(class_to_sample)
                class_to_sample.append([i])
        return sample_to_class

    # --- 13 UQ Methods ---
    def m1_lexical_sim(self, texts):
        n = len(texts)
        if n <= 1: return 1.0
        sets = [set(t.lower().split()) for t in texts]
        sims = []
        for i in range(n):
            for j in range(i+1, n):
                if not sets[i] or not sets[j]: sims.append(1.0 if not sets[i] and not sets[j] else 0.0)
                else: sims.append(len(sets[i] & sets[j]) / len(sets[i] | sets[j]))
        return np.mean(sims)

    def m2_consistency(self, embeddings):
        n = len(embeddings)
        if n <= 1: return 1.0
        sims = cosine_similarity(embeddings)
        mask = ~np.eye(n, dtype=bool)
        return np.mean(sims[mask])

    def m3_semantic_entropy(self, sample_to_class):
        n = len(sample_to_class)
        if n == 0: return 1.0
        counts = np.unique(list(sample_to_class.values()), return_counts=True)[1]
        probs = counts / n
        return np.exp(-entropy(probs))

    def m4_deg_mat(self, entail_matrix):
        n = entail_matrix.shape[0]
        if n <= 1: return 1.0
        W = (entail_matrix + entail_matrix.T) / 2
        return np.mean(W.sum(axis=1)) / n

    def m5_eig_val_laplacian(self, entail_matrix):
        n = entail_matrix.shape[0]
        if n <= 1: return 1.0
        W = (entail_matrix + entail_matrix.T) / 2
        D = np.diag(W.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-9))
        L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
        try:
            eigs = np.linalg.eigvalsh(L)
            return np.sum(np.maximum(0, 1 - eigs)) / n
        except:
            return 0.5

    def m6_semantic_density(self, embeddings):
        n = len(embeddings)
        if n <= 1: return 1.0
        dists = pdist(embeddings, metric='euclidean')
        return 1.0 / (1.0 + np.mean(dists))

    def m7_eccentricity(self, embeddings):
        n = len(embeddings)
        if n <= 1: return 1.0
        centroid = np.mean(embeddings, axis=0)
        dists = np.linalg.norm(embeddings - centroid, axis=1)
        return 1.0 / (1.0 + np.mean(dists))

    def m8_num_sem_sets(self, sample_to_class):
        n = len(sample_to_class)
        if n <= 1: return 1.0
        num_classes = len(set(sample_to_class.values()))
        return 1.0 - (num_classes - 1) / (n - 1)

    def m9_sar(self, entail_matrix):
        n = entail_matrix.shape[0]
        if n <= 1: return 1.0
        mask = ~np.eye(n, dtype=bool)
        return np.mean(entail_matrix[mask])

    def m10_sentence_sar(self, entail_matrix):
        n = entail_matrix.shape[0]
        if n <= 1: return 1.0
        mask = ~np.eye(n, dtype=bool)
        return np.min(entail_matrix[mask])

    def m11_rss(self, embeddings):
        n = len(embeddings)
        if n <= 1: return 1.0
        E = embeddings - np.mean(embeddings, axis=0)
        try:
            s = np.linalg.svd(E, compute_uv=False)
            if len(s) == 0 or np.max(s) == 0: return 1.0
            sr = (np.sum(s**2)) / (np.max(s)**2)
            return np.exp(-(sr - 1))
        except:
            return 0.5

    def m12_ce(self, responses):
        n = len(responses)
        if n <= 1: return 1.0
        comps = [len(zlib.compress(r.encode())) for r in responses]
        total_c = sum(comps)
        if total_c == 0: return 0.0
        p = np.array(comps) / total_c
        h_comp = -np.sum(p * np.log(p + 1e-9))
        h_norm = h_comp / np.log(n)
        return np.exp(-h_norm)

    def m13_hef(self, s_rss, s_ce):
        m1f, m1u = s_rss, 1 - s_rss
        m2f, m2u = s_ce, 1 - s_ce
        return np.clip(m1f*m2f + m1f*m2u + m2f*m1u, 0, 1)

    def evaluate_metrics(self, probs, labels):
        probs = np.array(probs)
        labels = np.array(labels)
        if np.max(probs) > np.min(probs):
            probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs) + 1e-9)

        metrics = {}
        if len(np.unique(labels)) < 2:
            metrics.update({'auroc': 0.5, 'auroc_ci': [0.5, 0.5]})
        else:
            try:
                metrics['auroc'] = roc_auc_score(labels, probs)
                # Small resamples for speed
                res = bootstrap((probs, labels), lambda p, l: roc_auc_score(l, p), paired=True, n_resamples=50, random_state=42)
                metrics['auroc_ci'] = [res.confidence_interval.low, res.confidence_interval.high]
            except:
                metrics['auroc'] = 0.5
                metrics['auroc_ci'] = [0.5, 0.5]

        # ECE
        bin_boundaries = np.linspace(0, 1, 11)
        ece = 0
        for i in range(10):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i+1])
            if i == 9: mask = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
            if np.any(mask):
                ece += np.abs(np.mean(labels[mask]) - np.mean(probs[mask])) * np.sum(mask) / len(probs)
        metrics['ece'] = ece
        metrics['brier'] = brier_score_loss(labels, probs)

        # PRR
        idx = np.argsort(1 - probs)[::-1]
        sorted_labels = labels[idx]
        rejection_rates = np.linspace(0, 0.5, 100)
        area_method = auc(rejection_rates, [np.mean(sorted_labels[int(r*len(labels)):]) if int(r*len(labels)) < len(labels) else 1.0 for r in rejection_rates])
        area_random = auc(rejection_rates, [np.mean(labels)] * 100)
        sorted_oracle = labels[np.argsort(labels)]
        area_oracle = auc(rejection_rates, [np.mean(sorted_oracle[int(r*len(labels)):]) if int(r*len(labels)) < len(labels) else 1.0 for r in rejection_rates])
        metrics['prr'] = (area_method - area_random) / (area_oracle - area_random + 1e-9)
        return metrics

def main():
    parquet_path = "combine/embeddings/combined_data.parquet"
    embeddings_path = "combine/embeddings/response_embeddings.npy"

    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loading embeddings from {embeddings_path}...")
    resp_embeddings = np.load(embeddings_path)

    calculator = UQCalculatorFast()
    models = df['model_name'].unique()
    methods_list = ['LexicalSimilarity', 'Consistency', 'SemanticEntropy', 'DegMat', 'EigValLaplacian',
                    'SemanticDensity', 'Eccentricity', 'NumSemSets', 'SAR', 'SentenceSAR',
                    'RSS', 'CE', 'HEF']

    results_all = {}
    summary_data = []

    for model_name in models:
        print(f"\n--- Processing Model: {model_name} ---")
        model_indices = df[df['model_name'] == model_name].index
        model_df = df.loc[model_indices].copy()
        model_df['target'] = model_df['difficulty_type'].map({'Factual': 1, 'Adversarial': 0})
        eval_df = model_df.dropna(subset=['target'])
        indices = eval_df.index

        model_probs = {m: [] for m in methods_list}
        start_t = time.time()

        for idx in indices:
            res_text = eval_df.loc[idx, 'response_text']
            if isinstance(res_text, str):
                try: responses = json.loads(res_text)
                except: responses = [res_text]
            else: responses = res_text
            responses = [str(r) for r in responses if r is not None][:10]
            emb = resp_embeddings[idx]

            if len(responses) < 2:
                for m in methods_list: model_probs[m].append(0.5)
                continue

            try:
                # Fast embedding-based entailment matrix
                nli_proxy = calculator.get_cosine_entail_matrix(emb)
                s2c = calculator.get_semantic_classes(nli_proxy['entail'])

                model_probs['LexicalSimilarity'].append(calculator.m1_lexical_sim(responses))
                model_probs['Consistency'].append(calculator.m2_consistency(emb))
                model_probs['SemanticEntropy'].append(calculator.m3_semantic_entropy(s2c))
                model_probs['DegMat'].append(calculator.m4_deg_mat(nli_proxy['entail']))
                model_probs['EigValLaplacian'].append(calculator.m5_eig_val_laplacian(nli_proxy['entail']))
                model_probs['SemanticDensity'].append(calculator.m6_semantic_density(emb))
                model_probs['Eccentricity'].append(calculator.m7_eccentricity(emb))
                model_probs['NumSemSets'].append(calculator.m8_num_sem_sets(s2c))
                model_probs['SAR'].append(calculator.m9_sar(nli_proxy['entail']))
                model_probs['SentenceSAR'].append(calculator.m10_sentence_sar(nli_proxy['entail']))

                s_rss = calculator.m11_rss(emb)
                s_ce = calculator.m12_ce(responses)
                model_probs['RSS'].append(s_rss)
                model_probs['CE'].append(s_ce)
                model_probs['HEF'].append(calculator.m13_hef(s_rss, s_ce))
            except:
                for m in methods_list: model_probs[m].append(0.5)

        total_time = time.time() - start_t
        m_results = {}
        for m_name in methods_list:
            metrics = calculator.evaluate_metrics(model_probs[m_name], eval_df['target'].values)
            metrics['compute_time'] = total_time
            m_results[m_name] = metrics
            summary_data.append({
                'Model': model_name, 'Method': m_name, 'AUROC': metrics['auroc'],
                'ECE': metrics['ece'], 'PRR': metrics['prr'], 'Brier': metrics['brier'],
                'ComputeTime': total_time
            })
        results_all[model_name] = m_results
        print(f"Finished {model_name} in {total_time:.2f}s")

    with open("calculated_uq_results.json", "w") as f:
        json.dump(results_all, f, indent=2)
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("calculated_uq_summary.csv", index=False)
    print("\nFull results saved successfully.")

if __name__ == "__main__":
    main()
