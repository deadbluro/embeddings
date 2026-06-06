import os
import json
import time
import zlib
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics import roc_auc_score, brier_score_loss, auc
from scipy.stats import bootstrap, entropy
from scipy.spatial.distance import pdist
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Setup Seeds for Reproducibility
np.random.seed(42)
torch.manual_seed(42)

class UQCalculator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")

        # Load models
        print("Loading SentenceTransformer...")
        self.st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

        print("Loading NLI model...")
        self.nli_name = "cross-encoder/nli-deberta-v3-small"
        self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_name).to(device)
        self.nli_model.eval()

    def get_nli_matrix(self, texts: List[str]) -> Dict[str, np.ndarray]:
        n = len(texts)
        entail_matrix = np.eye(n)
        contra_matrix = np.zeros((n, n))
        unique_texts, inv = np.unique(texts, return_inverse=True)
        nu = len(unique_texts)
        u_entail = np.eye(nu)
        u_contra = np.zeros((nu, nu))
        pairs = []
        indices = []
        for i in range(nu):
            for j in range(nu):
                if i != j:
                    pairs.append((unique_texts[i], unique_texts[j]))
                    indices.append((i, j))
        if pairs:
            batch_size = 32
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i+batch_size]
                encoded = self.nli_tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(self.device)
                with torch.no_grad():
                    logits = self.nli_model(**encoded).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                for k, (idx_i, idx_j) in enumerate(indices[i:i+batch_size]):
                    # nli-deberta-v3-small output labels: 0: contradiction, 1: neutral, 2: entailment
                    u_contra[idx_i, idx_j] = probs[k, 0]
                    u_entail[idx_i, idx_j] = probs[k, 2]
        for i in range(n):
            for j in range(n):
                entail_matrix[i, j] = u_entail[inv[i], inv[j]]
                contra_matrix[i, j] = u_contra[inv[i], inv[j]]
        return {"entail": entail_matrix, "contra": contra_matrix}

    def get_semantic_classes(self, entail_matrix, threshold=0.6):
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
        # Normalize probs for AUROC/Calibration
        if np.max(probs) > np.min(probs):
            probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs) + 1e-9)

        metrics = {}
        if len(np.unique(labels)) < 2:
            metrics.update({'auroc': 0.5, 'auroc_ci': [0.5, 0.5]})
        else:
            try:
                metrics['auroc'] = roc_auc_score(labels, probs)
                # Lower number of resamples for speed
                res = bootstrap((probs, labels), lambda p, l: roc_auc_score(l, p), paired=True, n_resamples=50, random_state=42)
                metrics['auroc_ci'] = [res.confidence_interval.low, res.confidence_interval.high]
            except Exception as e:
                print(f"AUROC error: {e}")
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
    # Data paths
    parquet_path = "combine/embeddings/combined_data.parquet"
    embeddings_path = "combine/embeddings/response_embeddings.npy"

    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} not found.")
        return

    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loading embeddings from {embeddings_path}...")
    resp_embeddings = np.load(embeddings_path)

    print(f"Total samples: {len(df)}")
    print(f"Embeddings shape: {resp_embeddings.shape}")

    # Initialize calculator
    calculator = UQCalculator()

    models = df['model_name'].unique()
    methods_list = ['LexicalSimilarity', 'Consistency', 'SemanticEntropy', 'DegMat', 'EigValLaplacian',
                    'SemanticDensity', 'Eccentricity', 'NumSemSets', 'SAR', 'SentenceSAR',
                    'RSS', 'CE', 'HEF']

    results_all = {}
    summary_data = []

    # For testing, we can limit the number of samples per model
    SAMPLE_LIMIT = None # Set to an integer for testing, e.g., 5

    for model_name in models:
        print(f"\n--- Processing Model: {model_name} ---")
        model_indices = df[df['model_name'] == model_name].index
        model_df = df.loc[model_indices].copy()

        # Prepare target labels
        model_df['target'] = model_df['difficulty_type'].map({'Factual': 1, 'Adversarial': 0})
        eval_df = model_df.dropna(subset=['target'])
        indices = eval_df.index

        model_probs = {m: [] for m in methods_list}
        start_t = time.time()

        count = 0
        for idx in indices:
            if SAMPLE_LIMIT and count >= SAMPLE_LIMIT:
                break
            count += 1

            # responses are stored as JSON string of list
            res_text = eval_df.loc[idx, 'response_text']
            if isinstance(res_text, str):
                try:
                    responses = json.loads(res_text)
                except:
                    responses = [res_text]
            else:
                responses = res_text

            responses = [str(r) for r in responses if r is not None][:10]
            emb = resp_embeddings[idx] # (10, 384)

            if len(responses) < 2:
                for m in methods_list: model_probs[m].append(0.5)
                continue

            try:
                # NLI matrices
                nli = calculator.get_nli_matrix(responses)
                s2c = calculator.get_semantic_classes(nli['entail'])

                model_probs['LexicalSimilarity'].append(calculator.m1_lexical_sim(responses))
                model_probs['Consistency'].append(calculator.m2_consistency(emb))
                model_probs['SemanticEntropy'].append(calculator.m3_semantic_entropy(s2c))
                model_probs['DegMat'].append(calculator.m4_deg_mat(nli['entail']))
                model_probs['EigValLaplacian'].append(calculator.m5_eig_val_laplacian(nli['entail']))
                model_probs['SemanticDensity'].append(calculator.m6_semantic_density(emb))
                model_probs['Eccentricity'].append(calculator.m7_eccentricity(emb))
                model_probs['NumSemSets'].append(calculator.m8_num_sem_sets(s2c))
                model_probs['SAR'].append(calculator.m9_sar(nli['entail']))
                model_probs['SentenceSAR'].append(calculator.m10_sentence_sar(nli['entail']))

                s_rss = calculator.m11_rss(emb)
                s_ce = calculator.m12_ce(responses)
                model_probs['RSS'].append(s_rss)
                model_probs['CE'].append(s_ce)
                model_probs['HEF'].append(calculator.m13_hef(s_rss, s_ce))

            except Exception as e:
                print(f"Error on sample {idx}: {e}")
                for m in methods_list: model_probs[m].append(0.5)

        total_time = time.time() - start_t

        m_results = {}
        for m_name in methods_list:
            metrics = calculator.evaluate_metrics(model_probs[m_name], eval_df['target'].values[:len(model_probs[m_name])])
            metrics['compute_time'] = total_time
            m_results[m_name] = metrics
            summary_data.append({
                'Model': model_name,
                'Method': m_name,
                'AUROC': metrics['auroc'],
                'ECE': metrics['ece'],
                'PRR': metrics['prr'],
                'Brier': metrics['brier'],
                'ComputeTime': total_time
            })

        results_all[model_name] = m_results
        print(f"Finished {model_name} in {total_time:.2f}s")

    # Save Outputs
    with open("calculated_uq_results.json", "w") as f:
        json.dump(results_all, f, indent=2)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("calculated_uq_summary.csv", index=False)
    print("\nResults saved to calculated_uq_results.json and calculated_uq_summary.csv")

if __name__ == "__main__":
    main()
