# RESEARCH REPORT: ALGORITHMIC SPECTRAL GAP (ASG)

## Method Overview
**Algorithmic Spectral Gap (ASG)** is a zero-parameter uncertainty quantification method designed for Dense RLHF and MoE architectures. It identifies semantic consensus by analyzing the eigenvalue distribution of a similarity matrix derived from the **Normalized Compression Distance (NCD)**.

## Mathematical Formulation
The method is defined by the following steps:

1.  **Pairwise Algorithmic Distance**: For a set of $n$ responses $\{r_1, \dots, r_n\}$, we compute the pairwise NCD matrix $D$, where $D_{ij} = NCD(r_i, r_j)$.
    $$NCD(x, y) = \frac{C(xy) - \min(C(x), C(y))}{\max(C(x), C(y))}$$
    where $C(x)$ is the zlib-compressed length of string $x$.

2.  **Kernel Transformation**: We map the distances to a similarity matrix $W$ using a self-scaling heat kernel:
    $$W_{ij} = \exp\left(-\frac{D_{ij}}{\overline{D}}\right)$$
    where $\overline{D}$ is the mean of all off-diagonal distances. This ensures the method is parameter-free and adaptive to the local semantic density of the ensemble.

3.  **Spectral Gap Calculation**: We compute the eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n$ of the similarity matrix $W$. The **Algorithmic Spectral Gap** is defined as the normalized difference between the top two eigenvalues:
    $$P(\text{factual}) = \frac{\lambda_1 - \lambda_2}{n}$$

## Theoretical Grounding
-   **Algorithmic Information Theory (AIT)**: NCD approximates the Universal Metric based on Kolmogorov complexity. In factual ensembles, responses are semantically redundant, leading to low NCD.
-   **Spectral Graph Theory**: The spectral gap of a similarity matrix indicates the strength of the "principal cluster." A large gap $\lambda_1 \gg \lambda_2$ signifies a single dominant semantic direction (consensus), whereas a small gap indicates multiple competing clusters or high noise (uncertainty).

## Performance Summary (Dense RLHF)
| Model | ASG AUROC | Baseline (EigValLaplacian) |
| :--- | :--- | :--- |
| mistral-7b | **0.8062** | 0.6338 |
| llama-3.3-70b | 0.6002 | 0.6044 |

**Mean AUROC (Dense RLHF): ~0.703** (Exceeds success criteria of 0.65)

## Expected Strengths
-   **Architecture Agnostic**: Works purely on response text without requiring internal model states.
-   **Adaptive Scaling**: The self-scaling kernel handles varying lengths and styles of responses across different models.

## Expected Failure Modes
1.  **High Lexical Diversity with Low Semantic Change**: If a model produces correct but very diverse phrasings (e.g., mathematical proofs with different variable names), the compressor might fail to recognize the redundancy, lowering the gap.
2.  **Short Responses**: With very short strings (e.g., "Yes" vs "No"), the compression signal is noisy, leading to unreliable spectral distributions.
