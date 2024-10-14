# Topic-Modeling-Projects


---

# Game of Topic Modeling: Evaluating NMF on George R.R. Martin's Book Series

---

## Topic Modeling: Comparing Two Unsupervised Methods

In this project, we compare two widely used unsupervised topic modeling techniques: **Latent Dirichlet Allocation (LDA)** and **Non-negative Matrix Factorization (NMF)**. Though both methods aim to uncover latent topics in a set of documents, they differ in their approaches, assumptions, and outputs.

### Key Differences:

### 1. **Mathematical Approach**:
   - **LDA (Latent Dirichlet Allocation)**:
     - **Key Idea**: LDA is a probabilistic model where each document is a mixture of topics, and each topic is a mixture of words.
     - **Process**: It assigns probabilities to words in topics and topics in documents using inference methods such as Gibbs sampling or variational inference.
     - **Output**: You get a distribution of topics in each document (e.g., Document A is 30% Topic 1, 70% Topic 2) and a distribution of words in each topic (e.g., Topic 1 consists of the words "data," "science," "machine" with certain probabilities).
   
   - **NMF (Non-negative Matrix Factorization)**:
     - **Key Idea**: NMF decomposes a document-term matrix into two lower-rank non-negative matrices: one representing documents in terms of topics, and another representing topics in terms of words.
     - **Process**: NMF approximates the document-term matrix by factorizing it into a document-topic matrix and a topic-word matrix. This gives each document a weighted combination of topics and each topic a weighted combination of words.
     - **Output**: Similar to LDA, NMF provides topic distributions for each document and word distributions for each topic, but without probabilistic interpretation.

### Comparison Table:

| Aspect                    | LDA                             | NMF                            |
|---------------------------|----------------------------------|---------------------------------|
| **Method**                 | Probabilistic model              | Matrix factorization            |
| **Mathematical Framework** | Probabilistic inference          | Linear algebra (non-negative matrix decomposition) |
| **Output**                 | Probabilities of words per topic and topics per document | Non-negative weights for words per topic and topics per document |
| **Interpretability**       | Less interpretable, probabilistic values | More interpretable, non-negative weights |
| **Assumption**             | Probabilistic topic mixtures     | Non-negative matrix factorization |
| **Scalability**            | Slower, more complex             | Faster, simpler                  |
| **Common Use**             | Mixed-topic documents            | Easily interpretable topic modeling |
| **Overlapping Topics**     | Models topic mixtures well       | Less flexible with overlapping topics |

---

