# Topic-Modeling-Projects

#!/usr/bin/env python
# coding: utf-8

# # Game of Topic Modeling: Evaluating NMF Practice on the Book series by George R.R Martin
# #### DTSA 5510 Final Project
# ---
# ---

# # Topic Modeling: Comparing Two Unsupervised Methods

# **Latent Dirichlet Allocation (LDA)** and **Non-negative Matrix Factorization (NMF)** are both unsupervised topic modeling techniques, but they differ significantly in their underlying mechanisms, assumptions, and outputs. Here's a comparison of their key differences:
# 
# ### 1. **Mathematical Approach**:
#    - **LDA (Probabilistic Model)**:
#      - **Key Idea**: LDA is a probabilistic model that assumes each document is a mixture of topics, and each topic is a mixture of words.
#      - **Process**: It assigns probabilities to each word belonging to a topic and each document containing certain topics. The algorithm tries to estimate these probabilities through inference methods like Gibbs sampling or variational inference.
#      - **Output**: For each document, you get a distribution over topics (e.g., Document A is 30% Topic 1, 70% Topic 2), and for each topic, you get a distribution over words (e.g., Topic 1 consists of the words "data," "science," "machine" with certain probabilities).
#    
#    - **NMF (Matrix Factorization)**:
#      - **Key Idea**: NMF decomposes a document-term matrix into two lower-rank, non-negative matrices: one representing documents in terms of topics and another representing topics in terms of words.
#      - **Process**: NMF approximates the document-term matrix by factorizing it into a document-topic matrix and a topic-word matrix. This means each document is a weighted combination of topics, and each topic is a weighted combination of words.
#      - **Output**: Similar to LDA, NMF provides a topic distribution for each document and a word distribution for each topic, but without any probabilistic interpretation.
# 
# 
# ### Comparison Table:
# 
# | Aspect                    | LDA                             | NMF                            |
# |---------------------------|----------------------------------|---------------------------------|
# | **Method**                 | Probabilistic model              | Matrix factorization            |
# | **Mathematical Framework** | Probabilistic inference          | Linear algebra (non-negative matrix decomposition) |
# | **Output**                 | Probabilities of words per topic and topics per document | Non-negative weights for words per topic and topics per document |
# | **Interpretability**       | Less interpretable, probabilistic values | More interpretable, non-negative weights |
# | **Assumption**             | Probabilistic topic mixtures     | Non-negative matrix factorization |
# | **Scalability**            | Slower, more complex             | Faster, simpler                  |
# | **Common Use**             | Mixed-topic documents            | Easily interpretable topic modeling |
# | **Overlapping Topics**     | Models topic mixtures well       | Less flexible with overlapping topics |
# 
