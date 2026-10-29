# 🎬 Movie Recommender Systems with RAG 
## Description

This repository contains the group project developed for the **TACD (Topics in Advanced Data Science)** course at the **University of Porto (FCUP)**, academic year 2025/2026, developed by André Vieira,
Miguel Silva and Ricardo Ribeiro.

The project follows the **CRISP-DM** methodology and is structured in two legs:

- **1st Leg** — Implementation and evaluation of multiple interaction-based recommender system approaches on the [MovieLens (ml-latest-small)](https://grouplens.org/datasets/movielens/latest/) dataset.
- **2nd Leg** — Extension with **Retrieval-Augmented Generation (RAG)** and semantic retrieval using LLMs and a vector database built from movie metadata.

## 📁 Repository Structure

```
TACD_MovieRecommender/
│
├── dados/
│   ├── ml-latest-small/          # MovieLens dataset (ratings, movies, tags, links)
│   │   ├── movies.csv
│   │   ├── ratings.csv
│   │   ├── tags.csv
│   │   └── links.csv
│   └── sample_data/              # Subsampled data and pre-computed features
│       ├── ratings_sample_transformed.csv
│       ├── basket_matrix_ar.csv
│       ├── movies_embeddings.npy
│       ├── movies_faiss.index
│       ├── movies_meta.csv
│       ├── rag_explanations.csv
│       ├── train_explicit.csv
│       ├── train_bpr_ncf.csv
│       ├── test_explicit.csv
│       ├── val_explicit.csv
│       └── transactions_ar.csv
│
├── Trabalho_GrupoG_P.ipynb       # Final project notebook (full pipeline)
├── Assignment_TACD_2526_updated.pdf      # Original assignment specification
└── README.md                     # Project documentation
```

---

## Project Overview

### 1st Leg — Interaction-Based Recommender Systems

Following the CRISP-DM methodology:

**1. Business Understanding**
- Framing the movie recommendation problem as a machine learning task
- Defining success criteria (Precision@3, Recall@3) for binary and non-binary scenarios

**2. Data Understanding**
- Exploratory analysis of the `ratings` table
- Characterization of users, items, and basket size distribution
- Power-law analysis of rating frequency

**3. Data Transformation**
- Data splits: training, validation, and test sets
- 20% user subsample for probing experiments
- Transformations for explicit and implicit feedback scenarios

**4. Modeling**

The following recommender system approaches were implemented and compared:

| Method | Description |
|---|---|
| **AR** | Association Rules |
| **CF** | Collaborative Filtering (user/item-based) |
| **MFPQ** | Matrix Factorization with PQ decomposition |
| **BPR** | Bayesian Personalized Ranking |
| **NCF** | Neural Collaborative Filtering |

- Hyperparameter tuning (neighbours, latent factors, regularization λ)
- Precision-Recall curves for the best-performing approaches
- Regularization analysis across different λ values

---

### 2nd Leg — RAG-Based Semantic Recommendations

Extending the first leg with content-based and information retrieval methods:

- **LLM Integration** — LLM-based summarization of user sessions into preference queries (via IAEdu API)
- **Vector Database** — FAISS index built from movie metadata (titles, genres, tags)
- **Retrieval Pipeline** — Semantic retrieval of candidate movies from the vector database
- **Re-ranking** — Optional re-ranking step to improve final recommendation quality
- **Comparison** — RAG-based vs. interaction-based recommenders vs. ground truth
- **Explainability** — LLM-generated explanations for recommendations, evaluated on usefulness, faithfulness, clarity, and specificity

---

## Dataset — MovieLens (ml-latest-small)

| Property | Detail |
|---|---|
| Users | 610 |
| Movies | ~9,700 |
| Ratings | ~100,000 |
| Rating scale | 0.5 to 5.0 stars |
| Source | [GroupLens](https://grouplens.org/datasets/movielens/latest/) |

---

## Technologies & Libraries

- **Python 3**
- **Jupyter Notebook**
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — preprocessing and evaluation
- `faiss` — vector similarity search
- `sentence-transformers` — text embeddings for the vector database
- `scipy` — statistical analysis
- IAEdu / OpenAI-compatible API — LLM access for summarization and explanations

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/andrevieira1203/TACD_MovieRecommender.git
   cd TACD_MovieRecommender
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn faiss-cpu sentence-transformers scipy jupyter
   ```

3. Download the MovieLens dataset:
   ```bash
   # Dataset available at: https://grouplens.org/datasets/movielens/latest/
   # Place the extracted ml-latest-small/ folder inside data/
   ```

4. Open the notebook:
   ```bash
   jupyter notebook Trabalho_GrupoG_P.ipynb
   ```

---

## Languages

| Language | Percentage |
|---|---|
| Jupyter Notebook | 100% |

---

## 📄 License

This project was developed for academic purposes at FCUP/FEUP. For use or reproduction, please contact the authors.
