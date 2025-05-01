# PurelySemanticIDs

This is the official code repository for the paper _Purely Semantic Indexing for LLM-based Generative Recommendation and Retrieval_.

---

## Requirements

The code is written in **Python 3.8**. Before running, please install the required packages. Using a virtual environment is highly recommended:

```bash
conda create --name <env> --file requirements.txt
```

## Overview

This repository provides two algorithms for constructing **purely semantic IDs** for downstream tasks such as generative recommendation and retrieval.  
The implementation builds on top of [LMIndexer](https://github.com/PeterGriffinJin/LMIndexer), which serves as the base codebase and experimental setup.

---

## Data Preparation

All scripts for **raw data downloading**, **preprocessing**, and **data formatting** follow the structure of the original [LMIndexer](https://github.com/PeterGriffinJin/LMIndexer) repository.

---

## Constructing Semantic IDs

- Code for semantic ID construction is located in:
  - `RetrievalPipeline/LMIndexer/downstream/rqvae/`
  - `RetrievalPipeline/LMIndexer/downstream/hc-indexer/`

- ID generation scripts are located in `RetrievalPipeline/LMIndexer/` and are prefixed with `prepare_id_`.

---

## Downstream Tasks

- Code for downstream tasks is located in:
  - `RetrievalPipeline/LMIndexer/downstream/GEU/`

- Training scripts are located in `RetrievalPipeline/LMIndexer/` and are prefixed with `run_`.

- Evaluation scripts are also in `RetrievalPipeline/LMIndexer/`, prefixed with `test_`.