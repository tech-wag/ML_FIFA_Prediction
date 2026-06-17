# ML_FIFA Skills

## Overview

This file describes the skills that a custom agent or Copilot workflow should provide for the `ML_FIFA` repository.

The skills focus on repository navigation, model and data pipeline tasks, and production readiness.

## Skill categories

### 1. Repository Orientation

- Explain repository structure and core modules
- Detail file responsibilities for `app.py`, `TrainModel.py`, `Predictor.py`, `rag_enrich.py`, `rag_production.py`, and `vertex_deploy.py`
- Summarize the README quick start and any Docker/CI instructions

### 2. Environment Setup

- Activate and use the existing `fifa_env` Python virtual environment
- Install dependencies from `requirements.txt`
- Validate Python package availability

### 3. Training and Prediction

- Walk through training the model with `TrainModel.py`
- Explain how the trained artifact `fifa_model.pkl` is produced and used
- Show CLI prediction examples using `Predictor.py`

### 4. RAG Enrichment

- Describe the lightweight RAG source in `rag_enrich.py`
- Explain how news and keyword signals are merged into model features
- Detail the optional embedding/index build flow in `rag_production.py`

### 5. Deployment

- Explain deployment helper `vertex_deploy.py`
- Summarize Docker image build and local container run commands
- Describe how Vertex AI deployment is intended to work for this project

## Example prompts for skill use

- "How do I start the Streamlit UI in this repo?"
- "What does `data_refresh.py` do and when should I run it?"
- "Explain how to build the embedding index in `rag_production.py`."
- "What are the input features required by the FIFA prediction model?"
- "How can I add a new news source to the RAG enrichment pipeline?"

## Contribution guidance

If you want to extend this repo, add new skills to this file such as:
- "Data validation checks"
- "Hyperparameter tuning and model selection"
- "Cloud monitoring and inference logging"
- "Automated dataset refresh scheduling"

## Notes

This `skills.md` file is intended as a companion guide and can be used to define a custom agent's capabilities for non-technical users or reviewers.
