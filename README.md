# ML_FIFA — Match Outcome Prediction

This repository implements a Random Forest-based FIFA match outcome predictor enriched with a lightweight Retrieval-Augmented Generation (RAG) pipeline that adds live context (news mentions, injuries) to model inputs.

Contents
- `DataLoader.py` — data loading and feature engineering from `results.csv` (Kaggle dataset)
- `data_refresh.py` — downloads & merges latest Kaggle dataset
- `TrainModel.py` — trains Random Forest and saves `fifa_model.pkl`
- `Predictor.py` — loads model, enriches inputs with `rag_enrich.py`, returns prediction & probabilities
- `rag_enrich.py` — lightweight news scraper + keyword extraction (Google News RSS)
- `rag_production.py` — production scaffold: build embeddings (sentence-transformers) and optional upload to Vertex Matching Engine
- `app.py` — Streamlit UI
- `vertex_deploy.py` — helper for uploading model artifact to GCS and deploying to Vertex AI
- `requirements.txt` — Python dependencies

Quick Start
1. Create and activate venv (already present as `fifa_env` in this workspace):

```powershell
# PowerShell
.\fifa_env\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. (Optional) Refresh Kaggle data (requires Kaggle API token at `~/.kaggle/kaggle.json`):

```powershell
python data_refresh.py --local results.csv
```

4. Train model:

```powershell
python TrainModel.py
```

5. Run Streamlit UI:

```powershell
streamlit run app.py
# or using venv python
.\fifa_env\Scripts\python.exe -m streamlit run app.py
```

6. Predict from CLI:

```powershell
python Predictor.py "Brazil" "Argentina" "Friendly" --neutral
```

7. Build a local RAG index (embeddings):

```powershell
python rag_production.py --build-index --team "Brazil"
```

## Agent and skill guide

This repository includes `agents.md` and `skills.md` so contributors and reviewers can use a shared agent workflow to understand the project and run predictions.

- `agents.md`: describes the purpose of the ML_FIFA agent, supported repository tasks, and recommended prompts.
- `skills.md`: documents skill categories, supported workflows, and example prompts for using the project as an agent-driven experience.

How to leverage them:

1. Open `agents.md` to understand the intended agent capabilities and recommended prompts.
2. Open `skills.md` to see skill categories and sample command questions for prediction, training, RAG enrichment, and deployment.
3. Use the agent-style prompts from those files when asking a teammate or AI assistant to guide you through:
   - running the Streamlit UI
   - training the model
   - generating match predictions
   - building a RAG index
   - deploying with Vertex AI

Example prompt:

```powershell
# Ask an AI assistant or teammate
"How do I run the FIFA match predictor locally and generate a prediction for Brazil vs Argentina?"
```

## Docker and CI/CD

### Build the Docker image

```powershell
cd 'Development\ML_FIFA'
docker build -t ml_fifa:latest .
```

Run the container locally:

```powershell
docker run --rm -p 8501:8501 ml_fifa:latest
```

Then visit `http://localhost:8501` to access the Streamlit UI.

### Run the Jenkins pipeline

This repo includes a `Jenkinsfile` that installs dependencies, performs static checks, and validates the core Python modules.

To execute the pipeline in Jenkins:

1. Create a new Pipeline job in Jenkins.
2. Point the job to this repository.
3. Use the `Jenkinsfile` in the repo as the pipeline definition.
4. Run the job.

You can also run the commands manually in a shell:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m py_compile DataLoader.py TrainModel.py Predictor.py app.py rag_enrich.py rag_production.py vertex_deploy.py
python -c "import pandas, sklearn, streamlit"
```

Architecture Overview
See `ARCHITECTURE.md` for diagrams and component interactions.

Design decisions (short)
- Base model: Random Forest — robust, interpretable, fast to train and tune for structured sports data.
- RAG enrichment: adds recent news/injury signals as extra features; lightweight RSS-based prototype, with a production option using embeddings + Vertex Matching Engine.

Next steps / Suggestions
- Add scheduled refresh + retrain (Cloud Scheduler or Windows Task Scheduler)
- Replace RSS prototype with an embeddings-backed RAG index + semantic search for better contextual signals
- Experiment with XGBoost/LightGBM for potential performance gains

License: MIT
