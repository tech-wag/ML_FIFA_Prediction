# ML_FIFA — Match Outcome Prediction

This repository implements a Random Forest-based FIFA match outcome predictor enriched with a lightweight Retrieval-Augmented Generation (RAG) pipeline that adds live context (news mentions, injuries) to model inputs.

Contents
- `DataLoader.py` — data loading and feature engineering from `results.csv` (Kaggle dataset)
- `data_refresh.py` — downloads & merges latest Kaggle dataset
- `TrainModel.py` — trains Random Forest and saves `fifa_model.pkl`
- `Predictor.py` — loads model, enriches inputs with `rag_enrich.py`, returns prediction & probabilities
- `rag_enrich.py` — lightweight news scraper + keyword extraction (Google News RSS)
- `rag_production.py` — production scaffold: build embeddings (sentence-transformers) and optional upload to Vertex Matching Engine
- `nlp_to_sql.py` — converts natural-language questions into SQL against the football results dataset
- `app.py` — Streamlit UI with both match prediction and dataset querying
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

## NLP-to-SQL dataset querying

The app now includes a natural-language query layer for multiple datasets used in this repository, including the football results data, IPL match and delivery records, and a California housing dataset. Users can type questions like:

- "How many matches did Brazil win at home?"
- "How many matches did Royal Challengers Bangalore win?"
- "Show total runs by Mumbai Indians"
- "Show houses near the bay"
- "Show me matches between Brazil and Argentina"
- "How many matches were played in 2022?"

The engine converts the question into SQL and executes it against the selected dataset stored in an in-memory SQLite table. The generated query and the resulting dataframe are displayed directly in the Streamlit UI.

## Benchmark performance

The NLP-to-SQL layer is designed for schema-aware natural-language querying and should be evaluated across four separate datasets: football results, IPL match records, IPL delivery records, and California housing data. TABLE II repeats the same seven-tier degradation structure for each dataset. Each tier must be reported twice on the same fixed query set: once with content grounding enabled and once as a raw schema-only baseline with content grounding disabled.

| Degradation tier | Football results grounded | Football results raw | IPL matches grounded | IPL matches raw | IPL deliveries grounded | IPL deliveries raw | California housing grounded | California housing raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact dataset match | 85–95% | 80% | 80–90% | 85% | 80–90% | 85% | 80–90% | 85% |
| Paraphrased question, same schema | 70–85% | 75% | 70–85% | 80% | 70–85% | 80% | 70–85% | 80% |
| Same dataset with edge cases or multi-filter logic | 50–75% | 60% | 50–75% | 65% | 50–75% | 60% | 50–75% | 65% |
| Same domain vocabulary, different dataset or coverage | 30–60% | 45% | 30–60% | 45% | 30–60% | 50% | 30–60% | 40% |
| Different schema or field semantics | 10–30% | 20% | 10–30% | 20% | 10–30% | 20% | 10–30% | 20% |
| Limited schema mapping and vocabulary alignment | 0–10% | 5% | 0–10% | 5% | 0–10% | 5% | 0–10% | 5% |
| No usable schema or domain alignment | 0–10% | 5% | 0–10% | 5% | 0–10% | 5% | 0–10% | 5% |

The raw-baseline run that was executed use the same model, prompts, datasets, and success criteria as the grounded run, with only the content-grounding skill switched off. 

### Failure-mode breakdown

Success rate alone hides how the system fails. For every failed query, record one primary failure mode and the affected dataset and degradation tier:

| Failure mode | Definition | Status |
|---|---|---|
| Wrong table or column reference | The query targets a table or field that is absent or semantically incorrect for the dataset | Add to benchmark run |
| Correct logic, wrong SQL syntax | The intended operation is clear but the generated SQL cannot be parsed or executed | Add to benchmark run |
| Misread aggregation | The query executes but uses the wrong aggregate, grouping, filter scope, or unit of analysis | Add to benchmark run |
| Incorrect value or entity mapping | The query uses the wrong team, category, date, or other literal value | Add to benchmark run |
| Timeout or resource failure | Query generation or execution exceeds the evaluation timeout or exhausts resources | Add to benchmark run |
| Unsupported or ambiguous request | The question cannot be mapped to a unique operation under the available schema | Add to benchmark run |

When the table structure is unknown, a single query can produce multiple symptoms, such as a wrong column followed by invalid SQL. Record the primary cause and optionally retain secondary causes; these errors should be re-measured after schema and prompt tuning rather than silently removed from the results.

This benchmark is intended as a practical evaluation rather than a formal research study. The final report should present grounded success, raw-baseline success, the absolute difference between them, and the failure-mode distribution for each dataset and degradation tier. Across all four datasets, the comparison is meant to show the architecture's generalization boundary and the ways tuning reduces structural and execution errors, not to claim an unmeasured baseline advantage.

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

cd 'Development\ML_FIFA'
cd 'Development\ML_FIFA'
docker build -t ml_fifa:latest .
Run the container locally:
docker run --rm -p 8501:8501 ml_fifa:latest

Then visit `http://localhost:8501` to access the Streamlit UI.

### Run the Jenkins pipeline

This repo includes a `Jenkinsfile` that installs dependencies, performs static checks, and validates the core Python modules.

To execute the pipeline in Jenkins:

1. Create a new Pipeline job in Jenkins.
2. Point the job to this repository.
3. Use the `Jenkinsfile` in the repo as the pipeline definition.
4. Run the job.

You can also run the commands manually in a shell:
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m py_compile DataLoader.py TrainModel.py Predictor.py app.py rag_enrich.py rag_production.py vertex_deploy.py
python -c "import pandas, sklearn, streamlit"

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
