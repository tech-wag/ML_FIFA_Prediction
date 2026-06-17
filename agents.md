# ML_FIFA Agent

## Purpose

The `ML_FIFA` agent is designed to help developers and reviewers understand, run, and extend the FIFA match outcome prediction repository.

It supports:
- explaining repository architecture and workflow
- running training, prediction, and data refresh tasks
- reviewing model, RAG enrichment, and deployment code
- helping onboard new contributors with usage commands and environment setup

## What this agent knows

- repository structure and file purpose
- Streamlit app usage in `app.py`
- model training pipeline in `TrainModel.py`
- prediction workflow in `Predictor.py`
- lightweight RAG enrichment in `rag_enrich.py`
- embeddings/index build in `rag_production.py`
- Vertex AI deployment helper in `vertex_deploy.py`
- data refresh logic in `data_refresh.py`

## How to use

1. Open this repository in VS Code.
2. Activate the virtual environment:
   ```powershell
   .\fifa_env\Scripts\Activate.ps1
   ```
3. Install dependencies if needed:
   ```powershell
   pip install -r requirements.txt
   ```
4. Open the Copilot or custom agent view and select or refer to this agent.

## Recommended user prompts

- "Summarize how the RAG enrichment pipeline works in this repo."
- "What are the steps to train the model and run predictions locally?"
- "Help me debug why `Predictor.py` returns the wrong class probability."
- "List the deployment steps for Vertex AI in this repository."
- "Explain the differences between `rag_enrich.py` and `rag_production.py`."

## Sharing with others

Include this repository and `agents.md` when sharing the project. This file helps teammates understand the agent's scope and the kinds of tasks it supports.

> Note: This is a documentation file only. If you want to create a fully packaged VS Code Copilot agent, add corresponding `.agent.md` or prompt files in the repository and link them from workspace settings.
