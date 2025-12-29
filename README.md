# Project Overview

This repository contains a collection of modules and tools related to reinforcement learning, factor graphs, and agent-based systems, likely for Large Language Model (LLM) applications.

Below is a description of the main directories and their purposes:

## Directory Structure

### `CPT/`
Contains the core implementation of the CPT (likely Conditional Probability Table or a specific model component) module.
- **Key File:** `CPT.py` - Implements the CPT logic using PyTorch and Transformers.

### `CPT_FactorGraph_Run/`
Scripts and tools for running factor graph-based executions, possibly integrating with the CPT module.
- **Key Files:**
    - `run_online.py`: Script for running the system in an online mode.
    - `build_trajectory_to_graph.py`: Tools to convert execution trajectories into graph structures.

### `eval/`
Evaluation scripts and metrics for assessing model performance.
- **Key Files:**
    - `run.py`: Main evaluation entry point.
    - `aucroc.py`: Calculates Area Under the ROC Curve.
    - `run_ece_evalution.py`: Runs Expected Calibration Error evaluation.

### `hamster_agent/`
A comprehensive agent application framework, including backend, frontend, and core agent logic.
- **Subdirectories:**
    - `app/`: Core agent logic, tools, and flow definitions.
    - `backend/`: API server implementation (likely FastAPI or similar).
    - `frontend/`: Web interface for the agent.
    - `config/`: Configuration files for different models (Anthropic, Azure, Google, etc.).

### `hamster_factor_graph/`
A library implementing Factor Graphs and Belief Propagation algorithms.
- **Key File:** `factorgraph/factorgraph.py` - Core implementation of the factor graph data structure and inference algorithms.

### `Online_Search_Server/`
A standalone server or module dedicated to performing online web searches and information retrieval.
- **Key Files:**
    - `search_pipeline.py`: Orchestrates the search process.
    - `search.py`: Defines the Web Search Agent.
    - `fetch.py`: Defines the Web Read Agent for fetching page content.

### `verl_hamster/`
The **verl** (Volcano Engine Reinforcement Learning) library, which is a flexible and efficient RL training library for LLMs.
- **Purpose:** Provides infrastructure for Reinforcement Learning from Human Feedback (RLHF) and other RL algorithms for LLMs.
- **Contents:** Includes dockerfiles, documentation, examples, and the core `verl` package.
- **Key Modules:**
    - **`Hamster_Cpt_Worker/`**: A worker service for graph-based operations. It initializes a `GraphBuilder` with a CPT model and performs inference on factor graphs.
    - **`Hamster_Generation_Manager/`**: Manages the generation process, integrating LLM generation with factor graph construction and online search. It handles trajectory initialization, prompt extraction, and interaction with VLLM and search APIs.
    - **`Hamster_Reward_Manager/`**: Computes rewards for generated trajectories using a composite reward function that includes format scores, calibration scores (KL divergence), structure scores (entropy), and step-wise rewards.

## Getting Started

Please refer to the specific `README.md` files within subdirectories (e.g., `verl_hamster/README.md`) for detailed installation and usage instructions for each component.
