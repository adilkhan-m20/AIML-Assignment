# 🧩 AI Assignments Collection

This repository contains three mini-projects based on core AI and ML concepts:
1. **ML Regression Model** — Linear Regression with preprocessing and evaluation.
2. **8-Puzzle Problem Solver** — State-space search with BFS, DFS, UCS, and A*.
3. **Wumpus World Agent** — Logic-based agent reasoning with propositional inference.

---

## 📁 Project Structure

ai-assignments/
├── .gitignore               # Git ignore file (ignores venv, cache, etc.)
├── requirements.txt         # All dependencies for ML and AI projects
├── README.md                # Project documentation
│
├── ml_regression.py         # Machine Learning regression model (Linear Regression)
│   ├── Loads California Housing dataset (fallback: Diabetes)
│   ├── Builds a Scikit-learn pipeline with StandardScaler
│   ├── Evaluates using RMSE, MAE, R²
│   └── Saves plots and model_card.json
│
├── eight_puzzle.py          # State-space search problem (8-puzzle)
│   ├── Implements BFS, DFS, UCS, A*
│   ├── Uses Manhattan and Linear-Conflict heuristics
│   ├── Compares nodes expanded and runtime
│   └── Saves performance plots
│
└── wumpus_world.py          # Logic-based AI reasoning agent
    ├── Implements propositional reasoning for pits/safe cells
    ├── Uses breeze percepts for inference
    ├── Infers pit locations and explores safe cells
    └── Demonstrates gold collection and safe navigation
