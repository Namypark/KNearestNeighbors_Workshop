# 🧑‍💻 K-Nearest Neighbors (KNN) Workshop

## 📘 Workshop Summary
This repository contains the materials and student solutions for the **K-Nearest Neighbors (KNN) Workshop**.  
The workshop introduces students to data analysis, machine learning design patterns, and active learning practices through the implementation of the KNN algorithm.

Students will explore multiple types of datasets (CSV files, APIs, relational databases), implement KNN classifiers, and design their code using the **Machine Learning Pipeline Pattern**. The workshop emphasizes collaboration through **peer programming** and reflection on algorithm performance.

---

## ⚙️ Setup

### Requirements
- **Python 3.13** or newer (pinned in `.python-version`)
- An **internet connection**: the solution notebook downloads the Rice dataset from the UCI Machine Learning Repository API when it runs
- **VS Code** with the Jupyter extension, or **JupyterLab**, to open the notebooks

### Option 1: Install with uv (recommended)
[uv](https://docs.astral.sh/uv/) reads `pyproject.toml` and `uv.lock`, so everyone gets exactly the same package versions.

```bash
git clone https://github.com/Namypark/KNearestNeighbors_Workshop.git
cd KNearestNeighbors_Workshop
uv sync
```

`uv sync` creates a `.venv` folder (downloading Python 3.13 if you don't have it) and installs every dependency into it.

### Option 2: Install with pip
```bash
git clone https://github.com/Namypark/KNearestNeighbors_Workshop.git
cd KNearestNeighbors_Workshop
python3.13 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the notebook
- **VS Code:** open `KNN_Workshop_Solution.ipynb`, click **Select Kernel** in the top-right corner, and choose the `.venv` environment. Then click **Run All**.
- **JupyterLab:** run `uv run --with jupyterlab jupyter lab` (uv), or `pip install jupyterlab` then `jupyter lab` (pip), and open `KNN_Workshop_Solution.ipynb`.

### Notes
- Charts are built with **Plotly**. They are interactive in VS Code and JupyterLab, and the notebook also saves a static PNG copy of each chart (via `kaleido`) so they display on GitHub.
- `kaleido` needs Google Chrome to export those PNGs. If you see a Chrome-related error, run `plotly_get_chrome` inside the activated environment (or `uv run plotly_get_chrome`) to install a compatible copy.

---

## 🎯 Learning Objectives
By completing this workshop, you will be able to:
1. Understand the intuition behind the K-Nearest Neighbors algorithm.  
2. Acquire, clean, and preprocess data from different sources.  
3. Apply KNN for classification and evaluate its performance.  
4. Architect code using the **Machine Learning Pipeline Pattern**.  
5. Collaborate with peers through active learning and peer programming.  
6. Reflect on the strengths, limitations, and real-world applications of KNN.  

---

## 🤖 About the KNN Algorithm
The **K-Nearest Neighbors algorithm** is a simple, non-parametric machine learning method used for classification and regression.  
- A new data point is classified based on the **majority label of its *k* closest neighbors** in the training set.  
- The choice of *k* and the method of measuring distance (e.g., Euclidean, Manhattan) heavily influence performance.  
- KNN is **instance-based** (lazy learning): it stores the training data and makes predictions only at query time.  

---

## 🏗️ Implementation with ML Architecture and Design Patterns
This workshop emphasizes implementing KNN using the **Machine Learning Pipeline Pattern**:
- **Data Acquisition** (from CSV, API, or relational database)  
- **Data Cleaning & Preprocessing** (scaling, encoding, handling missing values)  
- **Feature Engineering**  
- **Model Training** (KNN classifier)  
- **Evaluation** (accuracy, confusion matrix, other metrics)  
- **Reflection & Peer Comparison**  

By structuring code into **modular components**, students will learn how to build reproducible and extensible ML workflows, aligning with professional MLOps practices.

---

## 📂 High-Level View of the Workshop
The workshop is split across two sessions (two hours total), with homework in between:

- **Session 1**  
  - Introduction to KNN (theory and Iris dataset demo)  
  - Hands-on: load and preprocess a dataset from a CSV, API, or relational DB  

- **Homework**  
  - Finalize preprocessing and prepare dataset for KNN  

- **Session 2**  
  - Implement KNN using the Pipeline Pattern  
  - Train, evaluate, and visualize results  
  - Reflect on algorithm performance  
  - Compare results with peers and discuss strengths/weaknesses of KNN  

---

## 📝 Deliverables
- `KNN_Workshop_Solution.ipynb`: Your notebook implementation with code and explanations.  
- `README.md`: This file, including your name, student ID, and a brief summary of your work.  
- A PDF file with your name, student ID, and the URL to your remote repository.  

---

## ✍️ Instructor Information
- **Name:** *David Espinosa*  
- **Release:** *October 2025*  
