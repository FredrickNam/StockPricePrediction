# 📈 Stock Pricing Prediction: A White-Box Approach

<br>

## 📖 About This Repository
An End-to-End Stock Pricing Prediction System built from scratch using Deep Learning and Reinforcement Learning. This project focuses on a 'White-Box' approach, reproducing core papers like Transformer with Python and PyTorch.

<br>

## 💡 About The Project
This project aims to develop a sophisticated stock pricing prediction algorithm based on non-linear financial data. Moving away from traditional 'Black Box' API calls, our team focuses on a Deep Dive Reproduction.

We manually construct complex neural network architectures and mathematical formulas from the ground up to secure true AI engineering capabilities. By combining the sequential understanding of Transformer models with the decision-making power of Reinforcement Learning (RL), we build an end-to-end autonomous trading pipeline.

<br>

## ✨ Key Features
### End-to-End Data Engineering 🌐

Custom web crawlers built to fetch raw stock prices, volumes etc.

Pipelines for handling missing values, removing outliers, and normalizing time-series data.

### White-Box Deep Learning (Transformer) 🧠

PyTorch implementations inheriting from nn.Module without relying on high-level pre-built layers.

Reproduction of the 'Attention Is All You Need' (2017) architecture, mathematically mapped and customized for noise-heavy time-series forecasting.

### Reinforcement Learning Trading Environment 🤖

Custom stock trading simulator calculating realistic returns, including transaction fees.

Integration of the custom Transformer as a feature extractor, feeding into an RL policy network to determine optimal Buy/Sell actions.

### Robust Backtesting & Analytics 📊

Comprehensive backtesting using historical data (unseen during training).

Deep evaluation using quantitative investment metrics such as Maximum Drawdown (MDD), win rates, and profit/loss ratios.

<br>

## 📂 Repository Structure
```  
Stock-Pricing-Prediction/  
├── data/                  # Local data storage (Ignored by Git)  
│   ├── raw/               # Raw crawled stock data   
│   └── processed/         # Preprocessed and normalized data  
├── docs/                  # Project documentation & Study notes  
│   ├── papers/            # Reference papers (Transformer, PPO, etc.)  
│   └── mathematics/       # Mathematical derivations and markdown notes  
├── notebooks/             # Jupyter notebooks for experiments & visualization  
│   └── week01_tensor/     # Week 1: Tensor manipulation and matrix math practice   
├── src/                   # Core source code modules  
│   ├── data_pipeline/     # Web crawlers (Selenium/BS4) & preprocessing scripts  
│   ├── models/            # Custom PyTorch nn.Module architectures  
│   ├── rl_env/            # Trading simulator and reward calculation logic  
│   └── backtesting/       # Backtesting and performance metric (MDD) analysis  
├── tests/                 # Unit testing scripts (e.g., Tensor I/O flow validation)  
├── .gitignore             # Git ignore file (Excludes data, venv, keys)  
├── requirements.txt       # Project dependencies (PyTorch, Numpy, Pandas, etc.)   
└── README.md              # Project overview and execution guide  
```

<br>

## 🛠️ Tech Stack

Language: Python

Deep Learning: PyTorch (Customized), Numpy, Pandas

Data Collection: Selenium, BeautifulSoup

<br>

## 👥 Team
Changwoo Nam (Project Lead / Model Architecture & End-to-End Integration)

Junghoon Kim (Data Pipeline & Backtesting System)

Yehwan Jang (RL Environment & Hyperparameter Optimization)

Developed as part of the 2026-1 Creative Semester Project.
