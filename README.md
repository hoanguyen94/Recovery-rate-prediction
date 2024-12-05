
# Recovery Rate Prediction

This repository contains resources for predicting recovery rates using machine learning techniques.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Structure](#structure)
- [Contact](#contact)

---

## Introduction

The **Recovery Rate Prediction** project aims to explore and implement machine learning models to predict recovery rates based on historical datasets. The project includes code for data analysis, and model training.

---

## Requirements

- Python 3.12
- Libraries listed in `requirements.txt`

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/hoanguyen94/Recovery-rate-prediction.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd Recovery-rate-prediction
   ```

3. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   ```

   - Activate the virtual environment:
     - On Linux/macOS:

       ```bash
       source venv/bin/activate
       ```

     - On Windows:

       ```bash
       venv\Scripts\activate
       ```

4. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Prepare your dataset, create a data folder and place it in the appropriate directory.
2. Run the provided scripts for analysis, training, or evaluation as needed.


## Structure

```
Recovery-rate-prediction/
.
├── README.md
├── data (not public)
│   ├── All_data_cleaned.xlsx
│   ├── all_data_cleaned_with_one_hot_encoding.xlsx
│   ├── profile_output.html
│   ├── test_features.xlsx
│   ├── test_features_w_one_hot_enc_normalized.xlsx
│   ├── test_labels.xlsx
│   ├── train_features.xlsx
│   ├── train_features_w_one_hot_enc.xlsx
│   ├── train_features_w_one_hot_enc_normalized.xlsx
│   ├── train_labels.xlsx
│   └── train_labels_w_one_hot_enc.xlsx
├── requirements.txt                            # all libraries needed for this repo
└── src
    ├── Cubist.R                                # Model training and validation with Cubist model
    ├── Data_analysis.ipynb                     # Exploratory data analysis and data splitting for experimentation
    ├── LightGBM.ipynb                          # Model training and validation with Light GBM
    ├── XGBoost.ipynb                           # Model training and validation with XGBoost
    ├── mlp_5fold.ipynb                         # Model training and validation with MLP
    ├── mlp_w_Embeddings.ipynb                  # Model training and validation with MLP & embeddings for categorical features from pretrained LLM
    ├── random_forest.ipynb                     # Model training and validation with Random Forest
    ├── resnet_5fold.ipynb                      # Model training and validation with Resnet 
    ├── resnet_5fold_w_Emb.ipynb                # Model training and validation with Resnet & embeddings for categorical features from pretrained LLM
    ├── svm.ipynb                               # Model training and validation with Support Vector Machine
    ├── transformer.ipynb                       # Model training and validation with Transformer
    └── utils.py                                # Utilities file

```

---

## Contact

- **GitHub:** [hoanguyen94](https://github.com/hoanguyen94)

For inquiries, open an issue or contact the repository owner through GitHub.

---

**Note:** Update this README file as the project evolves to include specific instructions and additional details.

