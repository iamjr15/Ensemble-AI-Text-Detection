

# Detecting AI-Generated Text using Ensemble Machine Learning

## Introduction

This project detects if a given piece of text is human-written or AI-generated using an ensemble of Machine Learning models.

## Datasets

The training data for this project comes from two Kaggle datasets:

1. [AI Generated Text Dataset (DAIGT)](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset): Contains over 300,000 text samples classified as AI or human-generated, collected from multiple models and fine-tuning methods. This forms the primary training data.

2. [LLM Detect Train Dataset](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data): Additional samples of human-written and AI-generated text from past Kaggle competitions, augmenting the variety of writing styles. 

## Data Preprocessing

As text data cannot be directly ingested by machine learning algorithms, the raw text requires substantial preprocessing to convert it into informative, numeric features.

### Duplicate Removal
Duplicate entries in the training set are dropped to prevent overfitting.

### Tokenization 
The text is broken down into words and n-gram tokens that retain local context, using Byte-pair encoding implemented through the HuggingFace Tokenizers library. Key hyperparamters like n-gram range, vocabulary size, case sensitivity etc. are configured for optimization.

### Vectorization
Scikit-Learn's TF-IDF (term frequency–inverse document frequency) vectorizer converts the processed tokens for each text sample into fixed-length numeric vectors. The vectors capture semantic relationships between words and phrases in the corpus. Custom parameters fine-tune aspects like n-gram ranging, sub-linear term frequencies etc. during vectorization.

## Exploratory Data Analysis

Once vectorized into consistent numeric representations, the training data can be statistically analyzed for insights through EDA techniques:

- Class imbalance checking between human vs AI categories
- Distribution analysis of feature coefficients
- Clustering using dimensionality reduction to assess separability  

Any additional data quality issues or anomalies can be handled at this stage.

## Model Development

A combination of 4 complementary machine learning models is selected after experimentation, to create an ensemble classifier for maximum accuracy and robustness:

### 1. Multinomial Naive Bayes
The Multinomial NB model makes strong independence assumptions between features but is simple, fast to train and avoids overfitting.

### 2. Logistic Regression with SGD Training
Logistic regression optimized through stochastic gradient descent can learn complex decision boundaries. Regularization handles high dimensionality.  

### 3. LightGBM
Gradient-boosted decision tree models like LightGBM perform well on tabular data and are robust to noise. The hyperparameters are tuned through Kaggle submissions.

### 4. CatBoost 
As a boosted tree ensemble, CatBoost auto-handles categorical variables and is resilient to overfitting through smart regularization techniques.

### Ensemble Configuration
The 4 models are combined using soft-voting, which averages their probabilistic predictions. Manual weighting can bias the ensemble towards the best-performing constituent models.

## Evaluation
The ensemble model is evaluated on a held-out test set using the AUC ROC metric to quantify its ability at separating the human and AI classes. As a probability classifier, ROC AUC captures performance better than raw accuracy. Other classification metrics can provide deeper insights as well.

## Conclusion
Through systematic data preprocessing, predictive modeling and evaluation, this project develops an automated solution for detecting AI-generated text with high accuracy. The ensemble approach leads to significant improvement over any single model. Additional gains can be achieved through more advanced natural language processing and neural network architectures at the cost of interpretability.
