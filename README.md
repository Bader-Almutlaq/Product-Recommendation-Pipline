# Machine Learning Pipeline for Product Recommendation

## Table of Contents
- [Machine Learning Pipeline for Product Recommendation](#machine-learning-pipeline-for-product-recommendation)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Project Structure](#project-structure)
  - [Features](#features)
  - [Datasets](#datasets)
  - [Process](#process)
  - [Technologies Used](#technologies-used)
  - [Requirements](#requirements)
  - [Installation Instructions](#installation-instructions)
  - [Usage](#usage)
  - [Machine Learning Model](#machine-learning-model)
  - [Results](#results)
  - [Acknowledgment](#acknowledgment)
  - [License](#license)
 
---

## Project Overview
This project builds a machine learning pipeline to predict whether a customer will recommend a product based on their review. The dataset contains numerical, categorical, and textual features, requiring a comprehensive preprocessing pipeline.

---

## Project Structure
```
├── data/                 # Dataset files
├── starter.ipynb         # Jupyter notebook containing the pipeline
├── README.md             # Project documentation
```

---

## Features
- **End-to-End Pipeline**: Integrates data ingestion, preprocessing, feature engineering, model training, and evaluation.
- **Modular Design**: Separate pipelines for numerical, categorical, and text data ensure clarity and maintainability.
- **Custom Transformers**: Includes a transformer for computing text length and a spaCy-based lemmatizer for text normalization.
- **Advanced Text Processing**: Utilizes TF-IDF vectorization and spaCy for efficient text feature extraction.
- **Model Optimization**: Fine-tuning using RandomizedSearchCV to enhance the Random Forest model performance.
- **Reproducibility**: Consistent results ensured through proper train-test splitting and fixed random state settings.

---

## Datasets
The dataset has been preprocessed to remove missing values. It contains eight features and one target variable (`Recommended IND`).

- **Clothing ID**: Categorical identifier for the reviewed product.
- **Age**: Customer's age.
- **Title**: Review title.
- **Review Text**: Full review text.
- **Department Name**: Product department category.
- **Division Name**: Product division category.
- **Class Name**: Product class category.
- **Recommended IND**: Target variable (1 = Recommended, 0 = Not recommended).

---

## Process
1. **Data Loading**: Import and inspect dataset.
2. **Feature Engineering**:
   - Numerical feature preprocessing (imputation, scaling).
   - Categorical feature encoding (One-Hot Encoding).
   - Text processing (spaCy lemmatization, TF-IDF transformation).
3. **Pipeline Construction**: Combine feature pipelines.
4. **Model Training**: Train a Random Forest Classifier.
5. **Evaluation**: Assess model performance and fine-tune parameters.

---

## Technologies Used
- Python
- Jupyter Notebook
- Pandas
- NumPy
- scikit-learn
- spaCy

---

## Requirements
- Python 3.8+
- Required libraries (install using `requirements.txt` or manually):
  ```
  pandas
  numpy
  scikit-learn
  spacy
  ```

---

## Installation Instructions
1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd <project-directory>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download spaCy model:
   ```sh
   python -m spacy download en_core_web_sm
   ```

---

## Usage
Run the Jupyter Notebook to execute the pipeline:
```sh
jupyter notebook starter.ipynb
```

---

## Machine Learning Model
- **Algorithm**: Random Forest Classifier
- **Pipeline Components**:
  - Numerical pipeline (imputation, standard scaling)
  - Categorical pipeline (One-Hot Encoding)
  - Text processing pipeline (spaCy lemmatization, TF-IDF)
- **Fine-Tuning**: Performed using RandomizedSearchCV

---

## Results
The pipeline was evaluated using accuracy metrics. Fine-tuning improved performance over the baseline model. Key results include:
- **Baseline Accuracy**: 85%
- **Optimized Accuracy**: 85%

The optimization process was shortened due to limitations in processing power, restricting the extent of hyperparameter tuning and model refinement.

## Acknowledgment
This project is part of the Udacity Data Scientist Nanodegree program

## License
[License](./License.txt)



