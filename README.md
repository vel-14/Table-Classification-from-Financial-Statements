# Table-Classification-from-Financial-Statements

## Project Overview

This project involves the processing and classification of text data extracted from HTML files. The main steps include parsing HTML, feature extraction, encoding, word embedding, handling class imbalance, model training, prediction, and visualization.

## Project Steps

### Parsing HTML Files:
Using BeautifulSoup, HTML files are parsed to extract features.
Extracted features are stored in a structured format.

### DataFrame Creation:
Features and class names are used to create a pandas DataFrame.

### Label Encoding:
Class names are encoded using LabelEncoder to convert them into numerical values.

### Word Embedding with BERT:
Extracted feature text is transformed using a BERT vectorizer.
BERT model is used to perform word embedding, converting text into numerical vectors.

### Handling Class Imbalance:
SMOTE (Synthetic Minority Over-sampling Technique) is applied to eliminate imbalance in the dataset.

### Dataset Splitting:
The dataset is split into training and testing sets.

### Model Training:
Logistic Regression model is trained using the training dataset.
The trained model achieved 99% accuracy on the training dataset.

### Model Prediction:
The trained model is used to predict the class labels on the test dataset.
The model achieved 92% accuracy on the test dataset.

### Model Saving:
The trained model is saved using pickle for future use.

### Visualization:
Word embeddings are plotted on a 2D plane using K-Means clustering for visualization.

