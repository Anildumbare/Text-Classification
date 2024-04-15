# Movie Overviews and Genres Text Classification

## Overview
This project aims to classify movie overviews based on their genres using natural language processing (NLP) techniques and machine learning algorithms. The goal is to accurately predict the genre(s) of a movie based on its textual description.

## Dataset
The dataset used for this project consists of movie overviews paired with their respective genres. Each overview is a textual description of the movie, and genres are categorical labels such as action, comedy, drama, etc.

## NLP Pipeline
The NLP pipeline involves several preprocessing steps to prepare the text data for classification:
- Tokenization: Breaking down text into individual tokens (words or subwords).
- Stopword Removal: Eliminating common words that do not carry significant meaning.
- Lemmatization/Stemming: Reducing words to their base forms to normalize the text.
- Vectorization: Converting text into numerical features using techniques like TF-IDF or word embeddings.

## Machine Learning Algorithms
The following machine learning algorithms are used for classification:
1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. Multinomial Naive Bayes

## Evaluation
The accuracy of the classification models is evaluated using standard evaluation metrics such as accuracy, precision, recall, and F1-score. The dataset is split into training and testing sets to assess the performance of each algorithm.

## Usage
1. **Data Preparation:** Ensure the dataset containing movie overviews and genres is available.
2. **Preprocessing:** Run the NLP pipeline to preprocess the text data.
3. **Model Training:** Train the classification models using the preprocessed data.
4. **Evaluation:** Evaluate the models using the testing set and calculate performance metrics.
5. **Accuracy Comparison:** Compare the accuracy of different algorithms to determine the most effective approach for genre classification.

## Dependencies
- Python 3.x
- Libraries: nltk, scikit-learn, pandas, numpy

## Files Included
- `README.md`: Overview and instructions for the project.
- `data.csv`: Dataset containing movie overviews and genres.
- `text_classification.ipynb`: Jupyter notebook containing the NLP pipeline, model training, and evaluation code.
- `requirements.txt`: List of dependencies required to run the project.

## Results
The classification results for each algorithm are as follows:
- Logistic Regression: Accuracy = 100%, Precision = 1%, Recall = 1%, F1-score = 1%
- SVM: Accuracy = 100%, Precision = 1%, Recall = 1%, F1-score = 1%
- Random Forest: Accuracy = 100%, Precision = 1%, Recall = 1%, F1-score = 1%
- Multinomial Naive Bayes: Accuracy = 100%, Precision = 1%, Recall = 1%, F1-score = 1%

## Conclusion
Based on the evaluation results, the most effective algorithm for movie genre classification is [Algorithm Name] with an accuracy of XX%. This project demonstrates the use of NLP techniques and machine learning algorithms for text classification tasks.
