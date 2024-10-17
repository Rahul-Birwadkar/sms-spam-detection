# sms-spam-detection
Implements SMS spam detection using Multinomial Naive Bayes classifier on the SMS Spam Collection dataset. Utilizes bi-grams and tri-grams for feature extraction. Evaluates classification performance with accuracy, precision, recall, and F1 score using Python and Scikit-learn.

## Project Overview

The project involves:
- Training a **Multinomial Naive Bayes** model on the SMS Spam Collection dataset.
- Extracting features using **N-gram vectorization** (bi-grams and tri-grams).
- Evaluating model performance using:
  - Confusion Matrix
  - Precision
  - Recall
  - F1 Score

## Technologies Used

- **Python** 
- **Scikit-learn**: For the Multinomial Naive Bayes classifier and evaluation metrics
- **Pandas**: For dataset manipulation
- **Numpy**: For numerical computations
- **VS Code IDE**

## Dataset

The dataset used is the **SMS Spam Collection dataset**, which contains a collection of 5,574 SMS messages tagged as spam or non-spam (ham).

- **Download Link (SMS Spam Dataset):** [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
