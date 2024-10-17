from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

# Load the SMS Spam Collection dataset
df = pd.read_csv(r"C:\Datasets\sms+spam+collection\SMSSpamCollection", sep="\t", names=["label", "message"])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2, random_state=42)

# Define the character n-gram vectorizer
vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 3))  
# print(type(vectorizer))

# Transform training and testing data into character n-gram features
x_train_features = vectorizer.fit_transform(x_train)
x_test_features = vectorizer.transform(x_test)

# Train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(x_train_features, y_train)

# Predict on the testing data
y_pred = model.predict(x_test_features)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Calculate confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_matrix)

true_positive = confusion_matrix[1, 1]
false_positive = confusion_matrix[0, 1]
false_negative = confusion_matrix[1, 0]

precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1 = 2 * (precision * recall) / (precision + recall)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
