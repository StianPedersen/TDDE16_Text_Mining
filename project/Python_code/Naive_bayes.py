from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import pandas as pd

# Read train/test CSV file
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# What to train on
train_X, train_Y = train['Description'], train['ClassIndex']

# Train model in pipeline
pipe = Pipeline([('scaler', TfidfVectorizer()), ('clf', MultinomialNB())])
pipe.fit(train_X, train_Y)

# Testing
print(classification_report(test['ClassIndex'], pipe.predict(
    test['Description']), labels=test['ClassIndex'].unique()))
