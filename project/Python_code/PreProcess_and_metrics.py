import pandas as pd
from gensim.utils import simple_preprocess
from collections import Counter

# Preprocess
train = pd.read_csv('../data/train.csv')
train = train[train["Description"].str.contains("#NAME?") == False]
train.iloc[:, 2] = train.iloc[:, 2].apply(
    lambda x: ' '.join(simple_preprocess(x)))
train_list = train['Description'].apply(lambda x: len(str(x).split(' ')))
train['ClassIndex'] = train['ClassIndex'].astype(str)

test = pd.read_csv('../data/test.csv')
test = test[test["Description"].str.contains("#NAME?") == False]
test.iloc[:, 2] = test.iloc[:, 2].apply(
    lambda x: ' '.join(simple_preprocess(x)))
test_list = test['Description'].apply(lambda x: len(str(x).split(' ')))
test['ClassIndex'] = test['ClassIndex'].astype(str)


# Count
print("Maximum amunt of words:", max(train_list))
print("Minimum amunt of words:", min(train_list))

print("Maximum amunt of words:", max(test_list))
print("Minimum amunt of words:", min(test_list))

# Train,test and validation split
train.to_csv('../data/train.csv', index=False)
test.to_csv('../data/test.csv', index=False)

# Additional metrics
result = pd.DataFrame([[]])
for col in train:
    result[col] = train[col].apply(len).mean()
print("Average character length train: \n",result)

result = pd.DataFrame([[]])
for col in test:
    result[col] = test[col].apply(len).mean()
print("Average character length train: \n",result)

# Print results
print("Train length: ", len(train))
print("Average character length: ",train["Description"].apply(len).mean())
print()
print("Test length: ", len(test))
print("Average character length test: ",test["Description"].apply(len).mean())

