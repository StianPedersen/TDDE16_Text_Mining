import csv
import fasttext
import pandas as pd

train = pd.read_csv('train.csv')[['Description', 'ClassIndex']].rename(
    columns={'Description': 'X', 'ClassIndex': 'Y'})

test = pd.read_csv('test.csv')[['Description', 'ClassIndex']].rename(
    columns={'Description': 'X', 'ClassIndex': 'Y'})

# Prefixing each  of the category column with '__label__'
train.iloc[:, 1] = train.iloc[:, 1].apply(lambda x: '__label__' + str(x))
test.iloc[:, 1] = test.iloc[:, 1].apply(lambda x: '__label__' + str(x))


# Saving the CSV file as a text file to train/test the classifier
train[['Y', 'X']].to_csv('../data/train.txt',
                         index=False,
                         sep=' ',
                         header=None,
                         quoting=csv.QUOTE_NONE,
                         quotechar="",
                         escapechar=" ")
test[['Y', 'X']].to_csv('../data/test.txt',
                        index=False,
                        sep=' ',
                        header=None,
                        quoting=csv.QUOTE_NONE,
                        quotechar="",
                        escapechar=" ")

# Training the fastText classifier
model = fasttext.train_supervised(
    'train.txt', autotuneValidationFile='../data/test.txt')


result = model.test('../data/test.txt')
print(result)
print("Attributes")
print("------------------")
args_obj = model.f.getArgs()
for hparam in dir(args_obj):
    if not hparam.startswith('__'):
        print(f"{hparam} -> {getattr(args_obj, hparam)}")
# (7600, 0.905921052631579, 0.905921052631579)
