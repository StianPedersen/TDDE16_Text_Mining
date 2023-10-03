import pandas as pd


import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal

from tensorflow.keras.losses import CategoricalCrossentropy

from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense

from tensorflow.keras.utils import to_categorical
from transformers import TFRobertaModel,AutoTokenizer

import numpy as np

from sklearn.metrics import classification_report

len = 181
def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df

df_train = pd.read_csv('../data/train.csv')


df_test = pd.read_csv('../data/test.csv')

df_train['ClassIndex'] = df_train['ClassIndex'].astype('object')
for i in df_train.index:
  if df_train['ClassIndex'][i] == 1:
    df_train.at[i, 'ClassIndex'] = 0
  elif df_train['ClassIndex'][i] == 2:
    df_train.at[i, 'ClassIndex'] = 1
  elif df_train['ClassIndex'][i] == 3:
    df_train.at[i,'ClassIndex'] = 2
  elif df_train['ClassIndex'][i] == 4:
    df_train.at[i, 'ClassIndex'] = 3
df_test['ClassIndex'] = df_test['ClassIndex'].astype('object')

for i in df_test.index:
  if df_test['ClassIndex'][i] == 1:
    df_test.at[i, 'ClassIndex'] = 0
  elif df_test['ClassIndex'][i] == 2:
    df_test.at[i, 'ClassIndex'] = 1
  elif df_test['ClassIndex'][i] == 3:
    df_test.at[i,'ClassIndex'] = 2
  elif df_test['ClassIndex'][i] == 4:
    df_test.at[i, 'ClassIndex'] = 3

df_train.drop('Title', inplace=True, axis=1)
df_test.drop('Title', inplace=True, axis=1)

df_train = swap_columns(df_train,'Description','ClassIndex')
df_test = swap_columns(df_test,'Description','ClassIndex')
y_train = to_categorical(df_train.ClassIndex)
y_test = to_categorical(df_test.ClassIndex)

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
bert = TFRobertaModel.from_pretrained('roberta-large')

x_train = tokenizer(
    text=df_train.Description.tolist(),
    add_special_tokens=True,
    max_length=len,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)
x_test = tokenizer(
    text=df_test.Description.tolist(),
    add_special_tokens=True,
    max_length=len,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)


input_ids = x_train['input_ids']
attention_mask = x_train['attention_mask']

input_ids = Input(shape=(len,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(len,), dtype=tf.int32, name="attention_mask")
embeddings = bert(input_ids,attention_mask = input_mask)[0] 
out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
out = Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(32,activation = 'relu')(out)
y = Dense(4,activation = 'sigmoid')(out)
model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
model.layers[2].trainable = True

optimizer = Adam(
    learning_rate=3e-05, 
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)


loss =CategoricalCrossentropy(from_logits = True)
metric = CategoricalAccuracy('balanced_accuracy'),

model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)

tf.test.gpu_device_name()

train_history = model.fit(
    x ={'input_ids':x_train['input_ids'],'attention_mask':x_train['attention_mask']} ,
    y = y_train,
    validation_data = ({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']}, y_test),
  epochs=3,
    batch_size=5
)
model.save('../saved_model/my_model')

predicted_raw = model.predict({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']})
y_predicted = np.argmax(predicted_raw, axis = 1)
y_true = df_test.ClassIndex

print(classification_report(list(y_true), y_predicted))
#               precision    recall  f1-score   support

#            0       0.94      0.91      0.92      3053
#            1       0.96      0.99      0.97      2963
#            2       0.89      0.89      0.89      2993
#            3       0.90      0.91      0.90      2991

#     accuracy                           0.92     12000
#    macro avg       0.92      0.92      0.92     12000
# weighted avg       0.92      0.92      0.92     12000
