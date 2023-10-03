import numpy as np
from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from tensorflow.keras.utils import to_categorical
from transformers import TFRobertaModel,AutoTokenizer



def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df

df_test = pd.read_csv('../data/test.csv')
tokenizer = AutoTokenizer.from_pretrained("../saved_model/roberta-large")

for i in df_test.index:
  if df_test['ClassIndex'][i] == 1:
    df_test.at[i, 'ClassIndex'] = 0
  elif df_test['ClassIndex'][i] == 2:
    df_test.at[i, 'ClassIndex'] = 1
  elif df_test['ClassIndex'][i] == 3:
    df_test.at[i,'ClassIndex'] = 2
  elif df_test['ClassIndex'][i] == 4:
    df_test.at[i, 'ClassIndex'] = 3
df_test.drop('Title', inplace=True, axis=1)
df_test = swap_columns(df_test,'Description','ClassIndex')

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

model = keras.models.load_model("my_model")
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
