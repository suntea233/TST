import pandas as pd
from sklearn.model_selection import train_test_split


polars = []
sents = []
data = pd.read_csv(r"C:\Attention\IMDB Dataset.csv",)


x_train,x_test,y_train,y_test = train_test_split(data['review'],data['sentiment'],test_size=0.2)

pd.DataFrame({"sentence":x_train,"label":y_train}).to_csv("train.csv",index=None)
pd.DataFrame({"sentence":x_test,"label":y_test}).to_csv("test.csv",index=None)
