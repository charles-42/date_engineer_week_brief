import pandas as pd
import sqlite3

connection = sqlite3.connect("olist.db")

df = pd.read_sql_query("SELECT * FROM CleanDataset",connection)

# Creation de la variable score
df['score'] = df['review_score'].apply(lambda x : 1 if x==5 else 0)


df.to_sql('TrainingDataset', connection, index=False, if_exists='replace')

print("Table TrainingDataset mise à jour")

connection.close()