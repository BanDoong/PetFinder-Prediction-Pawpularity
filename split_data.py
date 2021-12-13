from sklearn.model_selection import train_test_split
import os
import pandas as pd

dataset = pd.read_csv('./petfinder-pawpularity-score/train.csv')

data = dataset['Id']
target = dataset['Pawpularity']

x_train, x_valid, y_train, y_valid = train_test_split(data, target, test_size=0.2, shuffle=True, random_state=32)

train_save_path = './petfinder-pawpularity-score/train_label2.csv'
val_save_path = './petfinder-pawpularity-score/val_label2.csv'

train_df = pd.DataFrame({'Id': x_train, 'Pawpularity': y_train})
val_df = pd.DataFrame({'Id': x_valid, 'Pawpularity': y_valid})

train_df.to_csv(train_save_path)
val_df.to_csv(val_save_path)
