import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model
import keras
import logging
import pickle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_dataset(dataset_path):
    with open(dataset_path, 'rb') as file:
        images_dataset = pickle.load(file)
    
    desired_shape = (256, 256, 1)

    X, y = zip(*images_dataset)
    X = np.array(X).reshape(-1, desired_shape[0], desired_shape[1], desired_shape[2])
    y = np.array(y)    

    return X, y

def get_dense_layer(model_path):
    model = keras.models.load_model(model_path)

    dense_128_model = Model(inputs=model.input, outputs=model.get_layer('dense_layer').output)

    return dense_128_model

logging.basicConfig(encoding='utf-8', level=logging.INFO)


dense_layer = get_dense_layer('BaseMRI_CNN.keras')
logging.info("loaded dense layer")
df = pd.read_csv("../../participants.tsv", sep='\t')

dense_results = []
metadata_results = []

logging.info("loaded df")
for i in range(1,43):
    formatted_number = "{:03d}".format(i)

    X , y = load_dataset(f'dataset_div/{formatted_number}/{formatted_number}.pkl')
    logging.info(f"loaded {formatted_number} dataset")

    outT1 = dense_layer.predict(np.array([X[0]]))
    outT2 = dense_layer.predict(np.array([X[1]]))

    # T1T2 Delta calculation
    dense_layer_output = np.abs(outT1 - outT2)

    number_df = df[df["participant_id"] == "sub-"+ formatted_number]
    dense_results.append(dense_layer_output[0])
    metadata_results.append((number_df.values[0]))



metadata_columns = ["participant_id", "gender", "age", "tumor_type", "ki67", "grade", "idh1", "del19q"]

dense_df = pd.DataFrame(dense_results)

# Dimentionality Reduction
pca = PCA(n_components=42)
reduced_dense_output = pca.fit_transform(dense_df)
dense_df = pd.DataFrame(reduced_dense_output)

metadata_df = pd.DataFrame(metadata_results, columns=metadata_columns)
# Combining Features

df_combined = pd.concat([dense_df, metadata_df], axis=1)
df_combined = df_combined.drop(columns=['participant_id', 'tumor_type'])

df_combined['gender'] = df_combined['gender'].replace({'male': 0, 'female': 1})
df_combined['idh1'] = df_combined['idh1'].replace({'mutant': 1, 'wild': 0})
df_combined['del19q'] = df_combined['del19q'].replace({'yes': 1, 'no': 0})
df_combined.columns = df_combined.columns.astype(str)

print(df_combined)
X = df_combined.drop('grade', axis=1)
y = df_combined['grade']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ensemble_model = RandomForestClassifier(n_estimators=100)

ensemble_model.fit(X_train, y_train)

y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')