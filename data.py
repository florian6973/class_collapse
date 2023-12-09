from datasets import load_dataset
import pandas as pd
# dataset = load_dataset('maharshipandya/spotify-tracks-dataset', split='train')
# # export the dataset to a csv file
# # dataset.to_csv('spotify.csv')
# df = dataset.to_pandas()
df= pd.read_csv('spotify.csv')
df.drop(['Unnamed: 0', 'track_id'], axis=1, inplace=True)
df.drop(['artists','album_name','track_name'], axis=1, inplace=True)
print(len(df))
print(len(df['track_genre'].unique()))
print(df['track_genre'].unique())
print(df.columns)

# read json
import json
with open('dc.json') as f:
    data = json.load(f)
    dico_genres = {}
    for genre in data:
        for sous_genre in genre['genres']:
            dico_genres[sous_genre] = genre['name']
print(dico_genres)


df['superclass'] = df['track_genre'].map(dico_genres)
print(df['superclass'].isna().sum())
print(df[pd.isna(df['superclass'])]['track_genre'].unique())
print(df.groupby('superclass').count())

df.to_csv('spotifyWC.csv')

# split train and test data
from sklearn.model_selection import train_test_split
train, test, preds_train, preds_test = train_test_split(df.drop(['track_genre', 'superclass'], axis=1), df["track_genre"], test_size=0.005, random_state=42, stratify=df['track_genre'])
print(train.shape)
print(test.shape)
print(preds_train.isna().sum())

# train a knn model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

# # encode the target variable
# le = LabelEncoder()
# train['track_genre'] = le.fit_transform(train['track_genre'])
# test['track_genre'] = le.transform(test['track_genre'])

# preds_train_all = train['track_genre']
# preds_test_all = test['track_genre']
# preds_train = train['superclass']
# preds_test = test['superclass']

print(preds_test.unique())
print(preds_test.values)

# train.drop(['track_genre', 'superclass'], axis=1, inplace=True)
# test.drop(['track_genre', 'superclass'], axis=1, inplace=True)

# normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# train = scaler.fit_transform(train)
# test = scaler.transform(test)
test= scaler.fit_transform(test)


import pandas as pd
import umap # pip install umap-learn
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your pandas DataFrame
# Make sure to replace 'your_column1' and 'your_column2' with the actual column names you want to visualize

# Standardize the data (optional but can be beneficial)
# data = (data - data.mean()) / data.std()

# Create a UMAP model
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)

# Fit the UMAP model to your data
print("Computing UMAP projection...")
embedding  = umap_model.fit_transform(test)
print("UMAP projection complete.")

# Create a new DataFrame with the UMAP coordinates
umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
umap_df['track_genre'] = preds_test.values
print(umap_df.info())

umap_df.to_csv('umap.csv')
# Plot the UMAP
# plt.figure(figsize=(10, 8))
# sns.scatterplot(x='UMAP1', y='UMAP2', data=umap_df, hue=preds_test, palette='Spectral', legend='full')
# plt.title('UMAP Projection of Data')
# plt.show()
sns.scatterplot(
    x='UMAP1', y='UMAP2', data=umap_df,
    hue='track_genre',
    palette='Spectral',
    legend='full',
)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Penguin dataset', fontsize=24)
plt.show()
exit()

# train a knn model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train, preds_train)

# make predictions on the test set
preds = knn.predict(test)
print(accuracy_score(preds_test, preds))
