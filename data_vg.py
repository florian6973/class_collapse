# train regression? improvement?

file = r"vg.csv"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print('Reading csv...')
df = pd.read_csv(file)
# df = pd.read_parquet('avion.parquet')
print('Done')
print(df.shape)
print(df.columns)
print(df['esrb_rating'].unique())

import json
with open('cqt2.json') as f:
    data = json.load(f)
    dico_genres = {}
    for genre in data.keys():
        for sous_genre in data[genre]:
            dico_genres[sous_genre] = genre
        # dico_genres[data[genre]] = genre

df['cat'] = df['esrb_rating'].map(dico_genres)

print(df['cat'].isna().sum())
print(df['cat'].unique())

print(df.groupby('cat').count())
print(df.groupby('esrb_rating').count().sort_values(by='esrb_rating', ascending=False)[10:])
print(df.head())
# csv.to_parquet('avion.parquet')

df_s = df.sample(frac=0.9, random_state=42)

# df_s = df_s[df_s['ProtocolName'].isin(['GOOGLE', 'AMAZON', 'HTTP'])]

preds= df_s['esrb_rating']

# keep only numerical columns
df_s = df_s.select_dtypes(include=np.number)
print(df_s.shape)

import pandas as pd
import umap # pip install umap-learn
import matplotlib.pyplot as plt
import seaborn as sns

# PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(df_s)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
print(pca.components_)
print(pca.n_components_)
print(pca.n_features_)
print(pca.n_samples_)
print(pca.noise_variance_)
print(pca.get_params())
print(pca.score(df_s))

tr = pca.transform(df_s)

sns.scatterplot(
    x=tr[:,0], y=tr[:,1], data=df_s,
    hue=preds,
    legend='full',
)
plt.gca().set_aspect('equal', 'datalim')
plt.title('PCA projection of the Penguin dataset', fontsize=24)
plt.show()

# Assuming df is your pandas DataFrame
# Make sure to replace 'your_column1' and 'your_column2' with the actual column names you want to visualize

# Standardize the data (optional but can be beneficial)
# data = (data - data.mean()) / data.std()

# Create a UMAP model
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)

# Fit the UMAP model to your data
print("Computing UMAP projection...")
embedding  = umap_model.fit_transform(df_s)
print("UMAP projection complete.")

# Create a new DataFrame with the UMAP coordinates
umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
umap_df['track_genre'] = preds.values
print(umap_df.info())

umap_df.to_csv('umap2.csv')
# Plot the UMAP
# plt.figure(figsize=(10, 8))
# sns.scatterplot(x='UMAP1', y='UMAP2', data=umap_df, hue=preds_test, palette='Spectral', legend='full')
# plt.title('UMAP Projection of Data')
# plt.show()
sns.scatterplot(
    x='UMAP1', y='UMAP2', data=umap_df,
    hue='track_genre',
    legend='full',
)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Penguin dataset', fontsize=24)
plt.show()