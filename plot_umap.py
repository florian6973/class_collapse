import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

csv = pd.read_csv('umap.csv')
# plt.scatter(csv['UMAP1'], csv['UMAP2'], c=csv['superclass'])
sns.scatterplot(x='UMAP1', y='UMAP2', hue='superclass', data=csv)
plt.show(block=True)