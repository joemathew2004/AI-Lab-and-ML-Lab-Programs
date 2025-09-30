import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data[:, :4]
y = iris.target

pca = PCA(n_components=2)
col = pca.fit_transform(X)
df = pd.DataFrame(col, columns=['PC1', 'PC2'])
df['target'] = y
print(df)

for i in range(3):
    plt.scatter(df.loc[y==i, 'PC1'], df.loc[y==i, 'PC2'], label=iris.target_names[i])

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Iris Dataset with 2 Features')
plt.legend(title='Species')
plt.show()