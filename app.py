from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from KMC import KMC


# Load Data
data = load_digits().data
pca = PCA(2)

# Transform the data
df = pca.fit_transform(data)

# Applying our function
K = KMC(df, 10)
K.k_means('euclidean', 1000)
K.visualize()
