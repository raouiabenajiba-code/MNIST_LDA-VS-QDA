import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.datasets import make_blobs

# 1. Génération de données fictives (2 groupes avec des variances différentes)
X, y = make_blobs(n_samples=300, centers=2, n_features=2, cluster_std=[1.0, 3.0], random_state=42)

# 2. Définition et entraînement des modèles
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

lda.fit(X, y)
qda.fit(X, y)

# 3. Fonction pour tracer les frontières de décision
def plot_boundaries(model, ax, title):
    # Création d'une grille pour le dessin
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Affichage des zones et des points
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm', s=30)
    ax.set_title(title)
    ax.set_xlabel('Caractéristique 1')
    ax.set_ylabel('Caractéristique 2')

# 4. Affichage des résultats
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

plot_boundaries(lda, ax1, "LDA (Frontière Linéaire)")
plot_boundaries(qda, ax2, "QDA (Frontière Quadratique)")

plt.tight_layout()
plt.show()