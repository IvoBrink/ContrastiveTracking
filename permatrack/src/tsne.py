from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch
from matplotlib import ticker


X = torch.cat(torch.load('post_convGRU_mot17_with_contrastive'))
X = torch.flatten(X, start_dim=1)

X_embedded = TSNE(perplexity=2).fit_transform(X)


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T

    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, None)
    plt.show()

#TODO: num samples is normally used for TSNE, but we have persons which is not first dimension?
plot_2d(X_embedded, ['r', 'g','b'], "T-distributed Stochastic  \n Neighbor Embedding")
print(X_embedded.shape)
