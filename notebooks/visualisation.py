# Imports 
import numpy as np
from scipy import linalg
# ML
import torch
from torchvision import datasets, transforms

# Plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import gridspec, colors
from matplotlib.widgets import TextBox
# Colourblind colourschemes
from colour_schemes import tol_cset
colours = list(tol_cset("bright"))
cmap = colors.ListedColormap(colours[:-2])

# Data
import pandas as pd

# Projection
from sklearn.decomposition import PCA
from umap import UMAP

# Clustering
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

#~~~~~~~~~~~~~~~~~~~~~~~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Projection - combined both options into a single one for ease of use
def projection(manuscript_vectors,method="umap"):
    """
    A low-dimensional (2D) projection of the latent feature vectors for each manuscript image. 

    Parameters
    ----------
    manuscript_vectors  (torch.tensor) (N,m): Embeddings for N manuscript images produced by ResNet.
    method              (string)            : Options for projection (umap or pca). 
    """
    if method.lower()=="umap":
        umap = UMAP(min_dist=0.5,n_neighbors=15)
        x,y = umap.fit_transform(manuscript_vectors).T
    elif method.lower()=="pca":
        pca2 = PCA()
        pca_transformed = pca2.fit_transform(manuscript_vectors).T
        x,y = pca_transformed[0],pca_transformed[1]
    else:
        raise RuntimeError("Only supported method options are 'pca' or 'umap'.")
    return x,y

def find_opt_gmm(X,min_clusters=5,n_max=15):
    """
    Find the optimal gaussian mixture model using information theoretic measure (BIC). 
    """
    scores = np.zeros(n_max-min_clusters+1)
    models = {}
    for i,n in enumerate(range(min_clusters,n_max+1)):
        gmm = GaussianMixture(n_components=n,n_init=min_clusters)
        gmm.fit(X)
        scores[i] = gmm.bic(X)
        models[i] = gmm
    return models[np.argmin(scores)]

def cluster(x,y,method="Gaussian mixture",min_clusters=5):
    """
    Cluster the (projected into 2d) data. 

    Parameters
    ----------
    x               (ndarray (N,))  : First embedding vector for data.
    y               (ndarray (N,))  : Second embedding vector for data.
    method          (string)        : Method of clustering (gaussian mixture or bayesian gaussian mixture).
    min_clusters    (int)           : Minimum number of clusters to return - only used by the gaussian mixture model. 
    """
    X = np.c_[x,y]
    if method.lower()=="bayesian gaussian mixture" or method.lower()=="pick cluster number":
        cluster_model = BayesianGaussianMixture(n_components=15, covariance_type="full",n_init=6,weight_concentration_prior=0.8).fit(X)
    elif method.lower()=="gaussian mixture" or method.lower()=="fix cluster number":
        cluster_model = find_opt_gmm(X,min_clusters=min_clusters)
    else:
        raise RuntimeError("Only supported method options are 'gaussian mixture'/'pick cluster number' or 'bayesian gaussian mixture'/'fix cluster number'.")
    return cluster_model

def gen_faux_metadata(manuscript_labels):
    """
    Generate some trial metadata to show how the keyword -> highlighted cluster tool could work.
    """
    rng = np.random.default_rng()
    tags = ["monk","astrology","fauna","flora","human","diagram","cosmology"]
    probs = {k:rng.random(len(tags)) for k in set(manuscript_labels)}
    trial_meta_data = [rng.choice(tags,rng.integers(0,len(tags)),p=probs[label_]/probs[label_].sum()) for label_ in manuscript_labels]
    return trial_meta_data

def display_visualisation(x,y,images,manuscript_database,cluster_model,prototype_fts=False):
    """
    This plot displays the visualisation using matplotlib widgets. 
    Absolutely must include '%matplotlib widget' somewhere at top of notebook. 

    Parameters
    ----------
    x                   (ndarray (N,))                  : First embedding vector for data.
    y                   (ndarray (N,))                  : Second embedding vector for data.
    images              (torch.tensor, (N,3,size,size)) : Tensor representing all the images. 
    manuscript_database (pd.dataframe)                  : Dataframe containing info on the manuscript (image tensors, embeddings, labels, pages)
    prototype_fts       (bool)                          : Whether to show prototype features (e.g. searchbox)
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~ FIGURE SETUP AND BASIC SCATTER PLOT ~~~~~~~~~~~~~~~~~~~
    # Figure
    gs = gridspec.GridSpec(5,10)
    fig = plt.figure(figsize=(10,5))
    rax = fig.add_subplot(gs[:,5:]) # Right axis - for image
    if prototype_fts:
        lax = fig.add_subplot(gs[:-1,:5]) # Left axis - for scatterplot
        sax = fig.add_subplot(gs[-1,2:5]) # searchbox axis
    else:
        lax = fig.add_subplot(gs[:,:5]) # Left axis - for scatterplot
        sax = None
    # Scatter plot for LHS
    scatter = lax.scatter(x,y,c=manuscript_database.loc[:,"label"].values,s=8,cmap=cmap,zorder=50)
    # Make it look nicer
    lax.set_xlabel("Embedding 1")
    lax.set_ylabel("Embedding 2")
    lax.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
    # Legend
    leg = lax.legend(scatter.legend_elements()[0],manuscript_database.loc[:,"manuscript"].unique(),
                    loc="lower center",
                    bbox_to_anchor=(0.5,1.01),
                    ncol=2,
                    frameon=False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CLUSTERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Cluster calcs
    clusters = cluster_model.predict(np.c_[x,y])
    cluster_probs = cluster_model.predict_proba(np.c_[x,y])
    # Plot clusters
    cluster_perims = {}
    for i, (mean, covar) in enumerate(zip(cluster_model.means_, cluster_model.covariances_)):
        v, w = linalg.eigh(covar)
        v = 2.5 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(clusters == i):
            continue

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, linewidth=1.5,fill=False,color='k',zorder=20)
        ell.set_clip_box(lax.bbox)
        cluster_perims[i] = lax.add_artist(ell)

    img = rax.imshow(images[0].permute(1, 2, 0))
    # Plot an image
    def plot_img_on_rax(ind):
        img.set_data(images[ind].permute(1, 2, 0))
        rax.set_visible(True)
        rax.set_title("{} p.{}".format(manuscript_database.loc[ind,"manuscript"],manuscript_database.loc[ind,"page"]))
    plot_img_on_rax(np.random.randint(manuscript_database.shape[0]))
    rax.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
    fig.tight_layout()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INTERACTIVITY ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Mouseover functionality
    def hover(event):
        # if the mouse is over the scatter points
        if scatter.contains(event)[0]:
            # find out the index within the array from the event
            ind, = scatter.contains(event)[1]["ind"]
            # Check whether cursor overlaps with two or more scatter points. 
            if hasattr(ind,"__len__"):
                mx,my = event.xdata, event.ydata
                dists = (x[ind]-mx)**2+(y[ind]-my)**2
                ind = ind[np.argmin(dists)]
            # Show view and make it visible
            plot_img_on_rax(ind)
        # else:
        #     #if the mouse is not over a scatter point
        #     rax.set_visible(False)
        fig.canvas.draw_idle()
    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)    
    
    # PROTOTYPE FEATURES
    if prototype_fts:
        # Textbox functionality
        trial_meta_data = gen_faux_metadata()
        def submit(search_string):
            occurence_freqs = (np.array([search_string.lower() in words for words in trial_meta_data]) @ cluster_probs)/cluster_probs.sum(0)
            wts = occurence_freqs/occurence_freqs.max()
            for i,ellipse in cluster_perims.items():
                ellipse.set(edgecolor=cm.gray_r(wts[i]))

        text_box = TextBox(sax,"Search keyword:  ")
        text_box.on_submit(submit)

    plt.show()
    return fig,(lax,rax,sax)

