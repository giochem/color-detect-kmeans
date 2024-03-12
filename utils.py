from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import cv2

# the closest rgb for main rgb image
def closest_rgb(main_rgb, color_detect):
    r, g, b = main_rgb
    color_diffs = []
    for color, color_name in color_detect:
        cr, cg, cb = color
        color_diff = (r - cr)**2 + (g - cg)**2 + (b - cb)**2
        color_diffs.append((color_diff, color, color_name))
    return min(color_diffs)[1:]

def find_picture_colors(image_array, min_cluter, max_cluter):
    best_silhouette = -1
    best_clt = None
    for clusters in range(min_cluter, max_cluter):
        # Cluster colours
        clt = KMeans(n_clusters = clusters)
        clt.fit(image_array)
        unique = np.unique(clt.labels_)
        # Number of distinct clusters (n) found smaller than n_clusters (>n) -> get best result clt
        if len(unique) < clusters:
            if best_silhouette == -1:
                best_clt = clt
            break
        # Validate clustering result
        silhouette = metrics.silhouette_score(image_array, clt.labels_, metric='euclidean')
        # Find the best one
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_clt = clt

    unique, counts = np.unique(best_clt.labels_, return_counts=True)
    palette = np.array(best_clt.cluster_centers_, dtype=np.uint8)
    
    return [best_silhouette, unique, counts, palette]






