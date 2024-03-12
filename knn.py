from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import cv2
from utils import closest_rgb, find_picture_colors

COLOR_DETECT = [                                                                                                                    
    [(0, 0, 0),              "Black"],
    [(127, 127, 127),        "Gray"],
    [(136, 0, 21),           "Bordeaux"],
    [(237, 28, 36),          "red"],
    [(255, 127, 39),         "orange"],
    [(255, 242, 0),          "yellow"],
    [(34, 177, 76),          "green"],
    [(203, 228, 253),        "blue"],
    [(0, 162, 232),          "dark blue"],
    [(63, 72, 204),          "purple"],
    [(255, 255, 255),        "white"],
    [(195, 195, 195),        "light gray"],
    [(185, 122, 87),         "light brown"],
    [(255, 174, 201),        "light pink"],
    [(255, 201, 14),         "dark yellow"],
    [(239, 228, 176),        "light yellow"],
    [(181, 230, 29),         "light green"],
    [(153, 217, 234),        "light blue"],
    [(112, 146, 190),        "dark blue"],
    [(200, 191, 231),        "light purple"]
]
IMAGE_PATHS = [
               "./test/black_car.jpg",
               "./test/black.jpg",
               "./test/green.png",
               "./test/light purple.png",
               "./test/orange.webp",
               "./test/red.png",
               "./test/white.jpg",
               "./test/yellow.jpg"]
MAX_SIZE = 50 # reduce size image to increate speed handle
# normal image have different 4 - 6 colors
MIN_CLUSTER = 2 # min clusters must >=2
MAX_CLUSTER = 10 # recomment for classify (little different color)

# Load images
for image_path in IMAGE_PATHS:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize it
    h, w, _ = image.shape
    w_new = int(MAX_SIZE * w / max(w, h) )
    h_new = int(MAX_SIZE * h / max(w, h) )
    # print(w,h,w_new,h_new)
    image = cv2.resize(image, (w_new, h_new))

    # Reshape the image to be a list of pixels
    image_array = image.reshape((image.shape[0] * image.shape[1], 3))
    
    best_silhouette, unique, counts, palette = find_picture_colors(image_array, MIN_CLUSTER, MAX_CLUSTER) 
    
    idx = np.unravel_index(counts.argmax(), counts.shape)
    main_rgb = palette[idx]
    the_closest_rgb, color_name = closest_rgb(main_rgb, COLOR_DETECT)
    
    print(image_path)
    print("The best Silhouette(KNN): ", best_silhouette)
    print("The labels are: ", unique)
    print("Count of  items: ", counts)
    print("Value rbg for each items: ")
    print(palette)
    print("Main rgb: ", main_rgb) 
    print("The closest rgb: ", the_closest_rgb) 
    print("Color name: ", color_name)