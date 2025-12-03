
import zipfile

import numpy as np
from tensorflow.keras.preprocessing import image

def load_data(n=600, target_size = (200,200), test_split=0.2):
    with zipfile.ZipFile("data/faces.zip", 'r') as zip_ref:
        zip_ref.extractall("data/faces")
    
    path_smile="data/labels/SMILE_list.txt"
    path_nonSmile="data/labels/NON-SMILE_list.txt"

    try:
        with open(path_smile, 'r') as f:
            files_smile = []
            for line in f:
                line.strip()
                line=line[:-4]+"ppm"
                files_smile.append(line)
    except FileNotFoundError:
        print(f"Error: The file '{path_smile}' was not found.")

    try:
        with open(path_nonSmile, 'r') as f:
            files_nonSmile = []
            for line in f:
                line.strip()
                line=line[:-4]+"ppm"
                files_nonSmile.append(line)
    except FileNotFoundError:
        print(f"Error: The file '{path_nonSmile}' was not found.")

    rootpath = "data/faces/" 

    imgs_smile = [] 
    imgs_nonSmile = [] 

    for i in range(n):
        img2 = image.load_img(rootpath + files_smile[i], target_size=target_size)
        img2 = image.img_to_array(img2)
        img2 = img2/255.
        imgs_smile.append(img2)

        img = image.load_img(rootpath + files_nonSmile[i], target_size=target_size)
        img = image.img_to_array(img)
        img = img/255.
        imgs_nonSmile.append(img)

    imgs_smile=np.array(imgs_smile)
    imgs_nonSmile=np.array(imgs_nonSmile)

    # shuffle images
    np.random.seed(123)
    idxs = np.random.choice(n, size=n, replace=False)
    imgs_smile = imgs_smile[idxs,:,:,:]
    imgs_nonSmile = imgs_nonSmile[idxs,:,:,:]

    n_test = int(n*test_split)

    train_smile = imgs_smile[n_test:,:,:,:]
    train_nonSmile = imgs_nonSmile[n_test:,:,:,:]

    test_smile = imgs_smile[0:n_test,:,:,:]
    test_nonSmile = imgs_nonSmile[0:n_test,:,:,:]

    return n_test,train_smile,train_nonSmile,test_smile,test_nonSmile