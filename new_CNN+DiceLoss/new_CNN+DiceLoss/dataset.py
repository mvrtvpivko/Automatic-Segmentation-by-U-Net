# zarzadzanie plikami
import os
#obliczenia numeryczne
import numpy as np
#wyswietlanie plikow/obrazow etc
import matplotlib.pyplot as plt
#zarzadzanie obrazami 
import cv2
# transformacje
import albumentations as A
import random
import matplotlib

matplotlib.use('TkAgg',force=True)

def get_filepaths(image_directory, mask_directory):
    # ma zwrocic słownik gdzie kluczem bedzie sciezka do obrazu a wartoscia bedzie sciezka do maski
    filepaths = {}

    image_files = os.listdir(image_directory)

    for image_file in image_files:

        image_filepath = os.path.join(image_directory, image_file)
        filename, extension = os.path.splitext(image_file)
        mask_filename = f"{filename}_mask{extension}"
        mask_filepath = os.path.join(mask_directory, mask_filename)

        if os.path.exists(image_filepath) is False:
            print("Image file does not exist")
    
        elif os.path.exists(mask_filepath) is False:
            print("Mask file does not exist")

        filepaths[image_filepath] = mask_filepath
    
    return filepaths


def visualize(filepaths):

    index = random.randint(0, len(filepaths))

    image_filepath = list(filepaths.keys())[index]
    mask_filepath = list(filepaths.values())[index]

    image = cv2.imread(image_filepath)
    mask  = cv2.imread(mask_filepath)


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,9))

    axes[0].imshow(image)
    axes[1].imshow(mask)

    plt.show()


def encode_mask(mask):
    return np.where(mask !=0, 1, 0)

def get_data(image_path, mask_path):
    
    filepaths = get_filepaths(image_path, mask_path)
    #visualize(filepaths)

    images, masks = list(), list()
    size = 256

    transformation = A.Compose([
        A.Resize(width=size, height=size)
    ])

    for image_filepath, mask_filepath in filepaths.items():


        image = cv2.cvtColor(cv2.imread(image_filepath), cv2.COLOR_BGR2RGB)
        mask = encode_mask(cv2.imread(mask_filepath, 0)).astype(np.float32)
        
        transformed = transformation(image=image, mask=mask)

        transformed_image = (transformed['image'] / 255.0).astype(np.float32)
        transformed_mask = np.expand_dims(transformed['mask'], axis=-1)

        images.append(transformed_image)
        masks.append(transformed_mask)

    return np.array(images), np.array(masks)





if __name__ == '__main__':

    train_image_path = os.path.join(os.getcwd(), "DATABASE/USGimage/train")
    train_mask_path = os.path.join(os.getcwd(), "DATABASE/USGimages_MASKS/train")

    get_data(train_image_path, train_mask_path)
