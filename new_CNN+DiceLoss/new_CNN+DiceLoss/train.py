import tensorflow as tf
from dataset import get_data
from model import get_model
import os
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models as sm
from folder import KFolder

def train_model():

    train_image_directory = os.path.join(os.getcwd(), "DATABASE/USGimage/train")
    train_mask_directory = os.path.join(os.getcwd(), "DATABASE/USGimages_MASKS/train")

    val_image_directory = os.path.join(os.getcwd(), "DATABASE/USGimage/valid")
    val_mask_directory = os.path.join(os.getcwd(),"DATABASE/USGimages_MASKS/valid")

    test_image_directory = os.path.join(os.getcwd(), "DATABASE/USGimage/test")
    test_mask_directory = os.path.join(os.getcwd(), "DATABASE/USGimages_MASKS/test")


    X_train, y_train = get_data(image_path=train_image_directory, mask_path=train_mask_directory)
    X_valid, y_valid = get_data(image_path=val_image_directory, mask_path=val_mask_directory)
    X_test, y_test = get_data(image_path=test_image_directory, mask_path=test_mask_directory)

    X,y = np.concatenate([X_train, X_valid], axis=0), np.concatenate([y_train, y_valid], axis=0)

    del X_train, y_train, X_valid, y_valid
    
    img_size = (256,256)
    num_classes = 1
    batch_size = 16
    epochs = 100
    optimizer = tf.keras.optimizers.Adam(0.0003)
    callbacks = [tf.keras.callbacks.ModelCheckpoint("./best_model.h5", save_best_only=True, verbose=1)]
    metrics = [tf.keras.metrics.BinaryAccuracy(), sm.metrics.FScore()]
    
    kfolder = KFolder(k=5, x=X, y=y)

    for i, fold in enumerate(kfolder.folds):

        callbacks = [tf.keras.callbacks.ModelCheckpoint(f"./best_model_fold_{i}.h5", save_best_only=True, verbose=1)]
        model = get_model(img_size=img_size, num_classes=num_classes)
        model.compile(loss=sm.losses.bce_dice_loss, optimizer=optimizer,metrics=metrics)

        X_train, y_train, X_valid, y_valid = fold

        model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=1)

        del model
        best_model = tf.keras.models.load_model(f"./best_model_fold_{i}.h5", custom_objects={"binary_crossentropy_plus_dice_loss":sm.losses.bce_dice_loss, "f1-score":sm.metrics.FScore()})

        evaluation = best_model.evaluate(X_test, y_test)

        with open(f"./results_fold_{i}.txt",'w') as f:
            f.write(f"test loss: {evaluation[0]}\n")
            f.write(f"accuracy: {evaluation[1]}\n")
            f.write(f"dice-score: {evaluation[2]}\n")
        print(100*'--')


if __name__=='__main__':
    train_model()
