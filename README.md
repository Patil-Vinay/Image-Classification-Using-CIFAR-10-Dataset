# Image-Classification-Using-CIFAR-10-Dataset

A CNN in TensorFlow/Keras that classifies CIFAR-10 images. I trained the same model twice, once without
and once with data augmentation, to see how augmentation affects overfitting.

CIFAR-10 has 60000 color images of size 32x32 in 10 classes (airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck), 6000 per class. Keras splits it into 50000 training and 10000 test images.

## Files

- `cifar_10_prediction_without_data_augmentation.ipynb` - trains the model on the plain training set
- `cifar_10_prediction_with_data_augmentation.ipynb` - trains the same model with augmentation and
  predicts the class of one test image
- `predict.py` - loads a saved `.h5` model and predicts the class of any image file
- `Result without using Data Augmentation.png`, `Result using Data Augmentation.png` - accuracy curves

The notebooks were run on Google Colab with a GPU (TensorFlow 2.8.2).

## Model

- Pixel values are scaled to 0-1
- 3 blocks of two `Conv2D` layers (32, 64, 128 filters, 3x3, relu) with `BatchNormalization`,
  each block ending in 2x2 `MaxPooling2D`
- Then `Flatten`, `Dropout(0.2)`, `Dense(1024, relu)`, `Dropout(0.2)` and `Dense(10, softmax)`
- Adam optimizer, sparse categorical crossentropy, 50 epochs
- About 2.4M parameters

The augmented version uses `ImageDataGenerator` with random width/height shifts of 10% and
horizontal flips, batch size 32. The 10000 test images are used as the validation set in both notebooks.

## Results

Without data augmentation:

![without augmentation](Result%20without%20using%20Data%20Augmentation.png)

With data augmentation:

![with augmentation](Result%20using%20Data%20Augmentation.png)

Last epoch of the runs saved in the notebooks:

| | train accuracy | validation accuracy |
|---|---|---|
| without augmentation | 0.9869 | 0.8421 |
| with augmentation | 0.9371 | 0.8858 |

Without augmentation the training accuracy goes close to 99% while validation accuracy stays around 84%,
so the model is overfitting. With augmentation the gap between the two curves is much smaller and the
validation accuracy is higher. The numbers will change a little on every run.

## Running

Open either notebook in Colab (or Jupyter) and run all cells. The dataset is downloaded by
`tf.keras.datasets.cifar10`. At the end the model is saved as `model_without_data_augmentation.h5` or
`model_with_data_augmentation.h5`. The last cell uses `google.colab.files` to download it, skip that
cell when running locally.

The trained models are not kept in the repo because of their size, run a notebook to generate one.

To predict an image with a saved model:

```
pip install -r requirements.txt
python predict.py my_image.jpg --model model_with_data_augmentation.h5
```

It resizes the image to 32x32 and prints the top 3 classes with their probabilities.
