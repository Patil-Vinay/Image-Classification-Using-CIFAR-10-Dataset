import argparse

import numpy as np
import tensorflow as tf

LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']


def load_image(path):
    # same input as training: 32x32 RGB scaled to 0-1
    img = tf.keras.preprocessing.image.load_img(path, target_size=(32, 32))
    x = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return x.reshape(1, 32, 32, 3)


def main():
    parser = argparse.ArgumentParser(description='Predict the CIFAR-10 class of an image')
    parser.add_argument('image', help='path to the image file')
    parser.add_argument('--model', default='model_with_data_augmentation.h5',
                        help='.h5 model saved by one of the notebooks')
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    probs = model.predict(load_image(args.image))[0]

    # top 3 guesses
    for idx in np.argsort(probs)[::-1][:3]:
        print('{}: {:.2f}%'.format(LABELS[idx], probs[idx] * 100))


if __name__ == '__main__':
    main()
