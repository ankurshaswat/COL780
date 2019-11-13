"""
Any extra functions required
"""

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision import transforms

NUM_CHANNELS = 5


def parse_args():
    """
    Parse Arguments
    """

    parser = argparse.ArgumentParser(
        description='Get all arguments for training.')

    parser.add_argument('-disable_cuda', action='store_false',
                        dest='cuda', default=True)
    parser.add_argument('--shuffle', action='store_false', default=True)

    parser.add_argument('--all_data_path', action='store',
                        default='../data/Generic')
    parser.add_argument('--load_model_path', action='store',
                        default='')
    parser.add_argument('--validation_split_size',
                        action='store', type=float, default=0.1)
    parser.add_argument('--batch_size', action='store', type=int, default=1024)
    parser.add_argument('--num_workers', action='store', type=int, default=8)

    parser.add_argument('--epoch', action='store', type=int, default=20)
    parser.add_argument('--lr', action='store', type=float, default=0.1)
    parser.add_argument('--momentum', action='store', type=float, default=0.9)
    parser.add_argument('--l2_regularization',
                        action='store', type=float, default=1e-5)

    args = parser.parse_args()

    return args


def draw_game(frame, size, text):
    """
    Draw game using game_object on frame.
    """

    size_y = size[0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left = (10, size_y-10)
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2

    cv2.putText(frame, str(text), bottom_left, font,
                font_scale, font_color, line_type)

    return frame


def imshow(img):
    """
    Show image after unnormalizing
    """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_binary(img_gray):
    """
    Get binary threshold filter of image.
    """
    thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)[1]
    thresh = thresh[:, :, np.newaxis]
    return thresh


def add_channels(img):
    """
    Add extra channels to loaded image.
    """
    opencv_img = np.array(img)

    img_gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)

    thresh_holded_img = get_binary(img_gray)

    img_gray = img_gray[:, :, np.newaxis]

    img_combined = np.concatenate(
        (opencv_img, img_gray, thresh_holded_img), axis=2)

    resized_img = cv2.resize(img_combined, (50, 50))
    return resized_img


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    Plot Confusion Matrix
    """
    if title is not None:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks and label them with the respective list entries
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes,
           yticklabels=classes, title=title, ylabel='True label', xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax


TRANSFORM = transforms.Compose([transforms.Lambda(add_channels), transforms.ToTensor(
), transforms.Normalize([0.5]*NUM_CHANNELS, [0.5]*NUM_CHANNELS)])
