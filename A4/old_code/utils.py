"""
All functions combined
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

NUM_CHANNELS = 5


def draw_game(frame, size, text):
    """
    Draw game using game_object on frame.
    """
    # pos_float = game_obj['pos']
    # pos_int = ((int)(pos_float[0]), (int)(pos_float[1]))
    # frame = cv2.circle(frame, pos_int, 15, (0, 0, 255), -1)
    # frame = cv2.rectangle(frame, (15, 15), (640-15, 480-15), (0, 255, 0), 1)

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
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_binary(img_gray):
    thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)[1]
    thresh = thresh[:, :, np.newaxis]
    return thresh


def add_channels(img):
    openCVim = np.array(img)

    img_gray = cv2.cvtColor(openCVim, cv2.COLOR_BGR2GRAY)
#     print('img_gray {}'.format(img_gray.shape))
    thresh_holded_img = get_binary(img_gray)
#     print(thresh_holded_img)
    img_gray = img_gray[:, :, np.newaxis]
#     edges = cv2.Canny(openCVim,100,200)
#     edges = edges[:,:,np.newaxis]
#     print(edges.shape)
    img_combined = np.concatenate(
        (openCVim, img_gray, thresh_holded_img), axis=2)
#     print(img_combined.shape)

    resized_img = cv2.resize(img_combined, (50, 50))
    # resized_img = cv2.resize(openCVim, (50, 50))
#     print(openCVim.shape,img_gray.shape,thresh_holded_img.shape)
#     img_combined = np.concatenate((openCVim,img_gray), axis=2)
#     PILim = Image.fromarray(img_combined)
    return resized_img


TRANSFORM = transforms.Compose(
    [
        transforms.Lambda(add_channels),
        # transforms.Resize((50,50)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*NUM_CHANNELS, [0.5]*NUM_CHANNELS)])

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
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
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax