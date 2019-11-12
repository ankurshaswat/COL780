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
