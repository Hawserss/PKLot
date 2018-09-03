import cv2

#Size of images
IMAGE_WIDTH = 70
IMAGE_HEIGHT = 70

'''
Image processing helper function
'''
def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


'''
Image loading and processing helper function
'''
def load_transform_img(img_path, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
