import cv2

def preprocessing(image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    # image = np.copy(input_image)
    try:
        image = cv2.resize(image, (width, height), cv2.INTER_CUBIC)
    except TypeError:
        return None
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image
