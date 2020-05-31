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

def draw_box(image, person, thres):
    '''
    Draw boxes around a person in the image.

    '''
    H,W,_ = image.shape

    _,_ , score, x1, y1, x2, y2 = person

    x1 = int(W * x1)
    x2 = int(W * x2)
    y1 = int(H * y1)
    y2 = int(H * y2)

    if score>=thres:
        # green
        colour = (0,255,0)
    # else:
    #     #red
    #     colour = (0,0,255)

        image = cv2.rectangle(image, (x1,y1), (x2,y2), colour, 2)

    return image