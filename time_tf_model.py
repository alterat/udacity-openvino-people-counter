import tensorflow as tf 
import os
import time
from PIL import Image
import numpy as np

model_path = "./model/ssd_mobilenet_v2_coco_2018_03_29/model.ckpt"

image = Image.open('./writeup-images/test-image.jpg')

image_np_expanded = np.expand_dims(image, axis=0)
detection_graph = tf.Graph()
with tf.compat.v1.Session(graph=detection_graph) as sess:
    # Load the graph with the trained states
    loader = tf.compat.v1.train.import_meta_graph(model_path+'.meta')
    loader.restore(sess, model_path)

    # Get the tensors by their variable name
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    
    start = time.time()
    for ind in range(100):
        if (ind%10==0):
            print(ind)
        # Make predictions
        _boxes, _scores = sess.run([boxes, scores], feed_dict={image_tensor: image_np_expanded}) 
    end = time.time()

elapsed = end-start

print('=== Output of the inference')
print(_boxes)
print(_scores)

print('=== Average time for single inference')
print(f'    {elapsed/100 * 1000} ms')