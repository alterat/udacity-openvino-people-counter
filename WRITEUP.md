# People Counter

**Udacity Nanodegree - Project Write-up**

Alberto Torin

--- 

## Model Conversion

The first step in the construction of the project is selecting an appropriate model for detection. 

The chosen model is [SSD Mobilenet trained on the COCO dataset](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz). 

After saving the tar file on the Desktop, the conversion to an OpenVINO Intermediate Representation (IR) has been achieved with the following command:

```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --reverse_input_channels --saved_model_dir ~/Desktop/ssd_inception_v2_coco_2018_03_29/saved_model/ --tensorflow_object_detection_api_pipeline_config ~/Desktop/ssd_inception_v2_coco_2018_03_29/pipeline.config 
```

Note the `--transformations_config` parameter that replaces the deprecated `--tensorflow_use_custom_operations_config` in the latest OpenVINO versions.


## Explaining Custom Layers

In some occasions, specific layers of a model might not be supported by OpenVINO. These layers are called _custom layers_. 

Handling custom layers in OpenVINO first of all requires to identify such layers. This can be done either empirically, while trying to convert a model into an IR, or programmatically, by looking at the layers supported by a specific device. 

Once identified, custom layers can be isolated from the optimised model and their computation can be offloaded to the CPU. 

This strategy allows to perform inference with an otherwise unusable model in OpenVINO, while at the same time trying to optimise at least some of its calculations. 

## Comparing Model Performance

Model performance has been compared for three different scenarios: The original Tensorflow model, run on a CPU, the optimised IR run on CPU and the same IR run on a Neural Compute Stick 2 (NCS2).

Tests were performed on a MacBook with 2.8 GHz Intel Core i7.

In order to compare the model before and after conversion to IR, I created a separate script to deal with the original Tensorflow model, while a function in the `main.py` file takes care of the IR models.

Test were performed on the following test image:

![Man standing on a rock]('./writeup-images/test-image.jpg')

The difference between model accuracy pre- and post-conversion was negligible, with the following results (the first number represents the accuracy, while the other 4 are the bounding box. Notice how x and y coordinates are reversed in the original TF model.)

Tensorflow:

```
[0.9925442  0.52989864 0.39880767 0.770481 0.5286476 ]
```

OpenVINO IR on CPU:

```
[0.9893007  0.39436024  0.5280018 0.52694 0.7676021 ]
```

OpenVINO IR on NCS2:

```
[0.98828125 0.3942871 0.5283203 0.5263672 0.7685547 ]
```


The size of the model pre- and post-conversion was 210 Mb and 34 Mb respectively, considering the folder with all the files. 

The inference times of the model pre- and post-conversion is reported in the following graph.

![Inference times for SSD Mobilenet v2]('./writup-images/times.png')

A single inference with the original TF model requires 159 ms, while only 34 ms are required for the IR on a CPU. Surprisingly, inference on a NCS2 took twice as long (68 ms), but this is in line with [the results reported by other people](https://medium.com/@aallan/benchmarking-edge-computing-ce3f13942245). 


## Assess Model Use Cases

Some of the potential use cases of the people counter app are monitoring corridors or spaces with low traffic. The app might be able to identify unneeded loitering and raise alarm.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. 

An incorrect lighting of the scene will result in missing detections and lower model performance.
