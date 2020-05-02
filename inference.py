#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.ie = None
        self.input_blob = None
        self.network = None
        self.exec_network = None
        self.request_handle = None
        self.request_status = None

    def load_model(self, model, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device.
        Synchronous requests made within.
        ''' 
        
        # Initialize the ie
        self.ie = IECore()

        ### Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Read the IR as a IENetwork
        self.network = self.ie.read_network(model=model_xml, weights=model_bin)

        ### TODO: Check for supported layers ###

        ### Add any necessary extensions ###
        if cpu_extension and "CPU" in device:
            self.ie.add_extension(cpu_extension, device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        ### Return the loaded inference ie ###
        ### Note: You may need to update the function parameters. ###
        # Load the IENetwork into the ie
        self.exec_network = self.ie.load_network(network=self.network, device_name=device, num_requests=0)

        return self.exec_network

    def get_input_shape(self):
        ### Return the shape of the input layer ###
        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))

        # Return the input shape (to determine preprocessing)
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, image):
        ### Start an asynchronous request ###
        ### Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        '''
        Perform async inference and return status handle
        '''
        self.request_handle = self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})
        return self.request_handle

    def wait(self):
        ### Wait for the request to be complete. ###
        ### Return any necessary information ###
        ### Note: You may need to update the function parameters. ###

        self.request_status = self.request_handle.wait()
        return self.request_status

    def get_output(self, req_id=0):
        ### Extract and return the output results
        ### Note: You may need to update the function parameters. ###

        return self.exec_network.requests[req_id].outputs[self.output_blob]
