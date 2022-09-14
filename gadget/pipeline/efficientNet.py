from logging import warning
import pycuda.autoinit
import pycuda.driver as cuda
import cv2
import numpy as np
import time

from base_trt_model import TRT_Model

class EfficientNetTRT(TRT_Model):
    """
    description: A EfficientNet class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path, class_map):
        super().__init__(engine_file_path)
        self.class_map = class_map

    def _softmax(self, input, axis):
        #inference functions
        exps = np.exp(input)
        sums = np.expand_dims(np.sum(exps, axis=axis),-1)
        return exps/(sums+1e-16)

    def _reshape_outputs(self, outputs):
        outputs_new = np.reshape(outputs, [self.batch_max_size, -1])
        return outputs_new
        
    def infer(self, images_raw:list):
        start = time.time()
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        errors = []

        # Do image preprocess
        batch_input_images = np.empty(shape=[self.batch_max_size, 3, self.input_h, self.input_w])
        resized_raw_images = []
        batch_size = len(images_raw)
        for i, image in enumerate(images_raw):
            input_image, resized_raw_image, errs = self.preprocess_image(image)
            errors += errs
            resized_raw_images.append(resized_raw_image)
            np.copyto(batch_input_images[i], input_image)
        batch_input_images = np.ascontiguousarray(batch_input_images)

        # Copy input image to host buffer
        np.copyto(self.host_inputs[0], batch_input_images.ravel())
        preproc_time = time.time() - start
        
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        # Run inference.
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        exec_time = time.time() - start

        start = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        
        # Do postprocess
        reshaped_outputs = self._reshape_outputs(self.host_outputs[0])
        probs = self._softmax(reshaped_outputs, axis=1)
        if np.any(np.isnan(probs)):
            warning('classifier found NaN, replace to zero!')
            probs = np.nan_to_num(probs)
            print('classifier results: ', probs)
        
        pred_id = np.argmax(probs, axis=1)
        pred_class = [self.class_map[pid] for pid in pred_id]
        probs = probs.tolist()
        pred_id = pred_id.tolist()
        results = {'probs': probs, 'pred_id':pred_id, 'pred_class':pred_class}
        postproc_time = time.time() - start
        return resized_raw_images, results, {'pre':preproc_time, 'exec':exec_time, 'post':postproc_time}, errors

