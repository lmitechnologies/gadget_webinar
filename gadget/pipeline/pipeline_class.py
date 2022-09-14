"""
Description:
Nordson pipeline class test.

Requirements:
this pipeline class, needs to have the following methods:
    load
    clean_up
    warm_up
    predict
"""

import os
import time
import shutil
import collections
import numpy as np
import cv2
import logging
import traceback

#own modules
from yolov5 import YoLov5TRT



class ModelPipeline:

    logger = logging.getLogger()

    def __init__(self, **kwargs):
        """
        init the pipeline with kwargs
        """
        
        def convert_key_to_int(dt):
            """
            convert the class map <id, class name> to integer class id
            """
            new_dt = {int(k):dt[k] for k in dt}
            return new_dt

        self.yolov5_configs = {}
        self.yolov5_configs['engine_path'] = os.path.realpath(os.path.expandvars(kwargs.get('yolov5_engine_file',"")))
        self.yolov5_configs['plugin_path'] = os.path.realpath(os.path.expandvars(kwargs.get('yolov5_plugin_file',"")))
        self.yolov5_configs['class_map'] = convert_key_to_int(kwargs.get('yolov5_class_map', {}))
        self.yolov5_configs['iou_thres'] = kwargs.get('yolov5_iou_threshold', 0.4)
        self.yolov5_configs['vertical_padding'] = kwargs.get('yolov5_vertical_padding', 20)
        self.yolov5_configs['max_output_bbox_cnt'] = kwargs.get('yolov5_max_output_bbox_cnt',1000)
        self.yolov5_configs['conf_thres'] = {
            'scuff':kwargs.get('confidence_scuff',0.5),
            'white':kwargs.get('confidence_white',0.5),
            'peeling':kwargs.get('confidence_peeling',0.5),
            'catheter':kwargs.get('confidence_catheter',0.8),
        }

        #map model name -> model instance
        self.models = collections.OrderedDict()
        
        
    def load(self):
        """
        create model instances with engine files
        if loading files fail, then don't create model instances
        """
        try:
            if len(self.yolov5_configs['class_map'])==0:
                raise Exception('model class maps are not defined')
            self.models['yolov5'] = YoLov5TRT(
                self.yolov5_configs['engine_path'], self.yolov5_configs['plugin_path'], self.yolov5_configs['class_map'], {},
                self.yolov5_configs['conf_thres'], self.yolov5_configs['iou_thres'], self.yolov5_configs['max_output_bbox_cnt']
                )
            self.logger.info('models are loaded')
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error('models are failed to load')
            self.models = None
        

    def clean_up(self):
        """
        clean up the pipeline in REVERSED order, i.e., the last models get destoried first
        """
        L = list(reversed(self.models.keys())) if self.models else []
        self.logger.info('cleanning up pipeline...')
        for model_name in L:
            if self.models[model_name]:
                self.models[model_name].destroy()
            del self.models[model_name]
            self.logger.debug(f'{model_name} is cleaned up')

        #del the pipeline
        del self.models
        self.models = None
        self.logger.info('pipeline is cleaned up')



    def warm_up(self):
        """
        warm up all the models in the pipeline
        """
        if not self.models:
            return
        for model_name in self.models:
            dummy_inputs = list(self.models[model_name].get_raw_image_zeros())
            self.logger.info(f'warming up {model_name} on dummy inputs with the size of {dummy_inputs[0].shape}')
            self.models[model_name].infer(dummy_inputs)


    def preprocess(self, input_images:list):
        """_summary_

        Args:
            input_images (list): _description_
        """
        MODEL_W = 512
        MODEL_H = 512
        
        orig_image = input_images[0]
        h, w = orig_image.shape[:2]
        
        #rotate image
        t1 = time.time()
        orig_image = np.rot90(orig_image)
        self.logger.debug(f'[{self.__class__.__name__}] rotate image from the size of {(h,w)} to {orig_image.shape}')
        self.logger.debug(f'rotate time: {time.time()-t1}')
        
        # resize image 
        t1 = time.time()
        image_raw = cv2.resize(orig_image.copy(), (MODEL_W, MODEL_H))
        self.logger.debug(f'resize time: {time.time()-t1}')
        return [image_raw]



    def predict(self, input_image: str, configs: dict, results_storage_path: str) -> dict:
        #load image from file
        start_time = time.time()
        errors = []
        try:
            #input_images = [np.array(Image.open(input_image))]
            input_images = [np.load(input_image, allow_pickle=True)] #in BGR format
        except Exception as e:
            input_images = []
            errors.append(e)
        image_load_time = time.time() - start_time

        if not self.models:
            errors.append('failed to load pipeline model(s)')

        conf_thres = {
            'scuff':configs.get('confidence_scuff', self.yolov5_configs['conf_thres']['scuff']),
            'white':configs.get('confidence_white', self.yolov5_configs['conf_thres']['white']),
            'peeling':configs.get('confidence_peeling', self.yolov5_configs['conf_thres']['peeling']),
            'catheter':configs.get('confidence_catheter', self.yolov5_configs['conf_thres']['catheter']),
        }
        padding = configs.get('veritical_padding', self.yolov5_configs['vertical_padding'])
        nms_iou_thres = configs.get('iou_threshold', self.yolov5_configs['iou_thres'])
        test_mode = configs.get('test_mode', False)
        
        is_run = 0
        preprocess_time = 0
        start_time = time.time()
        if len(input_images)>0 and self.models:
            t1 = time.time()
            processed_image = self.preprocess(input_images)
            preprocess_time += time.time() - t1
            is_run = 1
            
        # run inference 
        if is_run:
            annotated_images, obj_det_dict, proc_time_dict1, errors1 = self.models['yolov5'].infer(processed_image, conf_thres, nms_iou_thres, padding)
            errors += errors1
        pipeline_time = time.time() - start_time

        # log errors
        for e in errors:
            self.logger.error(f'[ERROR]: {e}')

        # gather time info
        inference_time, postprocess_time = 0, 0
        if is_run:
            preprocess_time += proc_time_dict1['pre']
            inference_time = proc_time_dict1['exec']
            postprocess_time = proc_time_dict1['post']
        
        # Save the edited image
        image_save_time = 0
        boxes,scores,classes = [],[],[]
        annotated_image_path = ''
        start_time = time.time()
        if is_run:
            annotated_image_path = os.path.join(results_storage_path, 'annotated_'+os.path.basename(input_image))
            img = annotated_images[0] #only one input image
            
            # gather results
            boxes = obj_det_dict[0]['boxes']
            scores = obj_det_dict[0]['scores']
            classes = obj_det_dict[0]['classes']
            
            if test_mode:
                cv2.imwrite(annotated_image_path.replace('.npy','.png'), img)
            else:
                np.save(annotated_image_path,img)
            
        image_save_time = time.time() - start_time
        total_time = image_load_time + pipeline_time + image_save_time
        
        result_dict = {
                'file_path': input_image,
                'automation_keys': [],
                'factory_keys': [],
                'errors': errors,
            }

        if is_run and not errors:
            # decision message for automation class
            classes_list = classes
            set_classes = set(classes_list)
            if 'catheter' not in set_classes:
                decision = 'empty' 
            elif len(set_classes)==1 and 'catheter' in set_classes:
                decision = 'good'
            else:
                # only return the defects
                decision = ','.join(cls for cls in classes_list if cls != 'catheter')
            
            result_dict['file_path'] = annotated_image_path
            result_dict['automation_keys'] = ['decision']
            result_dict['factory_keys'] = ['filename', 'catheter_height', 'obj_det_scores', 'obj_det_classes', 'total_proc_time']
            result_dict['decision'] = str(decision)
            result_dict['obj_det_boxes'] = str(boxes)
            result_dict['obj_det_scores'] = str(scores)
            result_dict['obj_det_classes'] = str(classes)
            result_dict['errors'] = errors
            
            self.logger.info(f'[FILE INFO]: input image: {os.path.basename(input_image)}')
            self.logger.info(f'[FILE INFO]: annotated output image: {os.path.basename(annotated_image_path)}')
            self.logger.info(f'[DECISION]: {decision}')
            self.logger.info(f'[TIME INFO]: image_load:{image_load_time:.4f}, preprocess:{preprocess_time:.4f}, inference:{inference_time:.4f}, ' +
                f'postprocess:{postprocess_time:.4f}, image_save:{image_save_time:.4f}, total:{total_time:.4f}\n')

        time_dict = {                
            'image_load_time': image_load_time,
            'preprocess_time': preprocess_time,
            'inference_time': inference_time,
            'postprocess_time': postprocess_time,
            'image_save_time': image_save_time,
            'total_proc_time': total_time
            }
        result_dict.update(time_dict)
        return result_dict



if __name__ == '__main__':
    import json
    def get_img_path_batches(batch_size, img_dir, fmt='png'):
        ret = []
        batch = []
        cnt_images = 0
        for root, dirs, files in os.walk(img_dir):
            for name in files:
                if name.find(f'.{fmt}')==-1:
                    continue
                if len(batch) == batch_size:
                    ret.append(batch)
                    batch = []
                batch.append(os.path.join(root, name))
                cnt_images += 1
        logger = logging.getLogger()
        logger.debug(f'loaded {cnt_images} files')
        if len(batch) > 0:
            ret.append(batch)
        return ret

    def get_classifier_img_path_batches(batch_size, img_dir):
        import json
        ret = []
        ret_label = []
        batch = []
        batch_label = []

        #load class map
        class_to_id = {}
        class_map_path = os.path.join(img_dir,'class_map.json')
        if os.path.isfile(class_map_path):
            class_to_id = json.load(open(class_map_path))
        else:
            raise Exception(f'cannot load the class_map.json in {img_dir}')

        # load images with class id
        cnt_images = 0
        for root, dirs, _files in os.walk(img_dir):
            for dir in dirs:
                label = class_to_id[dir]
                for file in os.listdir(os.path.join(root, dir)):
                    if len(batch) == batch_size:
                        ret.append(batch)
                        ret_label.append(batch_label)
                        batch = []
                        batch_label = []
                    batch.append(os.path.join(root, dir, file))
                    batch_label.append(label)
                    cnt_images += 1

        logger = logging.getLogger()
        logger.debug(f'loaded {cnt_images} images')
        if len(batch) > 0:
            ret.append(batch)
            ret_label.append(batch_label)
        return ret, ret_label

    def load_pipeline_def(filepath):
        with open(filepath) as f:
            dt_all = json.load(f)
            l = dt_all['configs_def']
            kwargs = {}
            for dt in l:
                kwargs[dt['name']] = dt['default_value']
        return kwargs
    

    BATCH_SIZE = 1
    os.environ['PIPELINE_SERVER_SETTINGS_MODELS_ROOT'] = './pipeline/models_x86'
    pipeline_def_file = './pipeline/pipeline_def.json'
    
    kwargs = load_pipeline_def(pipeline_def_file)
    pipeline = ModelPipeline(**kwargs)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.info('start loading the pipeline...')

    pipeline.load()
    pipeline.warm_up()

    #test on object detection training data 
    image_dir = './data/test_images'
    output_dir = './data/outputs'

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    image_path_batches = get_img_path_batches(BATCH_SIZE, image_dir, fmt='npy')
    for batch in image_path_batches:
        for image_path in batch:
            pipeline.predict(image_path, configs={'test_mode':True}, results_storage_path=output_dir)

    pipeline.clean_up()
