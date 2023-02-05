import cv2 as cv
import numpy as np

from process import NormalizeImage, TopDownEvalAffine, HRNetPostProcess


class PointProcess():
    def __init__(self, 
                 trainsize=[192,256],
                 mean=[0.485,0.456,0.406], 
                 std=[0.229,0.224,0.225]
                 ):
        self.normalizeImage = NormalizeImage(mean = mean,std =std)
        self.topDownEvalAffine =TopDownEvalAffine(trainsize)

    def get_person_from_rect(self, image, results):
        # crop the person result from image
        valid_rects = results['bboxes']
        print(valid_rects.shape)
        rect_images = []
        new_rects = []
        org_rects = []
        for rect in valid_rects:
            rect_image, new_rect, org_rect = self.expand_crop(image, rect)
            if rect_image is None or rect_image.size == 0:
                continue
            rect_images.append(rect_image)
            new_rects.append(new_rect)
            org_rects.append(org_rect)
        return rect_images, new_rects, org_rects
    
    def expand_crop(self, images, rect, expand_ratio=0.3):
        imgh, imgw, c = images.shape
        xmin, ymin, xmax, ymax = [int(x) for x in rect.tolist()]
        
        org_rect = [xmin, ymin, xmax, ymax]
        h_half = (ymax - ymin) * (1 + expand_ratio) / 2.
        w_half = (xmax - xmin) * (1 + expand_ratio) / 2.
        if h_half > w_half * 4 / 3:
            w_half = h_half * 0.75
        center = [(ymin + ymax) / 2., (xmin + xmax) / 2.]
        ymin = max(0, int(center[0] - h_half))
        ymax = min(imgh - 1, int(center[0] + h_half))
        xmin = max(0, int(center[1] - w_half))
        xmax = min(imgw - 1, int(center[1] + w_half))
        return images[ymin:ymax, xmin:xmax, :], [xmin, ymin, xmax, ymax], org_rect


    def preprocess(self, images):
        input_im_lst = []
        input_im_info_lst = []
        for im in images:
            im, im_info = self.processim(im)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
        inputs = self.create_inputs(input_im_lst, input_im_info_lst)
        return inputs


    def create_inputs(self, imgs, im_info):
        """generate input for different model type
        Args:
            imgs (list(numpy)): list of image (np.ndarray)
            im_info (list(dict)): list of image info
        Returns:
            inputs (dict): input of model
        """
        inputs = {}
        inputs['image'] = np.stack(imgs, axis=0).astype('float32')
        im_shape = []
        for e in im_info:
            im_shape.append(np.array((e['im_shape'])).astype('float32'))
        inputs['im_shape'] = np.stack(im_shape, axis=0)
        return inputs
    
    def processim(self, im):
            # process image by preprocess_ops
        im_info = {
            'scale_factor': np.array(
                [1., 1.], dtype=np.float32),
            'im_shape': None,
        }
        im_info['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
        im_info['scale_factor'] = np.array([1., 1.], dtype=np.float32)
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im,im_info = self.topDownEvalAffine(im,im_info)
        im,im_info = self.normalizeImage(im,im_info)
        # im = im.transpose((2, 0, 1)).copy()
        return im, im_info
    def postprocess(self, inputs, result):
        np_heatmap = result
        results = {}
        imshape = inputs['im_shape'][:, ::-1]
        print(imshape)
        center = np.round(imshape / 2.)
        scale = imshape 
        keypoint_postprocess = HRNetPostProcess(use_dark=False)
        kpts, scores = keypoint_postprocess(np_heatmap, center, scale)
        results['keypoint'] = kpts
        results['score'] = scores
        return results