
import cv2 as cv
import numpy as np

from process import Resize, NormalizeImage



class PicoDetProcess():
    def __init__(self, 
                 trainsize=[320,320],
                 mean=[0.485,0.456,0.406], 
                 std=[0.229,0.224,0.225],
                 score_threshold=0.4,
                 nms_threshold=0.5
                 ):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.resize =Resize(trainsize)
        self.normalizeImage = NormalizeImage(mean = mean,std =std)
        
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
            imgs (list(numpy)): list of images (np.ndarray)
            im_info (list(dict)): list of image info
        Returns:
            inputs (dict): input of model
        """
        inputs = {}
        im_shape = []
        scale_factor = []
        if len(imgs) == 1:
            inputs['image'] = np.array((imgs[0], )).astype('float32')
            inputs['im_shape'] = np.array(
                (im_info[0]['im_shape'], )).astype('float32')
            inputs['scale_factor'] = np.array(
                (im_info[0]['scale_factor'], )).astype('float32')
            return inputs

        for e in im_info:
            im_shape.append(np.array((e['im_shape'], )).astype('float32'))
            scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))

        inputs['im_shape'] = np.concatenate(im_shape, axis=0)
        inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

        imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
        max_shape_h = max([e[0] for e in imgs_shape])
        max_shape_w = max([e[1] for e in imgs_shape])
        padding_imgs = []
        for img in imgs:
            im_c, im_h, im_w = img.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape_h, max_shape_w), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = img
            padding_imgs.append(padding_im)
        inputs['image'] = np.stack(padding_imgs, axis=0)
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
        im,im_info = self.resize(im,im_info)
        im,im_info = self.normalizeImage(im,im_info)
        # im = im.transpose((2, 0, 1)).copy()
        return im, im_info
    
    
    def postprocess(self, inputs, scale_factor):
        bboxs = inputs['bboxes']
        scores = inputs['scores']
        bbox,score = self.nms(bboxs[0],scores[0][0])
        for box in bbox:
            box[0] = box[0] / scale_factor[1]
            box[1] = box[1] / scale_factor[0]
            box[2] = box[2] / scale_factor[1]
            box[3] = box[3] / scale_factor[0]
        outputs =  dict(bboxes=np.array(bbox), scores=np.array(score))
        return outputs
    
    def nms(self, bounding_boxes, confidence_score):
        '''
        :param bounding_boxes: 候选框列表，[左上角坐标, 右下角坐标], [min_x, min_y, max_x, max_y], 原点在图像左上角
        :param confidence_score: 候选框置信度
        :param threshold: IOU阈值
        :return: 抑制后的bbox和置信度
        '''
        picked = []
            
        for i in range(confidence_score.shape[-1]):
            if confidence_score[i] > self.score_threshold:
                    picked.append(i)
        bounding_boxes = bounding_boxes[picked,:]
        confidence_score = confidence_score[picked]
        # 如果没有bbox，则返回空列表
        if len(bounding_boxes) == 0:
            return [], []

        # bbox转为numpy格式方便计算
        boxes = np.array(bounding_boxes)

        # 分别取出bbox的坐标
        start_x = boxes[:, 0]
        start_y = boxes[:, 1]
        end_x = boxes[:, 2]
        end_y = boxes[:, 3]

        # 置信度转为numpy格式方便计算
        score = np.array(confidence_score)  # [0.9  0.75 0.8  0.85]

        # 筛选后的bbox和置信度
        picked_boxes = []
        picked_score = []

        # 计算每一个框的面积
        areas = (end_x - start_x + 1) * (end_y - start_y + 1)

        # 将score中的元素从小到大排列，提取其对应的index(索引)，然后输出到order
        order = np.argsort(score)   # [1 2 3 0]

        # Iterate bounding boxes
        while order.size > 0:

            # The index of largest confidence score
            # 取出最大置信度的索引
            index = order[-1]

            # Pick the bounding box with largest confidence score
            # 将最大置信度和最大置信度对应的框添加进筛选列表里
            picked_boxes.append(bounding_boxes[index])
            picked_score.append(confidence_score[index])

            # 求置信度最大的框与其他所有框相交的长宽，为下面计算相交面积做准备
            # 令左上角为原点，
            # 两个框的左上角坐标x取大值，右下角坐标x取小值，小值-大值+1==相交区域的长度
            # 两个框的左上角坐标y取大值，右下角坐标y取小值，小值-大值+1==相交区域的高度
            # 这里可以在草稿纸上画个图，清晰明了
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])

            # 计算相交面积，当两个框不相交时，w和h必有一个为0，面积也为0
            w = np.maximum(0.0, x2 - x1 + 1)
            h = np.maximum(0.0, y2 - y1 + 1)
            intersection = w * h

            # 计算IOU
            ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

            # 保留小于阈值的框的索引
            left = np.where(ratio < self.nms_threshold)
            # 根据该索引修正order中的索引（order里放的是按置信度从小到大排列的索引）
            order = order[left]

        return picked_boxes, picked_score