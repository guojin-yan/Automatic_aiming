import imp
import cv2 as cv
import numpy as np

from person_detector import PicoDetDetector
from person_process import PicoDetProcess
from point_detector import KeyPointDetector
from point_process import PointProcess


class PersonAiming:
    def __init__(self, det_model, point_model, device):
        self.preson_detecter = PicoDetDetector(model_dir=det_model)
        self.point_decter