
        
import openvino.runtime as ov


class KeyPointDetector():
    def __init__(self, 
                 model_dir,
                 device='CPU',
                 batch_size=1,
                 output_dir='output',
                 threshold=0.5
                 ):
        self.model_dir = model_dir
        self.device = device
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.threshold = threshold
        self.predictor = self.load_predictor(model_dir=self.model_dir, device=self.device) 
        
    def load_predictor(self, model_dir, device ):
        ie_core = ov.Core()
        model = ie_core.read_model(model=model_dir)
        compiled_model = ie_core.compile_model(model=model, device_name=device)
        infer_request = compiled_model.create_infer_request()
        return infer_request
    

    def predict_image(self, inputs):
        image = inputs['image']
        outputs = self.predict(image)
        return outputs
        
        
    def predict(self, image):
        self.predictor.infer(inputs={'image': image})
        outputs = self.predictor.get_output_tensor(0).data
        return outputs