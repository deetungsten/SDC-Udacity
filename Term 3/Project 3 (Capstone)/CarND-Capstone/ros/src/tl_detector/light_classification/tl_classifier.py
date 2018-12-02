from styx_msgs.msg import TrafficLight
from darkflow.net.build import TFNet
import cv2
import os
class TLClassifier(object):
    def __init__(self):
        
        #
        pass

    def load(self):
        options = {"model": "cfg/yolov2-voc-3c.cfg", "load": "bin/yolov2-voc.weights", "threshold": 0.1, "gpu": 0.5, "load":-1}
        
        tfnet = TFNet(options)
        #tfnet.load_from_ckpt()
        return tfnet

    def get_classification(self, image, tfnet):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        #cwd = os.getcwd()
        #print(cwd)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        

        #imgcv = cv2.imread("./sample_img/sample_dog.jpg")
        predictions = tfnet.return_predict(image)
        state = TrafficLight.UNKNOWN
        state_green = 0
        state_red = 0
        state_yellow = 0
        for result in predictions:
            if result['label'] == 'Green':
                state_green += 1
            if result['label'] == 'Red':
                state_red += 1
            if result['label'] == 'Yellow':
                state_yellow += 1
        
        if predictions != []:
            array_light = [state_green, state_red, state_yellow]
            max_idx = array_light.index(max(array_light))

            if max_idx == 0:
                state = TrafficLight.GREEN
                print('Green')
            if max_idx == 1:
                state = TrafficLight.RED
                print('Red')
            if max_idx == 2:
                state = TrafficLight.YELLOW
                print('Yellow')
        
        return state
