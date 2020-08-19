import os
import cv2
import numpy as np
from tqdm import tqdm


class DataPreparation():
    IMG_SIZE = 400
    CANCER = "D://Projects//Yura//data//TEST//Cancer//"
    HEALTHY = "D://Projects//Yura//data//TEST//Healthy//"
    LABELS = {CANCER: 1, HEALTHY: 0}
    training_data = []
    
    cancercount = 0
    healthycount = 0
    
    
    def make_training_data(self):
        for label in self.LABELS:
            #print(label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img),np.eye(2)[self.LABELS[label]]])

                        if label == self.CANCER:
                            self.cancercount += 1
                        elif label == self.HEALTHY:
                            self.healthycount += 1
                    except Exception as e:
                        pass
                
                
        np.random.shuffle(self.training_data)
        np.save("D://Projects//Yura//test_data.npy", self.training_data)
        print("Cancer: ", self.cancercount)
        print("Healthy: ", self.healthycount)


print("DATA PREPROCESSED")