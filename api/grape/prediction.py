import pandas as pd 
import pickle 
import sklearn
import re
import tensorflow.keras
import numpy as np
import cv2  
from tensorflow.keras import backend as K 
IMG_SIZE = 256 
from tensorflow.keras.models import model_from_json
 
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
print(loaded_model)
# load weights into new model
loaded_model.load_weights('model.h5')


class_list = ['Esca_(Black_Measles)',
                'Leaf_blight',
                'black rot',
                'healthy']


def preprocess_grabcut(image): 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image , (IMG_SIZE,IMG_SIZE))
    
    # create a simple mask image similar 
    # to the loaded image, with the  
    # shape and return type 
    mask = np.zeros(image.shape[:2], np.uint8) 

    # specify the background and foreground model 
    # using numpy the array is constructed of 1 row 
    # and 65 columns, and all array elements are 0 
    # Data type for the array is np.float64 (default) 
    backgroundModel = np.zeros((1, 65), np.float64) 
    foregroundModel = np.zeros((1, 65), np.float64) 

    # define the Region of Interest (ROI) as the coordinates of the rectangle 
    # where the values are entered as (startingPoint_x, startingPoint_y, width, height) 
    # these coordinates are according to the input image it may vary for different images 
    rectangle = (20, 20, 200, 200) 

    # apply the grabcut algorithm with appropriate  values as parameters, number of iterations = 3  
    # cv2.GC_INIT_WITH_RECT is used because of the rectangle mode is used  
    cv2.grabCut(image, mask, rectangle,backgroundModel, foregroundModel, 1,cv2.GC_INIT_WITH_RECT) 

    # In the new mask image, pixels will be marked with four flags four flags denote the background /
    # foreground mask is changed, all the 0 and 2 pixels are converted to the background 
    # mask is changed, all the 1 and 3 pixels are now the part of the foreground 
    # the return type is also mentioned,this gives us the final mask 
    mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8') 
    # The final mask is multiplied with the input image to give the segmented image. 
    image = image * mask2[:, :, np.newaxis] 
    image = image/255 
    # return image """
    return image
    
def pred(original_image):

  image = preprocess_grabcut(original_image)
  tab = []
  tab.append(image)
  tab = np.array(tab)
  h = np.expand_dims(tab, axis=2)
  h=loaded_model.predict(tab)
  predicted_class = np.argmax(h[0])
  confidence = np.max(h[0])
  result = class_list[predicted_class]+' with conf '+str(confidence)
  return result 



  