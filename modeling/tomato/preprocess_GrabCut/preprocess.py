class_list = ['Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

#preprocess
 
def preprocess_grabcut(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image , (224,224))
    mask = np.zeros(image.shape[:2], np.uint8) 
    backgroundModel = np.zeros((1, 65), np.float64) 
    foregroundModel = np.zeros((1, 65), np.float64)  
    rectangle = (20, 20, 200, 200) 
    cv2.grabCut(image, mask, rectangle,   
                backgroundModel, foregroundModel, 
                1, 
                cv2.GC_INIT_WITH_RECT) 
    mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8') 

    image = image * mask2[:, :, np.newaxis] 

    return image
