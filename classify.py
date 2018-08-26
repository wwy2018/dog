import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.applications.resnet50 import decode_predictions
from keras import backend as K

def train(img_path):
  K.clear_session()
  my_model = ResNet50(weights='./resnet50_weights_tf_dim_ordering_tf_kernels.h5')
  image_size = 224
  img=load_img(img_path,target_size=(image_size,image_size))
  img_array=np.array([img_to_array(img)])
  inpu=preprocess_input(img_array)
  preds=my_model.predict(inpu)
  deco=decode_predictions(preds)
  return deco
# avoid the bug
# train('./d1.jpg')
