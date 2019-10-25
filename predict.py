import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg


model = tf.keras.models.load_model('my_model.h5')

img = mpimg.imread('data/mfcc_image_ts/6/6_yicong_24.png')
img2 = np.zeros((1,img.shape[0],img.shape[1],img.shape[2]))
img2[0, :] = img

# img加载进来是(250, 250, 4)
# 这里对img进行预测的话会报错
# ValueError: Error when checking input:expected conv2d_1_input to have 4 dimensions, but got array with shape (X, X, X)
# 原因是方法predict_classes()要求输入的第一个dimension是bachsize，所以需要将数据reshape为(1，X, X, X)。
# 四个参数分别对应图片的个数，图片的通道数，图片的长与宽。具体的参加keras文档。

pre = model.predict_classes(img2)
result = pre[0]
print(result)
