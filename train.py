import keras.backend as K
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import numpy as np
import get_dataset_1
import get_dataset_2

# 加载训练数据和测试数据
(train_image, train_label) = get_dataset_1.load_data('data/mfcc_image_tr/')
(test_image, test_label) = get_dataset_2.load_data('data/mfcc_image_ts/')

# 打乱训练集顺序
np.random.seed(666)
np.random.shuffle(train_image)
np.random.seed(666)
np.random.shuffle(train_label)

input_shape = (250, 250, 4)
K.clear_session()

# 创建一个新模型
model = Sequential()

model.add(Conv2D(16, (7, 7), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.summary()

# 为训练选择优化器和损失函数
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练
model.fit(train_image, train_label, epochs=5)

# 测试
model.evaluate(test_image, test_label)

# 将模型保存为 HDF5 文件
model.save('my_model.h5')

print("模型已保存")
