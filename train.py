import keras.backend as K
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import get_dataset_1
import get_dataset_2

# 加载训练数据和测试数据
(train_image,train_label) = get_dataset_1.load_data('data/mfcc_image_tr/')
(test_image,test_label) = get_dataset_2.load_data('data/mfcc_image_ts/')


input_shape = (250, 250, 4)
K.clear_session()

# Create a new model
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

model.compile(optimizer='adam',       # 为训练选择优化器和损失函数
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_image, train_label, epochs=5)  # 训练

model.evaluate(test_image, test_label)  # 测试

# 将模型保存为 HDF5 文件
model.save('my_model.h5')

print("模型已保存")
