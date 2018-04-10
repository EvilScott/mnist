from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical

# retrieve and massage data
(pixel_train, label_train), (pixel_test, label_test) = mnist.load_data()
pixel_train = pixel_train.reshape(60000, 784)
pixel_test = pixel_test.reshape(10000, 784)
label_train = to_categorical(label_train, 10)
label_test = to_categorical(label_test, 10)

# define model
model = Sequential([
  Dense(256, activation='sigmoid', input_dim=784, use_bias=True),
  Dropout(0.2),
  Dense(32, activation='sigmoid'),
  Dropout(0.2),
  Dense(10, activation='softmax')
])
model.summary()

# compile model
model.compile(
  loss='categorical_crossentropy',
  optimizer='sgd',
  metrics=['accuracy']
)

# run model with training data
model.fit(
  pixel_train,
  label_train,
  epochs=20,
  batch_size=128
)

# evaluate model with test data
score = model.evaluate(
  pixel_test,
  label_test,
  batch_size=128
)
print('Test loss: %f' % score[0])
print('Test accuracy: %f' % score[1])
