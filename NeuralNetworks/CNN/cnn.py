# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Part 1 - Building the CNN

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))  

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))  
classifier.add(MaxPooling2D(pool_size=(2, 2)))


classifier.add(Conv2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=256, activation='relu'))  # Increased neurons
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)  

test_datagen = ImageDataGenerator(rescale=1./255)

# Training set
training_set = train_datagen.flow_from_directory(
    'D:/Cources/Machine_Learning_A-Z/NeuralNetworks/CNN/dataset/training_set',
    target_size=(256, 256),  
    batch_size=32,
    class_mode='binary',
    subset='training')  

validation_set = train_datagen.flow_from_directory(
    'D:/Cources/Machine_Learning_A-Z/NeuralNetworks/CNN/dataset/training_set',
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    subset='validation')  

test_set = test_datagen.flow_from_directory(
    'D:/Cources/Machine_Learning_A-Z/NeuralNetworks/CNN/dataset/test_set',
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary')


from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=3)

# Fit the model
history = classifier.fit(
    training_set,
    steps_per_epoch=len(training_set),  
    epochs=25,
    validation_data=validation_set,
    validation_steps=len(validation_set),
    callbacks=[early_stop])