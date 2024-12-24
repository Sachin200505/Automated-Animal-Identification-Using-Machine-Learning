from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

train_dir = "train"
validation_dir = "validate"

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,        
    validation_split=0.2      
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Flatten(input_shape=(224, 224, 3)),  # Flatten input images
    Dense(128, activation='relu'),       # Hidden layer
    Dense(len(train_generator.class_indices), activation='softmax')  # Output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

model.save("animal_classifier_model.h5")
