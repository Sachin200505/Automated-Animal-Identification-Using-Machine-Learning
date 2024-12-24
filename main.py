import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

model = load_model("animal_classifier_model.h5")
labels = ["butterfly", "cat", "chicken", "cow", "dog","elephant","horse","sheep","spider","squirrel"]  # Replace with your class labels

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (224, 224)) 
    input_frame = np.expand_dims(resized_frame, axis=0)
    input_frame = preprocess_input(input_frame)

    predictions = model.predict(input_frame)
    predicted_class = labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    text = f"{predicted_class} ({confidence * 100:.2f}%)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Animal Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


labels = list(train_generator.class_indices.keys())

