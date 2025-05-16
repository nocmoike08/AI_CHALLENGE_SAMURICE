import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
model = load_model(r"C:\Users\PC\Documents\Zalo Received Files\project-6-at-2025-05-15-11-51-8c159b71\food_cnn_model.h5")
class_names = [
    "Ca kho", "Canh bau", "Canh bi do", "Canh cai", "Canh chua", "Com",
    "Dau hu xao ca chua", "Ga chien", "Rau muong xao", "Thit kho",
    "Thit kho trung", "Trung chien"
]
price_dict = {
    "Ca kho": 20000, "Canh bau": 15000, "Canh bi do": 15000, "Canh cai": 15000,
    "Canh chua": 15000, "Com": 5000, "Dau hu xao ca chua": 18000, "Ga chien": 25000,
    "Rau muong xao": 12000, "Thit kho": 25000, "Thit kho trung": 30000, "Trung chien": 10000
}

cap = cv2.VideoCapture(0)
total_price = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.resize(frame, (128, 128))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)
    class_id = np.argmax(preds)
    confidence = np.max(preds)

    label = class_names[class_id]
    price = price_dict[label]

    text = f"{label} - {price} VND ({confidence:.2f})"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Tong tien: {total_price:,} VND", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Canteen CNN Realtime", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('c') and confidence >= 0.55:
        total_price += price
        print(f"Da them {label}: {price} VND")
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cnn_model.summary()

