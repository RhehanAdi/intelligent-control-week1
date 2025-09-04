import cv2
import numpy as np

# Buka kamera
cap = cv2.VideoCapture(0)

# Definisi rentang warna dalam HSV
color_ranges = {
    "Merah": [
        (np.array([0, 150, 100]), np.array([10, 255, 255])),
        (np.array([170, 150, 100]), np.array([180, 255, 255]))
    ],
    "Hijau": [
        (np.array([36, 100, 100]), np.array([86, 255, 255]))
    ],
    "Biru": [
        (np.array([94, 150, 80]), np.array([126, 255, 255]))
    ],
    "Kuning": [
        (np.array([15, 150, 150]), np.array([35, 255, 255]))
    ],
    "Hitam": [
        # 🔹 persempit range hitam → hanya nilai V sangat rendah
        (np.array([0, 0, 0]), np.array([180, 60, 60]))
    ]
}

# Warna kotak (BGR)
box_colors = {
    "Merah": (0, 0, 255),
    "Hijau": (0, 255, 0),
    "Biru": (255, 0, 0),
    "Kuning": (0, 255, 255),
    "Hitam": (80, 80, 80)   # abu-abu biar terlihat jelas
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, ranges in color_ranges.items():
        mask = None
        for (lower, upper) in ranges:
            temp_mask = cv2.inRange(hsv, lower, upper)
            mask = temp_mask if mask is None else mask + temp_mask

        # 🔹 Kurangi noise dengan operasi morfologi
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=3)

        # Cari kontur
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 🔹 khusus hitam, area minimum diperbesar
            min_area = 2000 if color_name == "Hitam" else 800

            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_colors[color_name], 2)
                cv2.putText(frame, color_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_colors[color_name], 2)

    cv2.imshow("Deteksi Warna", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
