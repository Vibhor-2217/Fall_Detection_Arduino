# 🤖 Fall Detection using YOLO & Arduino ESP32

A real-time fall detection system using a YOLO pretrained model and OpenCV, integrated with an Arduino ESP32 device. This project identifies human falls using computer vision techniques and sends alerts via WebSocket.

---

## 📌 Features
- 🚶‍♂️ Detects human falls from video feed
- 🧠 Utilizes YOLOv5 for object detection
- 📷 Real-time video processing using OpenCV
- 🌐 Communicates with ESP32 over WebSocket
- 💻 Python backend + Arduino frontend

---

## 📁 Dataset
The model is trained using the **Fall Detection Dataset** from Kaggle:  
🔗 [Fall Detection Dataset](https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset)

---

## ⚙️ Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Fall_Detection_Arduino.git
cd Fall_Detection_Arduino
```

### Step 2: Update `dataset.yaml`
Update the `path:` variable to the correct path where your dataset or pretrained YOLO weights are stored.

### Step 3: Configure WebSocket in `fall.py`
Inside `fall.py`, change the WebSocket `host` and `port` if needed (default is usually fine for local testing).

### Step 4: Setup Arduino ESP32 on Wokwi
Go to this project: [Wokwi ESP32 Project](https://wokwi.com/projects/421389136161014785)

Update the following lines:
```cpp
#define WIFI_NAME "YourWiFiName"
#define WIFI_PASSWORD "YourWiFiPassword"
```
Ensure the WebSocket URL matches the one from `fall.py`.

---

## 🚀 Running the Project

1. Run the Python server first:
   ```bash
   python fall.py
   ```

2. Then compile and run the ESP32 Arduino sketch from the Wokwi project.

> ⚠️ Ensure the WebSocket server is active before connecting from the ESP32.

---

## 📊 Tech Stack
- **Language**: Python, C++ (Arduino)
- **Libraries**: YOLOv5, OpenCV, WebSocket, Torch
- **Hardware**: Arduino ESP32 (simulated via Wokwi)

---

## 📸 Demo (Optional)
*Insert GIF or screenshot of fall detection in action*  
```markdown
![Demo](demo.gif)
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author & Contact
Developed by [Your Name]  
📧 Email: your@email.com  
🌐 Portfolio / GitHub / LinkedIn

---

_If you found this helpful, don't forget to ⭐ the repository!_
