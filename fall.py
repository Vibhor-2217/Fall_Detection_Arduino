import cv2
import asyncio
import websockets
from ultralytics import YOLO
from collections import deque
from contextlib import suppress
import time

# Configuration
CONFIG = {
    "model_path": "yolov8n.pt",
    "camera_index": 0,
    "websocket": {
        "host": "127.0.0.1",
        "port": 9090,
        "client_timeout": 5  # seconds
    },
    "detection": {
        "confidence_threshold": 0.8,
        "cooldown": 3,  # seconds
        "max_history": 10  # for temporal analysis
    }
}


class FallDetector:
    def __init__(self):
        self.model = YOLO(CONFIG["model_path"])
        self.history = deque(maxlen = CONFIG["detection"]["max_history"])
        self.last_alert = 0
        self.prev_box = None

    def detect_fall(self, frame):
        results = self.model(frame, verbose = False)[0]
        current_boxes = results.boxes

        # Store current detection for comparison
        if len(current_boxes) > 0:
            # Get the person detection with highest confidence
            person_boxes = [box for box in current_boxes if int(box.cls.item()) == 0]
            if person_boxes:
                current_box = max(person_boxes, key = lambda x: x.conf.item())

                if self.prev_box is not None:
                    # Get bounding box coordinates
                    prev_coords = self.prev_box.xyxy[0]  # Previous frame coordinates
                    curr_coords = current_box.xyxy[0]  # Current frame coordinates

                    # Calculate height change ratio
                    prev_height = prev_coords[3] - prev_coords[1]
                    curr_height = curr_coords[3] - curr_coords[1]

                    # Calculate vertical position change
                    prev_y = prev_coords[1]  # y coordinate of top of bounding box
                    curr_y = curr_coords[1]

                    # Detect fall based on:
                    # 1. Significant height change (person becoming shorter in frame)
                    # 2. Significant downward movement
                    height_change_ratio = curr_height / prev_height if prev_height > 0 else 1
                    vertical_movement = curr_y - prev_y

                    # Thresholds for fall detection
                    if (height_change_ratio < 0.7 and vertical_movement > 30):
                        return True

                self.prev_box = current_box

        return False


class WebSocketManager:
    def __init__(self):
        self.clients = set()
        self.server = None

    async def handler(self, websocket):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                if message == "ping":
                    await websocket.send("pong")
        finally:
            self.clients.remove(websocket)

    async def broadcast(self, message):
        if not self.clients:
            return
        websockets.broadcast(self.clients, message)

    async def start_server(self):
        self.server = await websockets.serve(
                self.handler,
                CONFIG["websocket"]["host"],
                CONFIG["websocket"]["port"]
        )


async def main():
    # Initialize components
    detector = FallDetector()
    ws_manager = WebSocketManager()

    # Start WebSocket server
    await ws_manager.start_server()
    print(f"WebSocket server started at ws://{CONFIG['websocket']['host']}:{CONFIG['websocket']['port']}")

    # Video capture setup
    cap = cv2.VideoCapture(CONFIG["camera_index"])
    if not cap.isOpened():
        raise RuntimeError("Failed to open video capture")

    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break

            # Detect falls
            if detector.detect_fall(frame):
                current_time = time.time()
                if current_time - detector.last_alert > CONFIG["detection"]["cooldown"]:
                    print("🚨 Fall detected! Sending alert...")
                    detector.last_alert = current_time
                    await ws_manager.broadcast("fall_detected")

            # Display frame
            cv2.imshow("Fall Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if ws_manager.server:
            ws_manager.server.close()
            await ws_manager.server.wait_closed()


if __name__ == "__main__":
    with suppress(asyncio.CancelledError):
        asyncio.run(main())