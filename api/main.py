import sys
import os
import shutil
import base64
import cv2
import asyncio
import numpy as np
import uuid # Add this import
import json
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# Add project root to path to allow importing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules.globals
from modules.face_analyser import get_one_face, get_unique_faces_from_target_image, get_unique_faces_from_target_video
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import is_image, is_video

# --- Globals for Server State ---
LIVE_SOURCE_FACE = None
# LIVE_SIMPLE_MAP = None # Not needed, we'll use modules.globals.simple_map directly
TEMP_SERVER_DIR = "temp_server_files"
os.makedirs(TEMP_SERVER_DIR, exist_ok=True)

# The execution provider is now set by the command-line arguments in `run.py`
# when starting the server with `--mode server`.
# For example: python run.py --mode server --execution-provider directml

app = FastAPI(title="Deep-Live-Cam Backend Server")

def serialize_face_map(face_map: list) -> list:
    """
    Serializes the face map data, converting CV2 images to Base64 strings
    and InsightFace objects to dictionaries for JSON compatibility.
    """
    serialized_map = []
    for item in face_map:
        serialized_item = {'id': item['id']}
        if 'target' in item:
            target_data = item['target']
            
            # Serialize CV2 image to Base64
            _, buffer = cv2.imencode('.png', target_data['cv2'])
            cv2_base64 = base64.b64encode(buffer).decode('utf-8')

            # Serialize InsightFace object
            face_obj = target_data['face']
            serialized_face = {
                'bbox': face_obj.bbox.tolist(),
                'kps': face_obj.kps.tolist(),
                'det_score': float(face_obj.det_score),
                'normed_embedding': face_obj.normed_embedding.tolist(),
                'landmark_2d_106': face_obj.landmark_2d_106.tolist() if hasattr(face_obj, 'landmark_2d_106') and face_obj.landmark_2d_106 is not None else None
            }

            serialized_item['target'] = {
                'cv2_base64': cv2_base64,
                'face': serialized_face
            }
        serialized_map.append(serialized_item)
    return serialized_map

@app.get("/")
def read_root():
    """
    Root endpoint to check if the server is running.
    """
    return {"status": "Deep-Live-Cam Server is running"}

@app.post("/analyze-faces")
async def analyze_faces_endpoint(file: UploadFile = File(...)):
    """
    Receives a target file, analyzes it for faces, and returns the face map.
    """
    try:
        temp_file_path = os.path.join(TEMP_SERVER_DIR, file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        face_map = []
        if is_image(temp_file_path):
            print("Server: Analyzing image...")
            face_map = get_unique_faces_from_target_image(temp_file_path)
        elif is_video(temp_file_path):
            print("Server: Analyzing video...")
            face_map = get_unique_faces_from_target_video(temp_file_path)
        else:
            return JSONResponse(status_code=400, content={"message": "Unsupported file type"})

        os.remove(temp_file_path)

        if not face_map:
            return JSONResponse(status_code=200, content={"message": "No faces found", "data": []})

        serialized_data = serialize_face_map(face_map)
        return JSONResponse(status_code=200, content={"message": "Analysis successful", "data": serialized_data})

    except Exception as e:
        print(f"Server Error: {e}")
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})

@app.post("/set-live-source")
async def set_live_source_endpoint(file: UploadFile = File(...)):
    """
    Receives a source image for the live session, analyzes it for a single face,
    and stores it in the server's memory.
    """
    global LIVE_SOURCE_FACE
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        cv2_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        LIVE_SOURCE_FACE = get_one_face(cv2_img)

        if LIVE_SOURCE_FACE:
            print("Server: Live source face set successfully.")
            return JSONResponse(status_code=200, content={"message": "Source face set successfully."})
        else:
            print("Server: No face found in the provided live source image.")
            return JSONResponse(status_code=400, content={"message": "No face found in source image."})
    except Exception as e:
        print(f"Server Error on /set-live-source: {e}")
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})


class LatestFrame:
    """
    A thread-safe class to hold the latest frame data from the client.
    This ensures the processor always works on the most recent frame,
    discarding stale ones, which is crucial for a low-latency live preview.
    """
    def __init__(self):
        self.frame_data = None
        self.lock = asyncio.Lock()

    async def set(self, frame_data: str):
        async with self.lock:
            self.frame_data = frame_data

    async def get(self) -> str | None:
        async with self.lock:
            frame_data = self.frame_data
            self.frame_data = None  # Consume the frame
            return frame_data

def process_and_encode_sync(payload_data: str) -> str:
    """
    Synchronous function to handle all CPU-bound processing.
    This is run in a separate thread to avoid blocking the asyncio event loop.
    """
    payload = json.loads(payload_data)

    # Update server's global state with client's options
    options = payload.get('options', {})
    for key, value in options.items():
        if hasattr(modules.globals, key):
            setattr(modules.globals, key, value)
    if 'fp_ui' in options:
        modules.globals.fp_ui = options['fp_ui']

    if 'simple_map' in payload:
        modules.globals.simple_map = payload['simple_map']

    # Decode the frame
    frame_b64 = payload['frame']
    img_data = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    processed_frame = frame

    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)

    if modules.globals.map_faces:
        if modules.globals.simple_map:
            for fp in frame_processors:
                processed_frame = fp.process_frame_v2(processed_frame)
        else:
            print("Server: simple_map not received for multi-face processing. Skipping.")
    elif LIVE_SOURCE_FACE:
        from modules.face_analyser import get_one_face, get_many_faces
        
        if modules.globals.many_faces:
            target_faces = get_many_faces(processed_frame)
        else:
            target_faces = [get_one_face(processed_frame)]
        
        if target_faces and target_faces[0] is not None:
            for fp in frame_processors:
                processed_frame = fp.process_frame(LIVE_SOURCE_FACE, processed_frame, target_faces)
        else:
            print("Server: No faces detected in live frame for single-face mode. Skipping processing.")
    else:
        print("Server: No source face set for single-face mode. Skipping processing.")

    # Encode the processed frame
    _, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return base64.b64encode(buffer).decode('utf-8')

async def receive_frames(websocket: WebSocket, latest_frame: LatestFrame):
    """Task to continuously receive frames and update the latest one."""
    while True:
        data = await websocket.receive_text()
        await latest_frame.set(data)

async def process_and_send_frames(websocket: WebSocket, latest_frame: LatestFrame):
    """Task to process the latest available frame and send it back."""
    while True:
        payload_data = await latest_frame.get()
        if payload_data:
            processed_frame_b64 = await asyncio.to_thread(process_and_encode_sync, payload_data)
            await websocket.send_text(processed_frame_b64)
        else:
            await asyncio.sleep(0.01) # Avoid busy-waiting

@app.websocket("/ws/live-preview")
async def websocket_live_preview(websocket: WebSocket):
    await websocket.accept()
    latest_frame = LatestFrame()
    from modules.face_analyser import get_face_analyser
    get_face_analyser() # Ensure analyser is initialized in the main thread before starting tasks

    receiver_task = asyncio.create_task(receive_frames(websocket, latest_frame))
    processor_task = asyncio.create_task(process_and_send_frames(websocket, latest_frame))

    try:
        done, pending = await asyncio.wait(
            [receiver_task, processor_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        # Re-raise exceptions from completed tasks if any
        for task in done:
            if task.exception() is not None:
                raise task.exception()
    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        print("WebSocket session ended.")