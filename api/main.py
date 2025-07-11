import sys
import os
import shutil
import base64
import cv2
import asyncio
import threading
from typing import Any
import numpy as np
import uuid # Add this import
import json
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks, Form
from fastapi.responses import JSONResponse

# Add project root to path to allow importing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules.globals
from modules.face_analyser import get_one_face, get_unique_faces_from_target_image, get_unique_faces_from_target_video
from modules.processors.frame.core import get_frame_processors_modules
from modules.typing import ProcessingContext # Import the new context class
from modules.utilities import is_image, is_video

# Attempt to pre-load the face enhancer module to make it available for dynamic use.
# This can help avoid "not found" errors during live preview on some systems.
try:
    import modules.processors.frame.face_enhancer
except ImportError as e:
    print(f"Warning: Could not pre-load face_enhancer module: {e}")

# --- Globals for Server State ---
LIVE_SOURCE_IMAGE = None # Store the raw CV2 image, not the analyzed Face object
TEMP_SERVER_DIR = "temp_server_files"
os.makedirs(TEMP_SERVER_DIR, exist_ok=True)

_thread_local_data = threading.local() # Thread-local storage for models

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
    global LIVE_SOURCE_IMAGE
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        cv2_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # We only check for a face here, but store the raw image for thread-safe processing later
        if get_one_face(cv2_img):
            LIVE_SOURCE_IMAGE = cv2_img
            print("Server: Live source face set successfully.")
            return JSONResponse(status_code=200, content={"message": "Source face set successfully."})
        else:
            print("Server: No face found in the provided live source image.")
            return JSONResponse(status_code=400, content={"message": "No face found in source image."})
    except Exception as e:
        print(f"Server Error on /set-live-source: {e}")
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})


JOBS = {} # To store job status and results

def process_batch_job_sync(job_id: str, source_path: str, target_path: str, output_path: str, context: ProcessingContext):
    """
    The core processing logic for a batch job. Runs in a background thread.
    This function now uses a ProcessingContext object to avoid global state modification.
    """
    # Import utilities here to avoid circular dependencies at startup
    from modules.utilities import create_temp, extract_frames, get_temp_frame_paths, detect_fps, create_video, restore_audio, move_temp, clean_temp

    try:
        # Update context with paths for internal use by utilities and processors
        context.source_path = source_path
        context.target_path = target_path
        context.output_path = output_path
        
        if is_image(target_path):
            shutil.copy2(target_path, output_path)

            active_processors = context.frame_processors.copy()
            if context.fp_ui.get('face_enhancer'):
                active_processors.append('face_enhancer')

            for frame_processor in get_frame_processors_modules(active_processors):
                frame_processor.process_image(source_path, output_path, output_path, context) # Pass context
        elif is_video(target_path):
            create_temp(target_path) # This utility needs target_path, which is now in context
            extract_frames(target_path) # This utility needs target_path
            temp_frame_paths = get_temp_frame_paths(target_path) # This utility needs target_path
            
            active_processors = context.frame_processors.copy()
            if context.fp_ui.get('face_enhancer'):
                active_processors.append('face_enhancer')

            for frame_processor in get_frame_processors_modules(active_processors):
                frame_processor.process_video(source_path, temp_frame_paths, context)
            
            if context.keep_fps:
                fps = detect_fps(target_path)
                create_video(target_path, fps, context) # Pass context
            else:
                create_video(target_path, 30.0, context) # Pass context

            if context.keep_audio:
                restore_audio(target_path, output_path) # This utility needs output_path
            else:
                move_temp(target_path, output_path) # This utility needs output_path
            
            clean_temp(target_path, context) # This function now respects context.keep_frames
        
        JOBS[job_id]['status'] = 'completed'
        JOBS[job_id]['result_path'] = output_path

    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        JOBS[job_id]['status'] = 'failed'
        JOBS[job_id]['error'] = str(e)
        
        # Clean up temporary source and target files
        if os.path.exists(source_path):
            os.remove(source_path)
        if os.path.exists(target_path):
            os.remove(target_path)

@app.post("/process-batch-job")
async def process_batch_job_endpoint(background_tasks: BackgroundTasks, source_file: UploadFile = File(...), target_file: UploadFile = File(...), options: str = Form(...)):
    job_id = str(uuid.uuid4())
    options_dict = json.loads(options)
    source_path = os.path.join(TEMP_SERVER_DIR, f"{job_id}_{source_file.filename}")
    target_path = os.path.join(TEMP_SERVER_DIR, f"{job_id}_{target_file.filename}")
    with open(source_path, "wb") as buffer:
        shutil.copyfileobj(source_file.file, buffer)
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(target_file.file, buffer)
    _, target_extension = os.path.splitext(target_file.filename)
    output_path = os.path.join(TEMP_SERVER_DIR, f"{job_id}_output{target_extension}")
    JOBS[job_id] = {"status": "processing", "result_path": None, "error": None}
    processing_context = ProcessingContext(**options_dict) # Create context from client options
    background_tasks.add_task(process_batch_job_sync, job_id, source_path, target_path, output_path, processing_context)
    return {"message": "Job initiated", "job_id": job_id}

@app.get("/download-result/{job_id}")
async def download_result_endpoint(job_id: str):
    from fastapi.responses import FileResponse
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"message": "Job not found"})
    if job['status'] == 'processing':
        return JSONResponse(status_code=202, content={"message": "Job is still processing"})
    if job['status'] == 'failed':
        return JSONResponse(status_code=500, content={"message": f"Job failed: {job['error']}"})
    if job['status'] == 'completed' and job['result_path'] and os.path.exists(job['result_path']):
        return FileResponse(job['result_path'], media_type='application/octet-stream', filename=os.path.basename(job['result_path']))
    return JSONResponse(status_code=404, content={"message": "Result file not found"})

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

def get_thread_local_models_and_source_face() -> tuple[Any, Any, Any]:
    """
    Initializes and returns thread-local models and the source face.
    This is the core of the CUDA thread-safety fix.
    """
    if not hasattr(_thread_local_data, 'face_analyser'):
        # This block runs once per thread
        print(f"Initializing models for thread: {threading.get_ident()}")
        # Import modules locally to avoid circular dependencies at startup
        import insightface
        import modules.processors.frame.face_swapper as face_swapper_module
        import modules.processors.frame.face_enhancer as face_enhancer_module

        # Analyser
        analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=modules.globals.execution_providers)
        analyser.prepare(ctx_id=0)
        _thread_local_data.face_analyser = analyser

        # Processors
        _thread_local_data.processors = {
            'face_swapper': face_swapper_module,
            'face_enhancer': face_enhancer_module
        }

    # Always re-analyze the source image to get a thread-local Face object.
    # This is optimized to only re-run analysis if the global source image has changed.
    if not hasattr(_thread_local_data, 'source_image_id') or _thread_local_data.source_image_id != id(LIVE_SOURCE_IMAGE):
        if LIVE_SOURCE_IMAGE is not None:
            print(f"Thread {threading.get_ident()}: New source image detected. Re-analyzing.")
            faces = _thread_local_data.face_analyser.get(LIVE_SOURCE_IMAGE)
            if faces:
                _thread_local_data.source_face = sorted(faces, key=lambda x: x.bbox[0])[0]
            else:
                _thread_local_data.source_face = None
        else:
            _thread_local_data.source_face = None
        _thread_local_data.source_image_id = id(LIVE_SOURCE_IMAGE)
        
    return (
        _thread_local_data.face_analyser,
        _thread_local_data.processors,
        _thread_local_data.source_face
    )

def process_and_encode_sync(payload: dict, context: ProcessingContext) -> str:
    """
    Synchronous function to handle all CPU-bound processing.
    This is run in a separate thread and uses thread-local models for safety.
    """
    # Get thread-local models. This will initialize them on the first call for this thread.
    face_analyser, processors, source_face = get_thread_local_models_and_source_face() 

    # The payload is now a pre-parsed dictionary.
    # The context object is the single source of truth for all options.

    # Decode the frame
    frame_b64 = payload['frame']
    img_data = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    processed_frame = frame

    # Build the list of active processors for this frame using thread-local instances
    active_processor_modules = [] # These are the actual processor module objects
    if 'face_swapper' in context.frame_processors:
        active_processor_modules.append(processors['face_swapper'])
    if context.fp_ui.get('face_enhancer'):
        active_processor_modules.append(processors['face_enhancer'])

    if context.map_faces:
        if context.simple_map: # Use context's simple_map
            for fp in active_processor_modules:
                processed_frame = fp.process_frame_v2(processed_frame, context) # Pass context
        else:
            print("Server: simple_map not received for multi-face processing. Skipping.")
    elif source_face: # Use the thread-local source_face
        # Use the thread-local face_analyser to find faces in the target frame
        if context.many_faces:
            target_faces = face_analyser.get(processed_frame)
        else:
            target_faces = face_analyser.get(processed_frame)
            if target_faces:
                # Get the first face like get_one_face does
                target_faces = [sorted(target_faces, key=lambda x: x.bbox[0])[0]]
        
        if target_faces and target_faces[0] is not None:
            for fp in active_processor_modules:
                processed_frame = fp.process_frame(source_face, processed_frame, target_faces, context) # Pass context
    
    # Encode the processed frame
    _, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return base64.b64encode(buffer).decode('utf-8')

async def receive_frames(websocket: WebSocket, latest_frame: LatestFrame):
    """Task to continuously receive frames and update the latest one."""
    try:
        while True:
            data = await websocket.receive_text()
            await latest_frame.set(data)
    except WebSocketDisconnect:
        print("Receiver task: WebSocket disconnected. Shutting down.")

async def process_and_send_frames(websocket: WebSocket, latest_frame: LatestFrame, context: ProcessingContext):
    """Task to process the latest available frame and send it back."""
    last_options = {} # Keep track of the last options received to avoid redundant context updates
    try:
        while True:
            payload_data = await latest_frame.get()
            if payload_data:
                payload = json.loads(payload_data)
                options = payload.get('options', {})

                # Only update the context object if the options have actually changed.
                if options != last_options:
                    for key, value in options.items():
                        if hasattr(context, key):
                            setattr(context, key, value)
                    last_options = options

                processed_frame_b64 = await asyncio.to_thread(process_and_encode_sync, payload, context)
                await websocket.send_text(processed_frame_b64)
            else:
                await asyncio.sleep(0.01) # Avoid busy-waiting
    except WebSocketDisconnect:
        print("Processor task: WebSocket disconnected. Shutting down.")

@app.websocket("/ws/live-preview")
async def websocket_live_preview(websocket: WebSocket):
    await websocket.accept()
    latest_frame = LatestFrame()
    session_context = ProcessingContext() # Create a single context for the entire session

    receiver_task = asyncio.create_task(receive_frames(websocket, latest_frame))
    processor_task = asyncio.create_task(process_and_send_frames(websocket, latest_frame, session_context))

    try:
        done, pending = await asyncio.wait(
            [receiver_task, processor_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in done:
            if task.exception() is not None:
                raise task.exception()
    except WebSocketDisconnect:
        print("Client disconnected gracefully.")
    except Exception as e:
        print(f"Error in WebSocket: {e}")
    finally:
        print("WebSocket session ended.")