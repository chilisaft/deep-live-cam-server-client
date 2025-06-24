import requests
import os
import base64
import numpy as np
import cv2
import modules.globals # Import modules.globals

class ReconstructedFace:
    """
    A client-side object that duck-types insightface.app.common.Face
    to avoid issues with immutable or slotted original Face objects.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def get_server_url() -> str:
    """Constructs the server URL dynamically based on modules.globals.port."""
    return f"http://{modules.globals.server_ip}:{modules.globals.port}"

def check_server_status() -> bool:
    """
    Pings the server's root endpoint to check for a connection.
    """
    try:
        response = requests.get(get_server_url(), timeout=2)
        # Check for a 2xx status code
        if response.ok:
            print(f"Successfully connected to server at {get_server_url()}")
            return True
    except requests.exceptions.RequestException:
        print(f"API Client Error: Could not connect to server at {get_server_url()}.")
        print("Please ensure the server is running with 'python run.py --mode server'")
    return False

def set_live_source(source_path: str) -> bool:
    """
    Uploads the source image to the server to be used for the live session.
    """
    endpoint = f"{get_server_url()}/set-live-source"
    print(f"Client: Setting live source image on server: {os.path.basename(source_path)}")

    with open(source_path, 'rb') as f:
        files = {'file': (os.path.basename(source_path), f, 'image/jpeg')}
        try:
            response = requests.post(endpoint, files=files, timeout=30)
            if response.ok:
                print("Client: Server confirmed live source.")
                return True
            else:
                print(f"Client: Server failed to set live source. Status: {response.status_code}, Message: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"API Client Error: Could not set live source on server.")
            print(f"Details: {e}")
            return False

def request_face_analysis(target_path: str) -> list:
    """
    Sends the target file to the server for face analysis and returns
    the reconstructed face map.
    """ 
    endpoint = f"{get_server_url()}/analyze-faces"
    print(f"Client: Sending {os.path.basename(target_path)} to {endpoint}")

    with open(target_path, 'rb') as f:
        files = {'file': (os.path.basename(target_path), f, 'application/octet-stream')}
        try:
            # Use a long timeout for potentially slow video processing on the server
            response = requests.post(endpoint, files=files, timeout=600)
            response.raise_for_status()

            response_data = response.json()
            if response_data.get("data"):
                return deserialize_face_map(response_data["data"])
            else:
                print(f"API Client: {response_data.get('message')}")
                return []

        except requests.exceptions.RequestException as e: # Use get_server_url() here too
            print(f"API Client Error: Could not connect to server at {get_server_url()}.")
            print("Please ensure the server is running with 'python run.py --mode server'")
            print(f"Details: {e}")
            return []

def deserialize_face_map(serialized_map: list) -> list:
    """
    Deserializes the map data from the server, converting Base64 images back
    to CV2 format and dicts back to a custom, mutable Face-like object.
    """
    reconstructed_map = []
    for item_data in serialized_map:
        reconstructed_item = {'id': item_data['id']}
        if 'target' in item_data:
            target_data = item_data['target']

            # Deserialize Base64 to CV2 image
            img_data = base64.b64decode(target_data['cv2_base64'])
            np_arr = np.frombuffer(img_data, np.uint8)
            cv2_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Deserialize dict to our custom Face-like object
            face_data = target_data['face']

            landmark_2d_106_data = face_data.get('landmark_2d_106')

            face_attributes = {
                'bbox': np.array(face_data['bbox'], dtype=np.float32),
                'kps': np.array(face_data['kps'], dtype=np.float32),
                'det_score': float(face_data['det_score']),
                'normed_embedding': np.array(face_data.get('normed_embedding', []), dtype=np.float32),
                'landmark_2d_106': np.array(landmark_2d_106_data, dtype=np.float32) if landmark_2d_106_data is not None else None
            }

            face = ReconstructedFace(**face_attributes)

            reconstructed_item['target'] = { 'cv2': cv2_img, 'face': face }
        reconstructed_map.append(reconstructed_item)
    return reconstructed_map

def initiate_batch_processing(source_path: str, target_path: str, options: dict) -> dict:
    """
    Sends source and target files along with processing options to the server
    to initiate a batch processing job.
    """
    endpoint = f"{get_server_url()}/process-batch-job"
    print(f"Client: Initiating batch job for {os.path.basename(target_path)} on server.")

    files = {
        'source_file': (os.path.basename(source_path), open(source_path, 'rb'), 'application/octet-stream'),
        'target_file': (os.path.basename(target_path), open(target_path, 'rb'), 'application/octet-stream'),
        'options': (None, json.dumps(options), 'application/json') # Send options as JSON string
    }

    try:
        # Use a long timeout for potentially very long processing jobs
        response = requests.post(endpoint, files=files, timeout=3600) # 1 hour timeout
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Client Error: Could not initiate batch processing on server.")
        print(f"Details: {e}")
        return {"message": f"Error: {e}"}
    finally:
        # Close file handles
        if 'source_file' in files and files['source_file'][1]:
            files['source_file'][1].close()
        if 'target_file' in files and files['target_file'][1]:
            files['target_file'][1].close()

def download_processed_result(job_id: str, output_path: str) -> bool:
    """
    Downloads the processed file from the server for a given job_id.
    """
    endpoint = f"{get_server_url()}/download-result/{job_id}"
    print(f"Client: Downloading result for job {job_id} to {output_path}")

    try:
        response = requests.get(endpoint, stream=True, timeout=3600) # Long timeout for download
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Client: Download complete for job {job_id}.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"API Client Error: Could not download result for job {job_id}.")
        print(f"Details: {e}")
        return False