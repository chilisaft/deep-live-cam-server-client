from typing import Any, List
import cv2
import threading
import torch

from modules.typing import Frame, ProcessingContext
from modules.utilities import resolve_relative_path, conditional_download

_thread_local_face_enhancer = threading.local()
NAME = 'DLC.FACE-ENHANCER'


def get_face_enhancer() -> Any:
	if not hasattr(_thread_local_face_enhancer, 'model'):
		# This is thread-safe. Each thread will initialize its own model instance once.
		# Imports are safe here because this is only called from a worker thread
		# after the main app has started, avoiding circular dependencies.
		from gfpgan.utils import GFPGANer
		model_path = resolve_relative_path('../../models/GFPGANv1.4.pth')
		_thread_local_face_enhancer.model = GFPGANer(
			model_path=model_path,
			upscale=1,
			arch='clean',
			channel_multiplier=2,
			bg_upsampler=None)
	return _thread_local_face_enhancer.model


def pre_check() -> bool:
	download_directory_path = resolve_relative_path('../../models')
	conditional_download(download_directory_path, ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'])
	return True


def enhance_face(target_face: Frame, temp_frame: Frame) -> Frame:
	start_x, start_y, end_x, end_y = map(int, target_face.bbox)
	padding_x = int((end_x - start_x) * 0.5)
	padding_y = int((end_y - start_y) * 0.5)
	start_x = max(0, start_x - padding_x)
	start_y = max(0, start_y - padding_y)
	end_x = min(temp_frame.shape[1], end_x + padding_x)
	end_y = min(temp_frame.shape[0], end_y + padding_y)
	crop_frame = temp_frame[start_y:end_y, start_x:end_x]
	if crop_frame.size:
		with torch.no_grad():
			_, _, crop_frame = get_face_enhancer().enhance(
				crop_frame,
				has_aligned=False,
				only_center_face=True,
				paste_back=True)
		temp_frame[start_y:end_y, start_x:end_x] = crop_frame
	return temp_frame


def process_frame(source_face: Any, temp_frame: Frame, target_faces: List[Any], context: ProcessingContext = None) -> Frame:
	for target_face in target_faces:
		temp_frame = enhance_face(target_face, temp_frame)
	return temp_frame


def process_image(source_path: str, target_path: str, output_path: str, context: ProcessingContext = None) -> None:
	from modules.core import update_status
	target_frame = cv2.imread(target_path)
	# In a real scenario, you'd get target faces herez
	# result = process_frame(None, target_frame, get_many_faces(target_frame), context)
	# cv2.imwrite(output_path, result)
	update_status('Processing...', NAME)


def process_video(source_path: str, temp_frame_paths: List[str], context: ProcessingContext = None) -> None:
	from modules.core import update_status
	update_status('Processing...', NAME)
	# for temp_frame_path in temp_frame_paths:
	# 	temp_frame = cv2.imread(temp_frame_path)
	# 	result = process_frame(None, temp_frame, get_many_faces(temp_frame), context)
	# 	cv2.imwrite(temp_frame_path, result)