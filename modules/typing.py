from typing import Any, List, Dict

import numpy

Face = Any # Define Face as Any to avoid direct insightface import on client
Frame = numpy.ndarray[Any, Any]

class ProcessingContext:
    """
    Encapsulates all options and state relevant to a single processing job.
    This avoids reliance on mutable global state (modules.globals) for concurrent operations.
    """
    def __init__(self, **kwargs):
        # Initialize with default values or values from kwargs
        self.frame_processors: List[str] = kwargs.get('frame_processors', [])
        self.keep_fps: bool = kwargs.get('keep_fps', True)
        self.keep_audio: bool = kwargs.get('keep_audio', True)
        self.keep_frames: bool = kwargs.get('keep_frames', False)
        self.many_faces: bool = kwargs.get('many_faces', False)
        self.map_faces: bool = kwargs.get('map_faces', False)
        self.color_correction: bool = kwargs.get('color_correction', False)
        self.nsfw_filter: bool = kwargs.get('nsfw_filter', False)
        self.video_encoder: str = kwargs.get('video_encoder', 'libx264')
        self.video_quality: int = kwargs.get('video_quality', 18)
        self.mouth_mask: bool = kwargs.get('mouth_mask', False)
        self.mask_feather_ratio: int = kwargs.get('mask_feather_ratio', 8)
        self.mask_down_size: float = kwargs.get('mask_down_size', 0.5)
        self.mask_size: int = kwargs.get('mask_size', 1)
        self.show_mouth_mask_box: bool = kwargs.get('show_mouth_mask_box', False)
        self.simple_map: Dict[str, Any] = kwargs.get('simple_map', {})
        self.source_target_map: List[Dict[str, Any]] = kwargs.get('source_target_map', [])
        self.fp_ui: Dict[str, bool] = kwargs.get('fp_ui', {"face_enhancer": False})
        self.execution_providers: List[str] = kwargs.get('execution_providers', ['cpu'])
        self.execution_threads: int = kwargs.get('execution_threads', 8)

        # Paths are specific to the job and will be set by the server
        self.source_path: str = kwargs.get('source_path', '')
        self.target_path: str = kwargs.get('target_path', '')
        self.output_path: str = kwargs.get('output_path', '')
