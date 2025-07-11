import os
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple, Any, List, Dict
import cv2
from cv2_enumerate_cameras import enumerate_cameras  # Add this import
from PIL import Image, ImageOps
import time
import numpy as np
import json
import modules.globals
import base64 # Add this import
import modules.metadata
import websocket # Add this import
import threading
import queue
import modules.api_client as api_client
from modules.face_analyser import add_blank_map, has_valid_map, simplify_maps
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    is_image,
    is_video,
    resolve_relative_path,
    has_image_extension,
)
from modules.video_capture import VideoCapturer
from modules.gettext import LanguageManager
import platform

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph


class ClientState:
    """
    Manages the client-side UI state, including selected paths,
    processing options, and mapping data.
    """

    def __init__(self):
        self.source_target_map: List[Dict[str, Any]] = []
        self.simple_map: Dict[str, Any] = {}

        self.source_path: str = None
        self.target_path: str = None
        self.output_path: str = None

        # UI Toggles and Options
        self.keep_fps: bool = True
        self.keep_audio: bool = True
        self.keep_frames: bool = False
        self.many_faces: bool = False
        self.map_faces: bool = False
        self.color_correction: bool = False
        self.nsfw_filter: bool = False  # Currently unused in UI, but kept for consistency
        self.video_encoder: str = "libx264"  # Default, can be overridden by CLI
        self.video_quality: int = 18  # Default, can be overridden by CLI
        self.live_mirror: bool = False
        self.live_resizable: bool = False
        self.fp_ui: Dict[str, bool] = {"face_enhancer": False}
        self.webcam_preview_running: bool = False
        self.show_fps: bool = False
        self.mouth_mask: bool = False
        self.show_mouth_mask_box: bool = False

        # These are set by core.py from CLI args, but UI needs to know them
        # They are not directly modified by UI switches, but reflect app config.
        # We'll initialize them from modules.globals in init()
        self.frame_processors: List[str] = []
        self.server_ip: str = "127.0.0.1"
        self.port: int = 8000
        self.headless: bool = False


ROOT = None
POPUP = None
POPUP_LIVE = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 960
PREVIEW_DEFAULT_HEIGHT = 540

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = (740,)
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = (890,)
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_MAX_HEIGHT = 100
MAPPER_PREVIEW_MAX_WIDTH = 100

DEFAULT_BUTTON_WIDTH = 200
DEFAULT_BUTTON_HEIGHT = 40

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

_ = None
preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None
popup_status_label = None
popup_status_label_live = None
source_label_dict = {}
source_label_dict_live = {}
target_label_dict_live = {}

_ = None # Initialized by LanguageManager

CLIENT_STATE: ClientState = None # Global instance of ClientState

img_ft, vid_ft = modules.globals.file_types # These are constants, keep in modules.globals


def init(start: Callable[[], None], destroy: Callable[[], None], lang: str) -> ctk.CTk:
    global ROOT, PREVIEW, _, CLIENT_STATE

    lang_manager = LanguageManager(lang)
    _ = lang_manager._

    CLIENT_STATE = ClientState()
    # Initialize ClientState with values from modules.globals that are set by CLI args
    CLIENT_STATE.frame_processors = modules.globals.frame_processors
    CLIENT_STATE.server_ip = modules.globals.server_ip
    CLIENT_STATE.port = modules.globals.port
    CLIENT_STATE.headless = modules.globals.headless
    CLIENT_STATE.fp_ui = modules.globals.fp_ui.copy() # Sync UI state with CLI args

    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT

def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            switch_states = json.load(f)
        CLIENT_STATE.keep_fps = switch_states.get("keep_fps", True)
        CLIENT_STATE.keep_audio = switch_states.get("keep_audio", True)
        CLIENT_STATE.keep_frames = switch_states.get("keep_frames", False)
        CLIENT_STATE.many_faces = switch_states.get("many_faces", False)
        CLIENT_STATE.map_faces = switch_states.get("map_faces", False)
        CLIENT_STATE.color_correction = switch_states.get("color_correction", False)
        CLIENT_STATE.nsfw_filter = switch_states.get("nsfw_filter", False)
        CLIENT_STATE.live_mirror = switch_states.get("live_mirror", False)
        CLIENT_STATE.live_resizable = switch_states.get("live_resizable", False)
        CLIENT_STATE.fp_ui = switch_states.get("fp_ui", {"face_enhancer": False})
        CLIENT_STATE.show_fps = switch_states.get("show_fps", False)
        CLIENT_STATE.mouth_mask = switch_states.get("mouth_mask", False)
        CLIENT_STATE.show_mouth_mask_box = switch_states.get("show_mouth_mask_box", False)
    except FileNotFoundError:
        # If the file doesn't exist, use default values
        pass

def save_switch_states():
    switch_states = {
        "keep_fps": CLIENT_STATE.keep_fps,
        "keep_audio": CLIENT_STATE.keep_audio,
        "keep_frames": CLIENT_STATE.keep_frames,
        "many_faces": CLIENT_STATE.many_faces,
        "map_faces": CLIENT_STATE.map_faces,
        "color_correction": CLIENT_STATE.color_correction,
        "nsfw_filter": CLIENT_STATE.nsfw_filter,
        "live_mirror": CLIENT_STATE.live_mirror,
        "live_resizable": CLIENT_STATE.live_resizable,
        "fp_ui": CLIENT_STATE.fp_ui,
        "show_fps": CLIENT_STATE.show_fps,
        "mouth_mask": CLIENT_STATE.mouth_mask,
        "show_mouth_mask_box": CLIENT_STATE.show_mouth_mask_box,
    }
    with open("switch_states.json", "w") as f:
        json.dump(switch_states, f)


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label, show_fps_switch

    load_switch_states()

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme(resolve_relative_path("ui.json"))

    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(
        f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}"
    )
    root.configure()
    root.protocol("WM_DELETE_WINDOW", lambda: destroy())

    source_label = ctk.CTkLabel(root, text=None)
    source_label.place(relx=0.1, rely=0.1, relwidth=0.3, relheight=0.25)

    target_label = ctk.CTkLabel(root, text=None)
    target_label.place(relx=0.6, rely=0.1, relwidth=0.3, relheight=0.25)

    select_face_button = ctk.CTkButton(
        root, text=_("Select a face"), cursor="hand2", command=lambda: select_source_path()
    )
    select_face_button.place(relx=0.1, rely=0.4, relwidth=0.3, relheight=0.1)

    swap_faces_button = ctk.CTkButton(
        root, text="↔", cursor="hand2", command=lambda: swap_faces_paths()
    )
    swap_faces_button.place(relx=0.45, rely=0.4, relwidth=0.1, relheight=0.1)

    select_target_button = ctk.CTkButton(
        root,
        text=_("Select a target"),
        cursor="hand2",
        command=lambda: select_target_path(),
    )
    select_target_button.place(relx=0.6, rely=0.4, relwidth=0.3, relheight=0.1)

    keep_fps_value = ctk.BooleanVar(value=CLIENT_STATE.keep_fps)
    keep_fps_checkbox = ctk.CTkSwitch(
        root,
        text=_("Keep fps"),
        variable=keep_fps_value,
        cursor="hand2",
        command=lambda: (
            setattr(CLIENT_STATE, "keep_fps", keep_fps_value.get()),
            save_switch_states(),
        ),
    )
    keep_fps_checkbox.place(relx=0.1, rely=0.6)

    keep_frames_value = ctk.BooleanVar(value=CLIENT_STATE.keep_frames)
    keep_frames_switch = ctk.CTkSwitch(
        root,
        text=_("Keep frames"),
        variable=keep_frames_value,
        cursor="hand2",
        command=lambda: (
            setattr(CLIENT_STATE, "keep_frames", keep_frames_value.get()),
            save_switch_states(),
        ),
    )
    keep_frames_switch.place(relx=0.1, rely=0.65)

    enhancer_value = ctk.BooleanVar(value=CLIENT_STATE.fp_ui["face_enhancer"])
    enhancer_switch = ctk.CTkSwitch(
        root,
        text=_("Face Enhancer"),
        variable=enhancer_value,
        cursor="hand2",
        command=lambda: (
            update_tumbler("face_enhancer", enhancer_value.get()),
            save_switch_states(),
        ),
    )
    enhancer_switch.place(relx=0.1, rely=0.7)

    keep_audio_value = ctk.BooleanVar(value=CLIENT_STATE.keep_audio)
    keep_audio_switch = ctk.CTkSwitch(
        root,
        text=_("Keep audio"),
        variable=keep_audio_value,
        cursor="hand2",
        command=lambda: (
            setattr(CLIENT_STATE, "keep_audio", keep_audio_value.get()),
            save_switch_states(),
        ),
    )
    keep_audio_switch.place(relx=0.6, rely=0.6)

    many_faces_value = ctk.BooleanVar(value=CLIENT_STATE.many_faces)
    many_faces_switch = ctk.CTkSwitch(
        root,
        text=_("Many faces"),
        variable=many_faces_value,
        cursor="hand2",
        command=lambda: (
            setattr(CLIENT_STATE, "many_faces", many_faces_value.get()),
            save_switch_states(),
        ),
    )
    many_faces_switch.place(relx=0.6, rely=0.65)

    color_correction_value = ctk.BooleanVar(value=CLIENT_STATE.color_correction)
    color_correction_switch = ctk.CTkSwitch(
        root,
        text=_("Fix Blueish Cam"),
        variable=color_correction_value,
        cursor="hand2",
        command=lambda: (
            setattr(CLIENT_STATE, "color_correction", color_correction_value.get()),
            save_switch_states(),
        ),
    )
    color_correction_switch.place(relx=0.6, rely=0.70)

    #    nsfw_value = ctk.BooleanVar(value=modules.globals.nsfw_filter)
    #    nsfw_switch = ctk.CTkSwitch(root, text='NSFW filter', variable=nsfw_value, cursor='hand2', command=lambda: setattr(modules.globals, 'nsfw_filter', nsfw_value.get()))
    #    nsfw_switch.place(relx=0.6, rely=0.7)

    map_faces = ctk.BooleanVar(value=CLIENT_STATE.map_faces)
    map_faces_switch = ctk.CTkSwitch(
        root,
        text=_("Map faces"),
        variable=map_faces,
        cursor="hand2",
        command=lambda: (
            setattr(CLIENT_STATE, "map_faces", map_faces.get()),
            save_switch_states(),
            close_mapper_window() if not map_faces.get() else None
        ),
    )
    map_faces_switch.place(relx=0.1, rely=0.75)

    show_fps_value = ctk.BooleanVar(value=CLIENT_STATE.show_fps)
    show_fps_switch = ctk.CTkSwitch(
        root,
        text=_("Show FPS"),
        variable=show_fps_value,
        cursor="hand2",
        command=lambda: (
            setattr(CLIENT_STATE, "show_fps", show_fps_value.get()),
            save_switch_states(),
        ),
    )
    show_fps_switch.place(relx=0.6, rely=0.75)

    mouth_mask_var = ctk.BooleanVar(value=CLIENT_STATE.mouth_mask)
    mouth_mask_switch = ctk.CTkSwitch(
        root,
        text=_("Mouth Mask"),
        variable=mouth_mask_var,
        cursor="hand2",
        command=lambda: setattr(CLIENT_STATE, "mouth_mask", mouth_mask_var.get()),
    )
    mouth_mask_switch.place(relx=0.1, rely=0.55)

    show_mouth_mask_box_var = ctk.BooleanVar(value=CLIENT_STATE.show_mouth_mask_box)
    show_mouth_mask_box_switch = ctk.CTkSwitch(
        root,
        text=_("Show Mouth Mask Box"),
        variable=show_mouth_mask_box_var,
        cursor="hand2",
        command=lambda: setattr(
            CLIENT_STATE, "show_mouth_mask_box", show_mouth_mask_box_var.get()
        ),
    )
    show_mouth_mask_box_switch.place(relx=0.6, rely=0.55)

    start_button = ctk.CTkButton(
        root, text=_("Start"), cursor="hand2", command=lambda: analyze_target(start)
    )
    start_button.place(relx=0.15, rely=0.80, relwidth=0.2, relheight=0.05)

    stop_button = ctk.CTkButton(
        root, text=_("Destroy"), cursor="hand2", command=lambda: destroy()
    )
    stop_button.place(relx=0.4, rely=0.80, relwidth=0.2, relheight=0.05)

    preview_button = ctk.CTkButton(
        root,
        text=_("Preview (Disabled)"),
        state="disabled",
        # command=lambda: toggle_preview() # Command removed
    )
    preview_button.place(relx=0.65, rely=0.80, relwidth=0.2, relheight=0.05)

    # --- Camera Selection ---
    camera_label = ctk.CTkLabel(root, text=_("Select Camera:"))
    camera_label.place(relx=0.1, rely=0.86, relwidth=0.2, relheight=0.05)

    available_cameras = get_available_cameras()
    camera_indices, camera_names = available_cameras

    if not camera_names or camera_names[0] == "No cameras found":
        camera_variable = ctk.StringVar(value="No cameras found")
        camera_optionmenu = ctk.CTkOptionMenu(
            root,
            variable=camera_variable,
            values=["No cameras found"],
            state="disabled",
        )
    else:
        camera_variable = ctk.StringVar(value=camera_names[0])
        camera_optionmenu = ctk.CTkOptionMenu(
            root, variable=camera_variable, values=camera_names
        )

    camera_optionmenu.place(relx=0.35, rely=0.86, relwidth=0.25, relheight=0.05)

    live_button = ctk.CTkButton(
        root,
        text=_("Live"),
        cursor="hand2",
        command=lambda: webcam_preview(
            root,
            (
                camera_indices[camera_names.index(camera_variable.get())]
                if camera_names and camera_names[0] != "No cameras found"
                else None
            ),
        ),
        state=(
            "normal"
            if camera_names and camera_names[0] != "No cameras found"
            else "disabled"
        ),
    )
    live_button.place(relx=0.65, rely=0.86, relwidth=0.2, relheight=0.05)
    # --- End Camera Selection ---
    
    status_label = ctk.CTkLabel(root, text=None, justify="center")
    status_label.place(relx=0.1, rely=0.9, relwidth=0.8)

    donate_label = ctk.CTkLabel(
        root, text="Deep Live Cam", justify="center", cursor="hand2"
    )
    donate_label.place(relx=0.1, rely=0.95, relwidth=0.8)
    donate_label.configure(
        text_color=ctk.ThemeManager.theme.get("URL").get("text_color")
    )
    donate_label.bind(
        "<Button>", lambda event: webbrowser.open("https://deeplivecam.net")
    )

    # Check server connection on startup and update status label
    if api_client.check_server_status():
        status_label.configure(text=_("Connected to backend server."))
    else:
        status_label.configure(text=_("Could not connect to backend server!"))

    return root

def close_mapper_window():
    global POPUP, POPUP_LIVE
    if POPUP and POPUP.winfo_exists():
        POPUP.destroy()
        POPUP = None
    if POPUP_LIVE and POPUP_LIVE.winfo_exists():
        POPUP_LIVE.destroy()
        POPUP_LIVE = None


def analyze_target(start: Callable[[], None]):
    if POPUP != None and POPUP.winfo_exists():
        update_status("Please complete pop-up or close it.")
        return

    if modules.globals.map_faces:
        if not CLIENT_STATE.target_path:
            update_status("Please select a target first.")
            return

        update_status("Requesting face analysis from server...")
        # Call the API client instead of local functions
        source_target_map = api_client.request_face_analysis(CLIENT_STATE.target_path)
        CLIENT_STATE.source_target_map = source_target_map  # Store result in client state for the popup

        if source_target_map:
            update_status(f"Found {len(source_target_map)} unique faces.")
            create_source_target_popup(start, source_target_map)
        else:
            update_status("No faces found in target or server error.")
    else:
        select_output_path(start)


def create_source_target_popup(
    start: Callable[[], None], face_map: list
) -> None:
    global POPUP, popup_status_label

    POPUP = ctk.CTkToplevel(ROOT)
    POPUP.title(_("Source x Target Mapper"))
    POPUP.geometry(f"{POPUP_WIDTH}x{POPUP_HEIGHT}")
    POPUP.focus()

    def on_submit_click(start):
        if has_valid_map():
            POPUP.destroy()
            select_output_path(start)
        else:
            update_pop_status("At least 1 source with target is required!")

    scrollable_frame = ctk.CTkScrollableFrame(
        POPUP, width=POPUP_SCROLL_WIDTH, height=POPUP_SCROLL_HEIGHT
    )
    scrollable_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

    def on_button_click(map, button_num):
        update_popup_source(scrollable_frame, map, button_num)

    for item in face_map:
        id = item["id"]

        button = ctk.CTkButton(
            scrollable_frame,
            text=_("Select source image"),
            command=lambda id=id: on_button_click(face_map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
        )
        button.grid(row=id, column=0, padx=50, pady=10)

        x_label = ctk.CTkLabel(
            scrollable_frame,
            text=f"X",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        x_label.grid(row=id, column=2, padx=10, pady=10)

        image = Image.fromarray(cv2.cvtColor(item["target"]["cv2"], cv2.COLOR_BGR2RGB))
        image = image.resize(
            (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        tk_image = ctk.CTkImage(image, size=image.size)

        target_image = ctk.CTkLabel(
            scrollable_frame,
            text=f"T-{id}",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        target_image.grid(row=id, column=3, padx=10, pady=10)
        target_image.configure(image=tk_image)

    popup_status_label = ctk.CTkLabel(POPUP, text=None, justify="center")
    popup_status_label.grid(row=1, column=0, pady=15)

    close_button = ctk.CTkButton(
        POPUP, text=_("Submit"), command=lambda: on_submit_click(start)
    )
    close_button.grid(row=2, column=0, pady=10)


def update_popup_source(
    scrollable_frame: ctk.CTkScrollableFrame, face_map: list, button_num: int
) -> None:
    global source_label_dict

    source_path = ctk.filedialog.askopenfilename(
        title=_("select a source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "source" in face_map[button_num]:
        face_map[button_num].pop("source")
        if button_num in source_label_dict:
            source_label_dict[button_num].destroy()
            del source_label_dict[button_num]

    if not source_path:
        return

    update_pop_status("Analyzing source face on server...")
    # Use the API client to analyze the selected source image
    api_face_map = api_client.request_face_analysis(source_path)

    if api_face_map and api_face_map[0].get("target"):
        # The server returns a list of faces. We'll use the first one.
        face_data = api_face_map[0]["target"]
        face = face_data["face"]
        cv2_img_face_crop = face_data["cv2"]

        face_map[button_num]["source"] = {
            "cv2": cv2_img_face_crop,
            "face": face,
        }

        try:
            image = Image.fromarray(
                cv2.cvtColor(face_map[button_num]["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=button_num, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)
            source_label_dict[button_num] = source_image
            update_pop_status("Source face set.")
        except Exception as e:
            update_pop_status(f"UI Error: {e}")
    else:
        update_pop_status("Face could not be detected in last upload!")


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title(_("Preview"))
    preview.configure()
    preview.protocol("WM_DELETE_WINDOW", lambda: toggle_preview())
    preview.resizable(width=True, height=True)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill="both", expand=True)

    preview_slider = ctk.CTkSlider(
        preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value)
    )

    return preview


def update_status(text: str) -> None:
    status_label.configure(text=_(text))
    ROOT.update()


def update_pop_status(text: str) -> None:
    popup_status_label.configure(text=_(text))


def update_pop_live_status(text: str) -> None:
    popup_status_label_live.configure(text=_(text))


def update_tumbler(var: str, value: bool) -> None:
    CLIENT_STATE.fp_ui[var] = value
    save_switch_states()
    # If we're currently in a live preview, update the frame processors
    if PREVIEW.state() == "normal" and CLIENT_STATE.webcam_preview_running:
        global frame_processors
        frame_processors = get_frame_processors_modules(
            modules.globals.frame_processors
        )


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE, img_ft, vid_ft

    PREVIEW.withdraw()
    source_path = ctk.filedialog.askopenfilename(
        title=_("select a source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )
    if is_image(source_path): # modules.globals.source_path is now CLIENT_STATE.source_path
        CLIENT_STATE.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(CLIENT_STATE.source_path)
        image = render_image_preview(CLIENT_STATE.source_path, (200, 200))
        source_label.configure(image=image)
    else:
        CLIENT_STATE.source_path = None
        source_label.configure(image=None)


def swap_faces_paths() -> None:
    global RECENT_DIRECTORY_SOURCE, RECENT_DIRECTORY_TARGET

    source_path = CLIENT_STATE.source_path
    target_path = CLIENT_STATE.target_path

    if not is_image(source_path) or not is_image(target_path):
        return

    CLIENT_STATE.source_path = target_path
    CLIENT_STATE.target_path = source_path

    RECENT_DIRECTORY_SOURCE = os.path.dirname(CLIENT_STATE.source_path)
    RECENT_DIRECTORY_TARGET = os.path.dirname(CLIENT_STATE.target_path)

    PREVIEW.withdraw()

    source_image = render_image_preview(CLIENT_STATE.source_path, (200, 200))
    source_label.configure(image=source_image)

    target_image = render_image_preview(CLIENT_STATE.target_path, (200, 200))
    target_label.configure(image=target_image)


def select_target_path() -> None:
    global RECENT_DIRECTORY_TARGET, img_ft, vid_ft

    PREVIEW.withdraw()
    target_path = ctk.filedialog.askopenfilename(
        title=_("select a target image or video"),
        initialdir=RECENT_DIRECTORY_TARGET,
        filetypes=[img_ft, vid_ft],
    )
    if is_image(target_path): # modules.globals.target_path is now CLIENT_STATE.target_path
        CLIENT_STATE.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(CLIENT_STATE.target_path)
        image = render_image_preview(CLIENT_STATE.target_path, (200, 200))
        target_label.configure(image=image)
    elif is_video(target_path): # modules.globals.target_path is now CLIENT_STATE.target_path
        CLIENT_STATE.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(CLIENT_STATE.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame)
    else:
        CLIENT_STATE.target_path = None
        target_label.configure(image=None)


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT, img_ft, vid_ft

    if is_image(CLIENT_STATE.target_path):
        output_path = ctk.filedialog.asksaveasfilename( # modules.globals.target_path is now CLIENT_STATE.target_path
            title=_("save image output file"),
            filetypes=[img_ft],
            defaultextension=".png",
            initialfile="output.png",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    elif is_video(CLIENT_STATE.target_path):
        output_path = ctk.filedialog.asksaveasfilename( # modules.globals.target_path is now CLIENT_STATE.target_path
            title=_("save video output file"),
            filetypes=[vid_ft],
            defaultextension=".mp4",
            initialfile="output.mp4",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    else:
        output_path = None
    if output_path:
        CLIENT_STATE.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(CLIENT_STATE.output_path)
        
        # Gather all relevant options from modules.globals
        options = {
            "frame_processors": CLIENT_STATE.frame_processors, # This comes from CLI, not UI switch
            "keep_fps": CLIENT_STATE.keep_fps,
            "keep_audio": CLIENT_STATE.keep_audio,
            "keep_frames": CLIENT_STATE.keep_frames,
            "many_faces": CLIENT_STATE.many_faces,
            "map_faces": CLIENT_STATE.map_faces,
            "color_correction": CLIENT_STATE.color_correction,
            "nsfw_filter": CLIENT_STATE.nsfw_filter,
            "video_encoder": CLIENT_STATE.video_encoder, # This comes from CLI, not UI switch
            "video_quality": CLIENT_STATE.video_quality, # This comes from CLI, not UI switch
            "mouth_mask": CLIENT_STATE.mouth_mask,
            "show_mouth_mask_box": CLIENT_STATE.show_mouth_mask_box,
            "simple_map": CLIENT_STATE.simple_map, # For mapped faces
            "source_target_map": CLIENT_STATE.source_target_map
        }

        update_status("Sending job to server...")
        response = api_client.initiate_batch_processing(
            CLIENT_STATE.source_path,
            CLIENT_STATE.target_path,
            options
        )

        if response.get("job_id"):
            job_id = response["job_id"] # modules.globals.output_path is now CLIENT_STATE.output_path
            update_status(f"Job {job_id} initiated on server. Downloading result...") 
            if api_client.download_processed_result(job_id, CLIENT_STATE.output_path):
                update_status(f"Job {job_id} completed and result downloaded to {CLIENT_STATE.output_path}")
            else:
                update_status(f"Failed to download result for job {job_id}.")
        else:
            update_status(f"Server failed to initiate job: {response.get('message', 'Unknown error')}")


def check_and_ignore_nsfw(target, destroy: Callable = None) -> bool:
    """Check if the target is NSFW.
    TODO: Consider to make blur the target.
    """
    from numpy import ndarray
    from modules.predicter import predict_image, predict_video, predict_frame

    if type(target) is str:  # image/video file path
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif type(target) is ndarray:  # frame object
        check_nsfw = predict_frame
    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(
                to_quit=False
            )  # Do not need to destroy the window frame if the target is NSFW
        update_status("Processing ignored!")
        return True
    else:
        return False


def fit_image_to_size(image, width: int, height: int):
    if width is None or height is None or width <= 0 or height <= 0:
        return image
    h, w, _ = image.shape
    ratio_h = 0.0
    ratio_w = 0.0
    ratio_w = width / w
    ratio_h = height / h
    # Use the smaller ratio to ensure the image fits within the given dimensions
    ratio = min(ratio_w, ratio_h)
    
    # Compute new dimensions, ensuring they're at least 1 pixel
    new_width = max(1, int(ratio * w))
    new_height = max(1, int(ratio * h))
    new_size = (new_width, new_height)

    return cv2.resize(image, dsize=new_size)


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(
        video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()


def toggle_preview() -> None:
    # This feature is disabled as it relies on local processing.
    # The live preview feature provides a better, server-based alternative.
    if PREVIEW.state() == "normal":
        PREVIEW.withdraw()
    elif CLIENT_STATE.source_path and CLIENT_STATE.target_path:
        # init_preview()
        # update_preview()
        pass


def init_preview() -> None:
    # This feature is disabled.
    if is_image(modules.globals.target_path):
        preview_slider.pack_forget()
    if is_video(modules.globals.target_path):
        video_frame_total = get_video_frame_total(modules.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill="x")
        preview_slider.set(0)


def update_preview(frame_number: int = 0) -> None:
    # This feature is disabled as it relies on local processing.
    if CLIENT_STATE.source_path and CLIENT_STATE.target_path:
        update_status("Processing...")
        temp_frame = get_video_frame(CLIENT_STATE.target_path, frame_number)
        # The following block is disabled because it performs local processing.
        # if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
        #     return
        # for frame_processor in get_frame_processors_modules(
        #         modules.globals.frame_processors
        # ):
        #     # This line requires local insightface and is therefore disabled.
        #     # temp_frame = frame_processor.process_frame(
        #     #     get_one_face(cv2.imread(modules.globals.source_path)), temp_frame
        #     # )
        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(
            image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
        update_status("Processing succeed!")
        PREVIEW.deiconify()


def webcam_preview(root: ctk.CTk, camera_index: int) -> None:
    global POPUP_LIVE

    if POPUP_LIVE and POPUP_LIVE.winfo_exists():
        update_status("Source x Target Mapper is already open.")
        POPUP_LIVE.focus()
        return
    
    if CLIENT_STATE.map_faces:
        # For multi-face, we must first open the mapper UI to define the mappings.
        # If the map is empty, add a default blank entry to start with.
        if not CLIENT_STATE.source_target_map:
            add_blank_map(CLIENT_STATE.source_target_map) # Pass the list to modify
        create_source_target_popup_for_webcam(root, CLIENT_STATE.source_target_map, camera_index)
    else:
        # For single-face mode, ensure a source is selected and set it on the server.
        if CLIENT_STATE.source_path is None:
            update_status("Please select a source image first.")
            return

        update_status("Setting source face on server...") # modules.globals.source_path is now CLIENT_STATE.source_path
        if not api_client.set_live_source(CLIENT_STATE.source_path):
            update_status("Could not set source face on server. Check server logs.")
            return

        create_networked_webcam_preview(camera_index)



def get_available_cameras():
    """Returns a list of available camera names and indices."""
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()

            # Create list of indices and names
            camera_indices = list(range(len(devices)))
            camera_names = devices

            # If no cameras found through DirectShow, try OpenCV fallback
            if not camera_names:
                # Try to open camera with index -1 and 0
                test_indices = [-1, 0]
                working_cameras = []

                for idx in test_indices:
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        working_cameras.append(f"Camera {idx}")
                        cap.release()

                if working_cameras:
                    return test_indices[: len(working_cameras)], working_cameras

            # If still no cameras found, return empty lists
            if not camera_names:
                return [], ["No cameras found"]

            return camera_indices, camera_names

        except Exception as e:
            print(f"Error detecting cameras: {str(e)}")
            return [], ["No cameras found"]
    else:
        # Unix-like systems (Linux/Mac) camera detection
        camera_indices = []
        camera_names = []

        if platform.system() == "Darwin":  # macOS specific handling
            # Try to open the default FaceTime camera first
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                camera_indices.append(0)
                camera_names.append("FaceTime Camera")
                cap.release()

            # On macOS, additional cameras typically use indices 1 and 2
            for i in [1, 2]:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    camera_indices.append(i)
                    camera_names.append(f"Camera {i}")
                    cap.release()
        else:
            # Linux camera detection - test first 10 indices
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    camera_indices.append(i)
                    camera_names.append(f"Camera {i}")
                    cap.release()

        if not camera_names:
            return [], ["No cameras found"]

        return camera_indices, camera_names


def create_networked_webcam_preview(camera_index: int) -> None:
    """
    Starts a high-performance, asynchronous live preview using WebSockets.

    This function decouples sending and receiving frames into separate threads
    to avoid blocking the UI and to maximize throughput.
    """
    global PREVIEW, preview_label, ROOT
    ws = None
    cap = None
    stop_event = threading.Event() # Event to signal stopping the threads
    frame_queue = queue.Queue(maxsize=2)  # Use a queue to pass frames from receiver to main thread
    can_send_frame = threading.Event() # Event to control frame sending rate
    can_send_frame.set() # Initially, we are allowed to send a frame
    CLIENT_STATE.webcam_preview_running = True # Set global state for UI

    def receiver_thread(ws_socket: websocket.WebSocket, q: queue.Queue, stop: threading.Event, can_send: threading.Event) -> None:
        """Listens for incoming messages from the server and puts them in a queue."""
        while not stop.is_set():
            try:
                result_b64 = ws_socket.recv()
                if not result_b64:
                    break
                img_data = base64.b64decode(result_b64)
                np_arr = np.frombuffer(img_data, np.uint8)
                processed_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if q.full():
                    q.get_nowait()  # Discard the oldest frame if the queue is full
                q.put(processed_frame)
                can_send.set() # Signal that we've received a response and can send the next frame
            except (websocket.WebSocketConnectionClosedException, ConnectionResetError):
                update_status("Connection to server lost.")
                break
            except Exception as e:
                print(f"Receiver thread error: {e}")
                break
        stop.set() # Signal the main loop to stop

    try:
        ws_url = f"ws://{api_client.get_server_url().split('//')[1]}/ws/live-preview"
        update_status(f"Connecting to {ws_url}...")
        ws = websocket.create_connection(ws_url, timeout=30)
        update_status("Connected to live preview server.")
        
        cap = VideoCapturer(camera_index)
        if not cap.start(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT, 30):
            raise RuntimeError("Failed to start camera")

        # Start the receiver thread
        receiver = threading.Thread(target=receiver_thread, args=(ws, frame_queue, stop_event, can_send_frame))
        receiver.start()

        preview_label.configure(width=PREVIEW_DEFAULT_WIDTH, height=PREVIEW_DEFAULT_HEIGHT)
        PREVIEW.deiconify()

        # Main loop for sending frames and updating the UI
        prev_frame_time = 0
        while not stop_event.is_set():
            # Always capture the latest frame from the camera to reduce latency
            ret, current_frame = cap.read()
            if not ret:
                break

            # --- Sending Part (with rate-limiting) ---
            if can_send_frame.is_set():
                can_send_frame.clear() # Reset the event until the next response is received

                frame_to_send = current_frame
                if CLIENT_STATE.live_mirror:
                    frame_to_send = cv2.flip(frame_to_send, 1)

                _, buffer = cv2.imencode('.jpg', frame_to_send, [int(cv2.IMWRITE_JPEG_QUALITY), 65]) # Reduced quality for faster transfer
                frame_b64 = base64.b64encode(buffer).decode('utf-8')

                options = {
                    "frame_processors": CLIENT_STATE.frame_processors,
                    "many_faces": CLIENT_STATE.many_faces,
                    "map_faces": CLIENT_STATE.map_faces,
                    "color_correction": CLIENT_STATE.color_correction,
                    "mouth_mask": CLIENT_STATE.mouth_mask,
                    "show_mouth_mask_box": CLIENT_STATE.show_mouth_mask_box,
                    "fp_ui": CLIENT_STATE.fp_ui
                }

                if CLIENT_STATE.map_faces:
                    # For live preview, simple_map is used for multi-face mapping
                    options['simple_map'] = CLIENT_STATE.simple_map
                    options['source_target_map'] = CLIENT_STATE.source_target_map

                payload_data = {'frame': frame_b64, 'options': options}
                
                try:
                    ws.send(json.dumps(payload_data))
                except (websocket.WebSocketConnectionClosedException, ConnectionResetError):
                    stop_event.set() # Stop the loop if connection is lost
                    continue

            # --- Receiving/Display Part (non-blocking) ---
            try:
                processed_frame = frame_queue.get_nowait()
                new_frame_time = time.time()
                if prev_frame_time > 0:
                    fps = 1 / (new_frame_time - prev_frame_time) # modules.globals.show_fps is now CLIENT_STATE.show_fps
                    if CLIENT_STATE.show_fps:
                        cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                prev_frame_time = new_frame_time
                image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ctk.CTkImage(image, size=(processed_frame.shape[1], processed_frame.shape[0]))
                preview_label.configure(image=image)
            except queue.Empty:
                pass  # No new frame from the server yet

            ROOT.update()
            if PREVIEW.state() == "withdrawn":
                break

    except Exception as e:
        update_status(f"Live preview error: {e}")
    finally:
        stop_event.set()
        if ws:
            ws.close()
        if cap:
            cap.release()
        CLIENT_STATE.webcam_preview_running = False # Reset global state
        PREVIEW.withdraw()
        update_status("Live preview stopped.")

def create_source_target_popup_for_webcam(
        root: ctk.CTk, map: list, camera_index: int
) -> None:
    global POPUP_LIVE, popup_status_label_live

    POPUP_LIVE = ctk.CTkToplevel(root)
    POPUP_LIVE.title(_("Source x Target Mapper"))
    POPUP_LIVE.geometry(f"{POPUP_LIVE_WIDTH}x{POPUP_LIVE_HEIGHT}")
    POPUP_LIVE.focus()

    def on_submit_click():
        if has_valid_map(CLIENT_STATE.source_target_map): # Pass the list to check
            simplify_maps(CLIENT_STATE.source_target_map, CLIENT_STATE.simple_map) # Pass lists to modify
            update_pop_live_status("Mappings successfully submitted!")
            create_networked_webcam_preview(camera_index)  # Open the preview window
        else:
            update_pop_live_status("At least 1 source with target is required!")

    def on_add_click():
        add_blank_map(CLIENT_STATE.source_target_map) # Pass the list to modify
        refresh_data(CLIENT_STATE.source_target_map)
        update_pop_live_status("Please provide mapping!")

    def on_clear_click():
        clear_source_target_images(map)
        refresh_data(map)
        update_pop_live_status("All mappings cleared!")

    popup_status_label_live = ctk.CTkLabel(POPUP_LIVE, text=None, justify="center")
    popup_status_label_live.grid(row=1, column=0, pady=15)

    add_button = ctk.CTkButton(POPUP_LIVE, text=_("Add"), command=lambda: on_add_click())
    add_button.place(relx=0.1, rely=0.92, relwidth=0.2, relheight=0.05)

    clear_button = ctk.CTkButton(POPUP_LIVE, text=_("Clear"), command=lambda: on_clear_click())
    clear_button.place(relx=0.4, rely=0.92, relwidth=0.2, relheight=0.05)

    close_button = ctk.CTkButton(
        POPUP_LIVE, text=_("Submit"), command=lambda: on_submit_click()
    )
    close_button.place(relx=0.7, rely=0.92, relwidth=0.2, relheight=0.05)



def clear_source_target_images(map: list):
    global source_label_dict_live, target_label_dict_live

    for item in map:
        if "source" in item:
            del item["source"]
        if "target" in item:
            del item["target"]

    for button_num in list(source_label_dict_live.keys()):
        source_label_dict_live[button_num].destroy()
        del source_label_dict_live[button_num]

    for button_num in list(target_label_dict_live.keys()):
        target_label_dict_live[button_num].destroy()
        del target_label_dict_live[button_num]


def refresh_data(map: list):
    global POPUP_LIVE

    scrollable_frame = ctk.CTkScrollableFrame(
        POPUP_LIVE, width=POPUP_LIVE_SCROLL_WIDTH, height=POPUP_LIVE_SCROLL_HEIGHT
    )
    scrollable_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

    def on_sbutton_click(map, button_num):
        map = update_webcam_source(scrollable_frame, map, button_num)

    def on_tbutton_click(map, button_num):
        map = update_webcam_target(scrollable_frame, map, button_num)

    for item in map:
        id = item["id"]

        button = ctk.CTkButton(
            scrollable_frame,
            text=_("Select source image"),
            command=lambda id=id: on_sbutton_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
        )
        button.grid(row=id, column=0, padx=30, pady=10)

        x_label = ctk.CTkLabel(
            scrollable_frame,
            text=f"X",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        x_label.grid(row=id, column=2, padx=10, pady=10)

        button = ctk.CTkButton(
            scrollable_frame,
            text=_("Select target image"),
            command=lambda id=id: on_tbutton_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
        )
        button.grid(row=id, column=3, padx=20, pady=10)

        if "source" in item:
            image = Image.fromarray(
                cv2.cvtColor(item["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{id}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=id, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)

        if "target" in item:
            image = Image.fromarray(
                cv2.cvtColor(item["target"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            target_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"T-{id}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            target_image.grid(row=id, column=4, padx=20, pady=10)
            target_image.configure(image=tk_image)


def update_webcam_source(
    scrollable_frame: ctk.CTkScrollableFrame, face_map: list, button_num: int # face_map is CLIENT_STATE.source_target_map
) -> None:
    global source_label_dict_live

    source_path = ctk.filedialog.askopenfilename(
        title=_("select a source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "source" in face_map[button_num]:
        face_map[button_num].pop("source")
        if button_num in source_label_dict_live:
            source_label_dict_live[button_num].destroy()
            del source_label_dict_live[button_num]

    if not source_path:
        return

    update_pop_live_status("Analyzing source face on server...")
    api_face_map = api_client.request_face_analysis(source_path)

    if api_face_map and api_face_map[0].get("target"):
        face_data = api_face_map[0]["target"]
        face = face_data["face"]
        cv2_img_face_crop = face_data["cv2"]

        face_map[button_num]["source"] = {
            "cv2": cv2_img_face_crop,
            "face": face,
        }

        try:
            image = Image.fromarray(
                cv2.cvtColor(face_map[button_num]["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=button_num, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)
            source_label_dict_live[button_num] = source_image
            update_pop_live_status("Source face set.")
        except Exception as e:
            update_pop_live_status(f"UI Error: {e}")
    else:
        update_pop_live_status("Face could not be detected in last upload!")


def update_webcam_target(
    scrollable_frame: ctk.CTkScrollableFrame, face_map: list, button_num: int # face_map is CLIENT_STATE.source_target_map
) -> None:
    global target_label_dict_live

    target_path = ctk.filedialog.askopenfilename(
        title=_("select a target image"),
        initialdir=RECENT_DIRECTORY_TARGET,
        filetypes=[img_ft],
    )

    if "target" in face_map[button_num]:
        face_map[button_num].pop("target")
        if button_num in target_label_dict_live:
            target_label_dict_live[button_num].destroy()
            del target_label_dict_live[button_num]

    if not target_path:
        return

    update_pop_live_status("Analyzing target face on server...")
    api_face_map = api_client.request_face_analysis(target_path)

    if api_face_map and api_face_map[0].get("target"):
        face_data = api_face_map[0]["target"]
        face = face_data["face"]
        cv2_img_face_crop = face_data["cv2"]

        face_map[button_num]["target"] = {
            "cv2": cv2_img_face_crop,
            "face": face,
        }

        try:
            image = Image.fromarray(
                cv2.cvtColor(face_map[button_num]["target"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            target_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"T-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            target_image.grid(row=button_num, column=4, padx=20, pady=10)
            target_image.configure(image=tk_image)
            target_label_dict_live[button_num] = target_image
            update_pop_live_status("Target face set.")
        except Exception as e:
            update_pop_live_status(f"UI Error: {e}")
    else:
        update_pop_live_status("Face could not be detected in last upload!")
