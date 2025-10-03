# md5_modify_mp4.py
#
# A tool for applying subtle perturbations to specific frames of MP4 video files.
# This can be used to alter the MD5 hash of a video with minimal visual impact.
# Features include noise injection and custom image overlays with transition effects.
# The application is built with Tkinter for a graphical user interface.
# It is anonymous and free to be used and modified.

import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
cv2 = None # Lazy import cv2
np = None  # Lazy import numpy
import threading
import json
import time
from PIL import Image, ImageTk

CONFIG_FILE = "config.json"

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def import_heavy_libraries():
    """Import computer vision libraries only when needed to speed up initial launch."""
    global cv2, np
    if cv2 is None:
        import cv2
    if np is None:
        import numpy as np


class PositionAndScaleWindow(tk.Toplevel):
    """
    An interactive window for setting the watermark's position and scale.
    This window is constrained to a maximum size to handle high-resolution videos,
    and it transparently converts coordinates between the preview and original video scales.
    """
    def __init__(self, parent, video_path, watermark_image, target_frames, initial_settings):
        super().__init__(parent)
        self.title("Set Position & Scale for Target Frames")
        self.iconbitmap(resource_path("mmmdog.ico"))
        self.transient(parent)
        self.grab_set()

        # --- NEW: Define maximum preview dimensions ---
        MAX_PREVIEW_WIDTH = 960
        MAX_PREVIEW_HEIGHT = 720

        self.video_path = video_path
        self.watermark_pil = watermark_image
        self.target_frames = target_frames
        self.settings = initial_settings
        self.final_settings = self.settings.copy()
        self.current_frame_index = self.target_frames[0]

        self.video_frames = {}
        self._load_video_frames()

        first_frame_pil = self.video_frames.get(self.current_frame_index)
        if not first_frame_pil:
            messagebox.showerror("Error", "Could not load any video frames.", parent=self)
            self.destroy()
            return
            
        # --- NEW: Calculate scaling factor to fit preview on screen ---
        self.original_width, self.original_height = first_frame_pil.size
        self.scale_factor = min(MAX_PREVIEW_WIDTH / self.original_width, MAX_PREVIEW_HEIGHT / self.original_height)

        # Ensure scale factor is not > 1 (don't enlarge small videos)
        if self.scale_factor > 1:
            self.scale_factor = 1

        self.preview_width = int(self.original_width * self.scale_factor)
        self.preview_height = int(self.original_height * self.scale_factor)
        
        # --- UI Layout (Uses new preview dimensions) ---
        top_frame = ttk.Frame(self, padding=5)
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="Select Target Frame:").pack(side=tk.LEFT, padx=(0, 5))
        self.frame_selector = ttk.Combobox(top_frame, values=self.target_frames, state="readonly")
        self.frame_selector.pack(side=tk.LEFT)
        self.frame_selector.set(self.current_frame_index)
        self.frame_selector.bind("<<ComboboxSelected>>", self.on_frame_select)

        # --- MODIFIED: Create canvas with scaled dimensions ---
        self.canvas = tk.Canvas(self, width=self.preview_width, height=self.preview_height)
        self.canvas.pack()
        
        # Display the first frame (resized)
        resized_bg = first_frame_pil.resize((self.preview_width, self.preview_height), Image.Resampling.LANCZOS)
        self.bg_image_tk = ImageTk.PhotoImage(resized_bg)
        self.bg_canvas_item = self.canvas.create_image(0, 0, anchor="nw", image=self.bg_image_tk)
        
        self.watermark_id = None
        self.watermark_tk = None
        
        scale_frame = ttk.Frame(self, padding=5)
        scale_frame.pack(fill=tk.X)
        self.scale_var = tk.IntVar(value=100)
        
        ttk.Label(scale_frame, text="Scale (%):").pack(side=tk.LEFT)
        self.scale_slider = ttk.Scale(scale_frame, from_=10, to=500, variable=self.scale_var, command=self._update_scale_from_slider)
        self.scale_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.scale_entry = ttk.Entry(scale_frame, textvariable=self.scale_var, width=5)
        self.scale_entry.pack(side=tk.LEFT)
        self.scale_entry.bind("<Return>", lambda e: self.update_watermark_preview())
        self.scale_var.trace_add("write", self._update_watermark_from_var)

        ttk.Button(self, text="Confirm Settings for All Frames", command=self.on_confirm).pack(pady=10)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.load_state_for_current_frame()

    def _load_video_frames(self):
        """Extracts and stores the specified target frames from the video at FULL resolution."""
        # This method is correct as is. We want to load the original frames.
        # The resizing will happen only for display purposes.
        cap = None
        try:
            import_heavy_libraries()
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened(): raise IOError("Cannot open video file")
            for frame_idx in self.target_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    self.video_frames[frame_idx] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    if self.video_frames:
                        w, h = next(iter(self.video_frames.values())).size
                        self.video_frames[frame_idx] = Image.new('RGB', (w, h))
            if not self.video_frames: raise ValueError("Could not read any target frames from the video.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video frames: {e}", parent=self)
            self.destroy()
        finally:
            if cap: cap.release()

    def on_frame_select(self, event=None):
        """Handles switching between target frames in the combobox."""
        self.save_state_for_current_frame()
        self.current_frame_index = int(self.frame_selector.get())
        
        # --- MODIFIED: Resize the background image for preview ---
        bg_pil = self.video_frames[self.current_frame_index]
        bg_pil_resized = bg_pil.resize((self.preview_width, self.preview_height), Image.Resampling.LANCZOS)
        self.bg_image_tk = ImageTk.PhotoImage(bg_pil_resized)
        self.canvas.itemconfig(self.bg_canvas_item, image=self.bg_image_tk)
        
        if self.watermark_id: self.canvas.delete(self.watermark_id); self.watermark_id = None
        self.load_state_for_current_frame()
        
    def save_state_for_current_frame(self):
        """Saves the current position and scale, converting preview coordinates to original video coordinates."""
        if not self.watermark_id: return
        
        # --- MODIFIED: Coordinate conversion ---
        # Get coordinates from the smaller preview canvas
        preview_coords = self.canvas.coords(self.watermark_id)
        
        # Scale the coordinates up to match the original video resolution
        original_pos = (
            int(preview_coords[0] / self.scale_factor),
            int(preview_coords[1] / self.scale_factor)
        )

        # Always use string keys for JSON compatibility
        self.settings[str(self.current_frame_index)] = { "pos": original_pos, "scale": self.scale_var.get() }

    def load_state_for_current_frame(self):
        """Loads the stored position and scale, converting original coordinates to preview coordinates."""
        # Always use string keys to read settings
        frame_settings = self.settings.get(str(self.current_frame_index), {"pos": (0, 0), "scale": 100})
        
        # --- MODIFIED: Coordinate conversion ---
        # Get the saved, original-scale position
        original_pos = frame_settings["pos"]
        scale = frame_settings["scale"]

        # Scale the coordinates down to fit the preview canvas
        preview_pos = (
            int(original_pos[0] * self.scale_factor),
            int(original_pos[1] * self.scale_factor)
        )
        
        self.scale_var.set(scale)
        # Use the converted preview position to place the watermark
        self.update_watermark_preview(position=preview_pos)

    # --- NO CHANGES REQUIRED FOR THE METHODS BELOW ---
    # These methods work with the preview canvas coordinates, which is correct.
    # The conversion logic is handled entirely by save_state and load_state.

    def update_watermark_preview(self, position=None):
        """Redraws the watermark on the canvas based on current scale and position."""
        scale_percent = self.scale_var.get()
        if scale_percent <= 0: return

        # The watermark's scale is relative to its own size, not the video size.
        # But we must also account for the preview's down-scaling.
        zoom = (scale_percent / 100.0) * self.scale_factor
        w, h = self.watermark_pil.size
        scaled_w, scaled_h = int(w * zoom), int(h * zoom)
        if scaled_w < 1 or scaled_h < 1: return # Prevent zero-size images
        
        scaled_watermark = self.watermark_pil.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
        self.watermark_tk = ImageTk.PhotoImage(scaled_watermark)
        
        if self.watermark_id:
            self.canvas.itemconfig(self.watermark_id, image=self.watermark_tk)
            if position: self.canvas.coords(self.watermark_id, position)
        else:
            initial_pos = position if position else (0, 0)
            self.watermark_id = self.canvas.create_image(initial_pos[0], initial_pos[1], anchor="nw", image=self.watermark_tk)
            self.canvas.tag_bind(self.watermark_id, "<ButtonPress-1>", self.on_press)
            self.canvas.tag_bind(self.watermark_id, "<B1-Motion>", self.on_drag)

    def on_press(self, event):
        """Records the starting position for a drag operation."""
        self._drag_start_x, self._drag_start_y = event.x, event.y

    def on_drag(self, event):
        """Moves the watermark on the canvas."""
        dx, dy = event.x - self._drag_start_x, event.y - self._drag_start_y
        self.canvas.move(self.watermark_id, dx, dy)
        self._drag_start_x, self._drag_start_y = event.x, event.y

    def on_mouse_wheel(self, event):
        """Adjusts the scale using the mouse wheel."""
        delta = 5 if event.delta > 0 else -5
        new_scale = self.scale_var.get() + delta
        if 10 <= new_scale <= 500:
            self.scale_var.set(new_scale)

    def _update_scale_from_slider(self, value):
        self.scale_var.set(int(float(value)))

    def _update_watermark_from_var(self, *args):
        self.update_watermark_preview(self.canvas.coords(self.watermark_id) if self.watermark_id else (0,0))

    def on_confirm(self):
        """Finalizes settings and closes the window."""
        self.save_state_for_current_frame()
        self.final_settings = self.settings.copy()
        self.destroy()


class MultiPreviewWindow(tk.Toplevel):
    """A window to display thumbnail previews of the modification on multiple videos."""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Batch Effect Preview")
        self.geometry("850x650")
        self.iconbitmap(resource_path("mmmdog.ico"))
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
    def add_image(self, filename, image_pil):
        """Adds a new video preview to the scrollable list."""
        frame = tk.Frame(self.scrollable_frame, bd=2, relief=tk.GROOVE)
        frame.pack(pady=5, padx=5, fill="x")
        tk.Label(frame, text=filename, fg="blue").pack()
        img_tk = ImageTk.PhotoImage(image_pil)
        label = tk.Label(frame, image=img_tk)
        label.image = img_tk  # Keep a reference to avoid garbage collection
        label.pack()

def parse_frame_string(frame_str, fps):
    """Converts a user-defined string of frames/timestamps into a sorted list of frame indices."""
    target_indices = set()
    parts = frame_str.replace(" ", "").split(',')
    for part in parts:
        if not part: continue
        try:
            if part.lower().endswith('s'):
                target_indices.add(int(float(part[:-1]) * fps))
            else:
                frame_num = int(part)
                if frame_num > 0:
                    target_indices.add(frame_num - 1) # Convert 1-based to 0-based index
        except ValueError:
            print(f"Could not parse: '{part}'")
    return sorted(list(target_indices))

def apply_perturbation(frame, intensity_level, settings, filename, frame_index, center_frame_for_pos):
    """Applies the selected perturbation effect to a single video frame."""
    mode = settings["perturbation_mode"]
    alpha_intensity = intensity_level / 100.0
    beta = 1.0 - alpha_intensity

    if mode in ["Color Noise", "Grayscale Noise"]:
        height, width, _ = frame.shape
        if mode == "Color Noise":
            noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        else: # Grayscale Noise
            noise_gray = np.random.randint(0, 256, (height, width), dtype=np.uint8)
            noise = cv2.cvtColor(noise_gray, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(frame, beta, noise, alpha_intensity, 0.0)

    elif mode == "Custom Image" and settings.get("custom_image_data"):
        base_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        watermark_pil = settings["custom_image_data"]
        if watermark_pil.mode != 'RGBA':
            watermark_pil = watermark_pil.convert('RGBA')
        
        # Apply the intensity as transparency to the watermark
        temp_watermark = watermark_pil.copy()
        alpha_channel = temp_watermark.split()[3]
        new_alpha = Image.eval(alpha_channel, lambda a: int(a * alpha_intensity))
        temp_watermark.putalpha(new_alpha)

        if settings["custom_image_mode"] == "Tile":
            zoom = settings["custom_image_zoom"] / 100.0
            w, h = temp_watermark.size
            scaled_w, scaled_h = int(w * zoom), int(h * zoom)
            if scaled_w > 0 and scaled_h > 0:
                scaled_watermark = temp_watermark.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
                for i in range(0, base_pil.width, scaled_w):
                    for j in range(0, base_pil.height, scaled_h):
                        base_pil.paste(scaled_watermark, (i, j), scaled_watermark)
        elif settings["custom_image_mode"] == "Local Overlay":
            file_settings = settings["watermark_positions"].get(filename, {})
            # V10 Bug Fix: Use center_frame_for_pos to look up settings, ensuring
            # transition frames use the correct position of their target frame.
            frame_settings = file_settings.get(str(center_frame_for_pos), {"pos": (0, 0), "scale": 100})
            position, scale = tuple(frame_settings.get("pos", (0, 0))), frame_settings.get("scale", 100)

            zoom = scale / 100.0
            w, h = temp_watermark.size
            scaled_w, scaled_h = int(w * zoom), int(h * zoom)
            if scaled_w > 0 and scaled_h > 0:
                final_watermark = temp_watermark.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
                base_pil.paste(final_watermark, position, final_watermark)

        return cv2.cvtColor(np.array(base_pil), cv2.COLOR_RGB2BGR)
        
    return frame

def modify_videos_worker(root, file_paths, status_callback, progress_callback, settings):
    """The core video processing function, designed to run in a separate thread."""
    import_heavy_libraries()
    processed_count = 0
    total_files = len(file_paths)
    try:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        noise_level, suffix, frame_str = settings["noise_level"], settings["suffix"], settings["frames"]
        transition_mode, transition_window = settings["transition_mode"], settings["transition_window"]
        
        for i, file_path in enumerate(file_paths):
            base_name = os.path.basename(file_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(output_dir, f"{name}{suffix}{ext}")
            status_callback(f"Processing: {base_name} ({i+1}/{total_files})")
            
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened(): raise IOError(f"Cannot open: {base_name}")
            width, height = int(cap.get(3)), int(cap.get(4))
            fps = cap.get(5)
            if fps == 0: raise ValueError(f"Cannot read FPS from: {base_name}")

            # V10 Bug Fix: The effects dictionary now stores both intensity and the
            # associated center frame. This is crucial for local overlays with transitions.
            frame_effects = {}
            center_frames = parse_frame_string(frame_str, fps)
            if transition_mode == "Fade In/Out":
                for center_frame in center_frames:
                    for j in range(-transition_window, transition_window + 1):
                        intensity = noise_level * (1 - abs(j) / (transition_window + 1))
                        frame_index = center_frame + j
                        if frame_index >= 0:
                            # If multiple effects overlap, use the one with the highest intensity
                            existing_intensity = frame_effects.get(frame_index, {}).get("intensity", -1)
                            if intensity > existing_intensity:
                                frame_effects[frame_index] = {"intensity": intensity, "center_frame": center_frame}
            else: # "Target Frames Only" mode
                for frame_index in center_frames:
                    frame_effects[frame_index] = {"intensity": noise_level, "center_frame": frame_index}
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            current_frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                if current_frame_index in frame_effects:
                    effect_data = frame_effects[current_frame_index]
                    out.write(apply_perturbation(frame, effect_data["intensity"], settings, base_name, current_frame_index, effect_data["center_frame"]))
                else:
                    out.write(frame)
                current_frame_index += 1
                
            cap.release()
            out.release()
            processed_count += 1
            progress_callback(processed_count)
            
        root.after(0, lambda: messagebox.showinfo("Complete", f"Task Complete!\n\nSuccessfully processed {processed_count}/{total_files} files.\n\nFiles saved to the '{output_dir}' folder."))
    except Exception as e:
        root.after(0, lambda: messagebox.showerror("Processing Failed", str(e)))
    finally:
        progress_callback(-1) # Signal completion

class VideoModifierApp:
    """The main application class for the Video Perturbation Tool."""
    def __init__(self, root):
        self.root = root
        self.root.title("MMMDog")
        self.root.geometry("800x650")
        self.root.iconbitmap(resource_path("mmmdog.ico"))
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.file_paths = []
        self.custom_image_data = None
        self.watermark_positions = {}
        self.processing_active = False
        self.status_animation_counter = 0

        self._setup_ui()
        self.load_settings()
        self.toggle_controls()

    def _setup_ui(self):
        """Initializes all the graphical user interface elements."""
        # --- UI Frames ---
        top_frame = tk.Frame(self.root, padx=10, pady=10)
        top_frame.pack(fill=tk.X)
        list_frame = tk.Frame(self.root, padx=10, pady=5)
        list_frame.pack(fill=tk.BOTH, expand=True)
        settings_frame = tk.Frame(self.root, padx=10, pady=10)
        settings_frame.pack(fill=tk.X)
        action_frame = tk.Frame(self.root, padx=10, pady=10)
        action_frame.pack(fill=tk.X)
        status_frame = tk.Frame(self.root, padx=10, pady=5)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # --- Top Frame: File Operations ---
        self.select_button = tk.Button(top_frame, text="1. Select Videos", command=self.select_files)
        self.select_button.pack(side=tk.LEFT, padx=5)
        self.preview_set_button = tk.Button(top_frame, text="Preview / Set Positions", command=self.run_preview_or_set, state=tk.DISABLED)
        self.preview_set_button.pack(side=tk.LEFT, padx=5)
        self.delete_button = tk.Button(top_frame, text="Delete Selected", command=self.delete_selected, state=tk.DISABLED)
        self.delete_button.pack(side=tk.LEFT, padx=5)
        self.clear_button = tk.Button(top_frame, text="Clear List", command=self.clear_list, state=tk.DISABLED)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # --- List Frame: File Listbox ---
        self.file_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        # --- Settings Frame ---
        # 2. Transition Effect
        transition_frame = tk.LabelFrame(settings_frame, text="2. Transition Effect", padx=5, pady=5)
        transition_frame.pack(fill=tk.X, expand=True)
        self.transition_mode_var = tk.StringVar(value="Fade In/Out")
        ttk.Radiobutton(transition_frame, text="Fade In/Out", variable=self.transition_mode_var, value="Fade In/Out", command=self.toggle_controls).pack(side=tk.LEFT)
        ttk.Radiobutton(transition_frame, text="Target Frames Only", variable=self.transition_mode_var, value="Target Frames Only", command=self.toggle_controls).pack(side=tk.LEFT, padx=10)
        self.transition_window_label = tk.Label(transition_frame, text="Transition Frames:")
        self.transition_window_label.pack(side=tk.LEFT, padx=(10, 0))
        self.transition_window_var = tk.IntVar(value=14)
        self.transition_window_entry = tk.Entry(transition_frame, textvariable=self.transition_window_var, width=5)
        self.transition_window_entry.pack(side=tk.LEFT, padx=5)

        # 3. Perturbation Mode
        perturb_frame = tk.LabelFrame(settings_frame, text="3. Perturbation Mode", padx=5, pady=5)
        perturb_frame.pack(fill=tk.X, expand=True, pady=5)
        self.perturbation_mode_var = tk.StringVar(value="Color Noise")
        for mode in ["Color Noise", "Grayscale Noise", "Custom Image"]:
            ttk.Radiobutton(perturb_frame, text=mode, variable=self.perturbation_mode_var, value=mode, command=self.toggle_controls).pack(side=tk.LEFT, padx=5)
        self.noise_label = tk.Label(perturb_frame, text="Intensity (%):")
        self.noise_label.pack(side=tk.LEFT, padx=(20,5))
        self.noise_scale_var = tk.IntVar(value=50)
        self.noise_scale = ttk.Scale(perturb_frame, from_=1, to=100, variable=self.noise_scale_var, command=lambda v: self.noise_scale_var.set(int(float(v))))
        self.noise_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.noise_entry = tk.Entry(perturb_frame, width=5, textvariable=self.noise_scale_var)
        self.noise_entry.pack(side=tk.LEFT, padx=5)
        self.noise_entry.bind("<Return>", lambda e: self.noise_scale.set(self.noise_scale_var.get()))

        # Custom Image Settings
        self.custom_image_frame = tk.LabelFrame(settings_frame, text="Custom Image Details", padx=5, pady=5)
        self.custom_image_frame.pack(fill=tk.X, expand=True)
        self.import_button = tk.Button(self.custom_image_frame, text="Import Image...", command=self.import_image)
        self.import_button.pack(side=tk.LEFT)
        self.image_path_label = tk.Label(self.custom_image_frame, text="No image selected", relief=tk.SUNKEN, width=25)
        self.image_path_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.custom_image_mode_var = tk.StringVar(value="Tile")
        ttk.Radiobutton(self.custom_image_frame, text="Tile", variable=self.custom_image_mode_var, value="Tile", command=self.toggle_controls).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(self.custom_image_frame, text="Local Overlay", variable=self.custom_image_mode_var, value="Local Overlay", command=self.toggle_controls).pack(side=tk.LEFT)
        self.zoom_label = tk.Label(self.custom_image_frame, text="Scale (%):")
        self.zoom_label.pack(side=tk.LEFT, padx=(10,0))
        self.zoom_var = tk.IntVar(value=100)
        self.zoom_scale = ttk.Scale(self.custom_image_frame, from_=10, to=500, variable=self.zoom_var, command=lambda v: self.zoom_var.set(int(float(v))))
        self.zoom_scale.pack(side=tk.LEFT)
        self.zoom_entry = tk.Entry(self.custom_image_frame, width=5, textvariable=self.zoom_var)
        self.zoom_entry.pack(side=tk.LEFT, padx=5)
        self.zoom_entry.bind("<Return>", lambda e: self.zoom_scale.set(self.zoom_var.get()))

        # 4. Output and Target Frames
        output_frame = tk.LabelFrame(settings_frame, text="4. Output and Target Frames", padx=5, pady=5)
        output_frame.pack(fill=tk.X, expand=True, pady=5)
        tk.Label(output_frame, text="Filename Suffix:").pack(side=tk.LEFT)
        self.suffix_var = tk.StringVar(value="-m")
        self.suffix_entry = tk.Entry(output_frame, textvariable=self.suffix_var, width=10)
        self.suffix_entry.pack(side=tk.LEFT, padx=(0,20))
        tk.Label(output_frame, text="Target Frames/Times (e.g., 1, 3, 4, 3.0s):").pack(side=tk.LEFT)
        self.frames_var = tk.StringVar(value="1,3,4,3.0s")
        self.frames_entry = tk.Entry(output_frame, textvariable=self.frames_var)
        self.frames_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- Action & Status ---
        self.modify_button = tk.Button(action_frame, text="5. Start Processing", bg="lightblue", font=("Helvetica", 12, "bold"), command=self.start_modification_thread, state=tk.DISABLED)
        self.modify_button.pack(fill=tk.X)
        self.status_label = tk.Label(status_frame, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.timer_label = tk.Label(status_frame, text="Time: 00:00", bd=1, relief=tk.SUNKEN, anchor=tk.E)
        self.timer_label.pack(side=tk.RIGHT)

    def start_modification_thread(self):
        """Validates settings and starts the video processing in a new thread."""
        if not self.file_paths:
            messagebox.showwarning("Warning", "Please select video files first!")
            return
        if self.perturbation_mode_var.get() == "Custom Image" and not self.custom_image_data:
            messagebox.showwarning("Warning", "Please import an image first!")
            return
        if messagebox.askyesno("Confirm Action", f"Are you sure you want to process these {len(self.file_paths)} video files?"):
            self.last_processed_count = 0
            self.set_ui_state(tk.DISABLED)
            settings = self.get_all_settings()
            self.processing_active = True
            self.start_timer()
            self.animate_status()
            
            thread = threading.Thread(target=modify_videos_worker, args=(self.root, self.file_paths, self.update_status, self.update_progress, settings))
            thread.daemon = True
            thread.start()

    def update_progress(self, count):
        """Callback function to update progress from the worker thread."""
        if count == -1: # Completion signal
            self.processing_active = False
            elapsed = int(time.time() - self.start_time)
            processed_count = getattr(self, 'last_processed_count', 0)
            status_text = f"Task finished. Total time: {elapsed} seconds."
            if processed_count > 0:
                 status_text = f"All tasks complete! Processed {processed_count} videos in {elapsed} seconds."
            self.status_label.config(text=status_text)
            # V10 Bug Fix: Re-enable the entire UI, not just some buttons.
            self.set_ui_state(tk.NORMAL)
        else:
            self.last_processed_count = count

    def set_ui_state(self, state):
        """Enables or disables all interactive widgets in the UI."""
        is_disabled = state == tk.DISABLED
        for child in self.root.winfo_children():
            if isinstance(child, (tk.Frame, tk.LabelFrame)):
                 for widget in child.winfo_children():
                    if isinstance(widget, (tk.Button, tk.Entry, ttk.Scale, ttk.Radiobutton, tk.Listbox, ttk.Combobox)):
                        try:
                           widget.config(state=state)
                        except tk.TclError:
                           pass # Ignore widgets that don't support the 'state' property
        if not is_disabled:
            self.update_button_state()
            self.toggle_controls()

    def run_preview_or_set(self):
        """Determines whether to run the batch preview or the position setting window."""
        is_local_overlay = self.perturbation_mode_var.get() == "Custom Image" and self.custom_image_mode_var.get() == "Local Overlay"
        if is_local_overlay:
            self.set_positions_for_selected()
        else:
            self.preview_all_videos()

    def set_positions_for_selected(self):
        """Opens the position/scale window for the selected video."""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Action Error", "Please select a video from the list to set its watermark position.")
            return
        if not self.custom_image_data:
            messagebox.showwarning("Action Error", "Please import an image first.")
            return
            
        import_heavy_libraries()
        video_path = self.file_paths[selected_indices[0]]
        filename = os.path.basename(video_path)
        self.update_status(f"Setting position for {filename}...")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): raise IOError("Cannot open video")
            fps = cap.get(5)
            cap.release()
            if fps == 0: raise ValueError(f"Cannot read FPS from {filename}.")
            
            target_frames = parse_frame_string(self.frames_var.get(), fps)
            if not target_frames:
                messagebox.showwarning("Action Error", "No valid target frames specified.")
                return

            initial_settings = self.watermark_positions.get(filename, {})
            pos_selector = PositionAndScaleWindow(self.root, video_path, self.custom_image_data, target_frames, initial_settings)
            self.root.wait_window(pos_selector)
            
            self.watermark_positions[filename] = pos_selector.final_settings
            self.update_status(f"Updated position/scale settings for {filename}.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def preview_all_videos(self):
        """Generates and displays a batch preview of the effects."""
        if not self.file_paths:
            messagebox.showwarning("Preview Error", "Please select video files first.")
            return
        if self.perturbation_mode_var.get() == "Custom Image" and not self.custom_image_data:
            messagebox.showwarning("Preview Error", "Please import an image first.")
            return
            
        import_heavy_libraries()
        self.update_status("Generating batch preview...")
        preview_win = MultiPreviewWindow(self.root)
        try:
            settings = self.get_all_settings()
            for video_path in self.file_paths:
                filename = os.path.basename(video_path)
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened(): continue
                fps = cap.get(5)
                if fps == 0: continue
                
                target_frames = parse_frame_string(self.frames_var.get(), fps)
                if not target_frames: continue
                
                # Use the last target frame for the preview
                preview_frame_index = target_frames[-1]
                cap.set(cv2.CAP_PROP_POS_FRAMES, preview_frame_index)
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    continue
                
                modified_frame = apply_perturbation(frame, settings["noise_level"], settings, filename, preview_frame_index, preview_frame_index)
                img_pil = Image.fromarray(cv2.cvtColor(modified_frame, cv2.COLOR_BGR2RGB))
                img_pil.thumbnail((800, 600))
                preview_win.add_image(filename, img_pil)
                cap.release()
        except Exception as e:
            messagebox.showerror("Preview Failed", str(e))
            preview_win.destroy()
        finally:
            self.update_status("Ready")

    def toggle_controls(self):
        """Dynamically enables/disables UI elements based on current selections."""
        is_custom_img = self.perturbation_mode_var.get() == "Custom Image"
        is_fade = self.transition_mode_var.get() == "Fade In/Out"
        is_tile = self.custom_image_mode_var.get() == "Tile"
        
        # Enable/disable all children of the custom image frame
        for child in self.custom_image_frame.winfo_children():
            # The state of some widgets might be tk.NORMAL even if the frame is disabled
            # so we explicitly set it.
            try:
                child.config(state=tk.NORMAL if is_custom_img else tk.DISABLED)
            except tk.TclError:
                pass
        
        # Fine-tune specific controls
        self.transition_window_entry.config(state=tk.NORMAL if is_fade else tk.DISABLED)
        self.transition_window_label.config(state=tk.NORMAL if is_fade else tk.DISABLED)
        
        zoom_state = tk.NORMAL if is_custom_img and is_tile else tk.DISABLED
        self.zoom_label.config(state=zoom_state)
        self.zoom_scale.config(state=zoom_state)
        self.zoom_entry.config(state=zoom_state)
        
        # Update the text of the preview/set button
        self.preview_set_button.config(text="Set Overlay Positions" if is_custom_img and not is_tile else "Preview All Effects")

    def import_image(self):
        """Opens a file dialog to select an image for the custom overlay."""
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        if path:
            try:
                self.custom_image_data = Image.open(path)
                self.image_path_label.config(text=path)
            except Exception as e:
                messagebox.showerror("Import Failed", f"Could not load image: {e}")
                self.custom_image_data = None
                self.image_path_label.config(text="No image selected")

    def delete_selected(self):
        """Removes the selected files from the listbox."""
        selected_indices = self.file_listbox.curselection()
        # Delete from the end to avoid index shifting issues
        for i in sorted(selected_indices, reverse=True):
            del self.file_paths[i]
            self.file_listbox.delete(i)
        self.update_button_state()

    def clear_list(self):
        """Clears all files from the listbox."""
        self.file_paths = []
        self.file_listbox.delete(0, tk.END)
        self.update_button_state()
        self.update_status("List cleared")

    def select_files(self, *args):
        """Opens a file dialog to select video files."""
        files = filedialog.askopenfilenames(title="Select Videos", filetypes=[("MP4", "*.mp4"), ("All files", "*.*")])
        if files:
            self.file_paths.extend(files)
            self.file_listbox.delete(0, tk.END)
            for f in self.file_paths:
                self.file_listbox.insert(tk.END, os.path.basename(f))
            self.update_button_state()

    def update_button_state(self):
        """Updates the state of buttons based on whether files are loaded."""
        has_files = bool(self.file_paths)
        state = tk.NORMAL if has_files else tk.DISABLED
        self.preview_set_button.config(state=state)
        self.delete_button.config(state=state)
        self.clear_button.config(state=state)
        self.modify_button.config(state=state)

    # --- Settings and State Management ---
    def get_all_settings(self):
        """Collects all user-defined settings from the UI into a dictionary."""
        return {
            "noise_level": self.noise_scale_var.get(),
            "suffix": self.suffix_var.get(),
            "frames": self.frames_var.get(),
            "transition_mode": self.transition_mode_var.get(),
            "transition_window": self.transition_window_var.get(),
            "perturbation_mode": self.perturbation_mode_var.get(),
            "custom_image_path": self.image_path_label.cget("text"),
            "custom_image_mode": self.custom_image_mode_var.get(),
            "custom_image_zoom": self.zoom_var.get(),
            "watermark_positions": self.watermark_positions,
            "custom_image_data": self.custom_image_data
        }

    def load_settings(self):
        """Loads settings from the config file at startup."""
        try:
            with open(CONFIG_FILE, 'r') as f:
                settings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            settings = {}
        
        self.noise_scale_var.set(settings.get("noise_level", 50))
        self.suffix_var.set(settings.get("suffix", "-m"))
        self.frames_var.set(settings.get("frames", "1,3,4,3.0s"))
        self.transition_mode_var.set(settings.get("transition_mode", "Fade In/Out"))
        self.transition_window_var.set(settings.get("transition_window", 14))
        self.perturbation_mode_var.set(settings.get("perturbation_mode", "Color Noise"))
        self.custom_image_mode_var.set(settings.get("custom_image_mode", "Tile"))
        self.zoom_var.set(settings.get("custom_image_zoom", 100))
        self.watermark_positions = settings.get("watermark_positions", {})
        
        img_path = settings.get("custom_image_path")
        if img_path and os.path.exists(img_path):
            try:
                self.custom_image_data = Image.open(img_path)
                self.image_path_label.config(text=img_path)
            except Exception:
                self.image_path_label.config(text="Failed to load image")

    def save_settings(self):
        """Saves current settings to the config file upon closing."""
        settings_to_save = self.get_all_settings()
        # Do not save the actual image data in the JSON file
        del settings_to_save['custom_image_data']
        # If the image path is no longer valid, save an empty string
        if not os.path.exists(settings_to_save.get("custom_image_path", "")):
            settings_to_save["custom_image_path"] = ""
        with open(CONFIG_FILE, 'w') as f:
            json.dump(settings_to_save, f, indent=4)

    # --- Timer and Status Animation ---
    def start_timer(self):
        """Starts the elapsed time counter."""
        self.start_time = time.time()
        def timer_update():
            if self.processing_active:
                elapsed = int(time.time() - self.start_time)
                self.timer_label.config(text=f"Time: {elapsed//60:02d}:{elapsed%60:02d}")
                self.root.after(1000, timer_update)
        timer_update()

    def animate_status(self):
        """Animates the status bar text to indicate processing."""
        if not self.processing_active: return
        states = ['Processing.', 'Processing..', 'Processing...']
        self.status_label.config(text=states[self.status_animation_counter % 3])
        self.status_animation_counter += 1
        self.root.after(500, self.animate_status)

    def update_status(self, message):
        """Updates the status bar with a new message."""
        self.status_label.config(text=message)

    def on_closing(self):
        """Saves settings and destroys the main window."""
        self.save_settings()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoModifierApp(root)
    root.mainloop()