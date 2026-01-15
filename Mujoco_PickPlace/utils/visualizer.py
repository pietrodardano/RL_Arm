import mujoco as mj
from mujoco import viewer
import numpy as np
import time


class Visualizer:
    def __init__(self, model: mj.MjModel, data: mj.MjData = None, fps: int = 60):
        self.model = model
        self.data = data or mj.MjData(model)
        self.viewer = None
        self.fps = fps
        self.frame_time = 1.0 / fps if fps > 0 else 0
        self.last_render_time = 0
    
    def create(self):
        """Create and launch viewer."""
        self.viewer = viewer.launch_passive(self.model, self.data)
        self.last_render_time = time.time()
    
    def sync(self):
        """Sync viewer with data and enforce frame rate."""
        if self.viewer:
            current_time = time.time()
            elapsed = current_time - self.last_render_time
            
            if elapsed < self.frame_time:
                time.sleep(self.frame_time - elapsed)
            
            with self.viewer.lock():
                self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = False
                self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            
            self.last_render_time = time.time()
    
    def close(self):
        """Close viewer."""
        if self.viewer:
            self.viewer.close()
    
    def is_running(self) -> bool:
        return self.viewer is not None and self.viewer.isRunning()
    
    def set_fps(self, fps: int):
        """Set target FPS."""
        self.fps = fps
        self.frame_time = 1.0 / fps if fps > 0 else 0
    
    def grab_image(self) -> np.ndarray:
        """
        Capture the rendered image from the viewer.
        Returns:
            numpy array of shape (height, width, 3) with RGB pixel data (uint8)
        """
        if not self.viewer:
            return None
        
        img = np.zeros(
            (self.viewer.viewport.height, self.viewer.viewport.width, 3),
            dtype=np.uint8
        )
        mj.mjr_render(self.viewer.viewport, self.viewer.scn, self.viewer.ctx)
        mj.mjr_readPixels(img, None, self.viewer.viewport, self.viewer.ctx)
        img = np.flipud(img)  # flip image (OpenGL origin is bottom-left)
        return img
