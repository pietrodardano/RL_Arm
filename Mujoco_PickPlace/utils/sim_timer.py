"""Speed control utilities for simulation."""

import time


class SimulationTimer:
    """Control simulation playback speed."""
    
    def __init__(self, target_fps: int = 60):
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps if target_fps > 0 else 0
        self.last_time = time.time()
        self.paused = False
        self.pause_time = 0
    
    def tick(self):
        """Enforce frame rate timing."""
        if self.paused:
            self.pause_time = time.time()
            return
        
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)
        
        self.last_time = time.time()
    
    def set_fps(self, fps: int):
        """Change target FPS."""
        self.target_fps = fps
        self.frame_time = 1.0 / fps if fps > 0 else 0
        self.last_time = time.time()
    
    def pause(self):
        """Pause timing."""
        self.paused = True
        self.pause_time = time.time()
    
    def resume(self):
        """Resume timing."""
        if self.paused and self.pause_time > 0:
            # Skip the pause duration
            self.last_time = time.time()
        self.paused = False
    
    def is_paused(self) -> bool:
        return self.paused
