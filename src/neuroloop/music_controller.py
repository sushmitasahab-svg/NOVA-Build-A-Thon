"""
music_controller.py

Controls music playback based on readiness state feedback.
Uses pygame-ce for audio playback.
"""

import pygame
from pathlib import Path


class MusicController:
    """
    Manages music playback with volume control linked to readiness feedback.
    """

    def __init__(self, music_file: str = None, auto_init: bool = True):
        """
        Initialize the music controller.

        Args:
            music_file: Path to music file to load (optional)
            auto_init: If True, automatically initialize pygame mixer on first use
        """
        self.music_file = music_file
        self.auto_init = auto_init
        self._mixer_initialized = False
        self.is_playing = False
        self.current_volume = 0.7

        if auto_init and music_file:
            self._ensure_mixer()

    def _ensure_mixer(self):
        """Lazy-initialize pygame mixer on first use."""
        if not self._mixer_initialized:
            pygame.mixer.init()
            self._mixer_initialized = True

    def load_music(self, music_file: str):
        """
        Load a music file.

        Args:
            music_file: Path to the music file
        """
        self._ensure_mixer()
        self.music_file = music_file
        try:
            pygame.mixer.music.load(music_file)
        except pygame.error as e:
            print(f"Error loading music file {music_file}: {e}")

    def play(self, loops: int = -1):
        """
        Start playing the loaded music.

        Args:
            loops: Number of times to loop (-1 = infinite)
        """
        if not self.music_file:
            print("No music file loaded. Call load_music() first.")
            return

        self._ensure_mixer()
        if not self.is_playing:
            pygame.mixer.music.play(loops=loops)
            self.is_playing = True

    def pause(self):
        """Pause the currently playing music."""
        self._ensure_mixer()
        if self.is_playing:
            pygame.mixer.music.pause()
            self.is_playing = False

    def unpause(self):
        """Resume paused music."""
        self._ensure_mixer()
        if not self.is_playing and pygame.mixer.music.get_busy():
            pygame.mixer.music.unpause()
            self.is_playing = True

    def stop(self):
        """Stop music playback."""
        self._ensure_mixer()
        pygame.mixer.music.stop()
        self.is_playing = False

    def set_volume(self, volume: float):
        """
        Set music volume.

        Args:
            volume: Volume level 0.0 to 1.0
        """
        self._ensure_mixer()
        self.current_volume = max(0.0, min(1.0, volume))
        pygame.mixer.music.set_volume(self.current_volume)

    def update_from_readiness(self, readiness_score: float):
        """
        Update music playback based on readiness score (0.0 to 1.0).
        
        Higher readiness typically means:
        - Volume increases
        - Music plays faster/more energetic

        Args:
            readiness_score: Readiness value from 0.0 to 1.0
        """
        # Map readiness to volume (0.0-1.0 range)
        volume = max(0.0, min(1.0, readiness_score))
        self.set_volume(volume)

        # Auto-play when readiness reaches a certain threshold
        if readiness_score > 0.5 and not self.is_playing and self.music_file:
            self.play()
        elif readiness_score < 0.3 and self.is_playing:
            self.pause()

    def get_status(self) -> dict:
        """
        Get current playback status.

        Returns:
            Dictionary with status information
        """
        self._ensure_mixer()
        return {
            "is_playing": self.is_playing,
            "current_volume": self.current_volume,
            "music_file": self.music_file,
            "mixer_busy": pygame.mixer.music.get_busy(),
        }

    def cleanup(self):
        """Stop music and clean up pygame mixer."""
        if self._mixer_initialized:
            self.stop()
            pygame.mixer.quit()
            self._mixer_initialized = False
