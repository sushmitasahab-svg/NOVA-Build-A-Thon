"""
music_controller.py

Adds ONE thing to the existing feedback loop: soft background music
that responds to the GREEN/YELLOW/RED feedback state. Nothing about
EEG preprocessing, features, or the state machine itself is touched -
this module only listens to the state the loop already computes
(READY/HIGH_LOAD/PAUSE) and controls audio playback/volume.

STATE MAPPING (uses the states our state_machine.py already produces):
    READY     -> GREEN  -> music plays at normal volume
    HIGH_LOAD -> YELLOW -> music plays at a lower volume
    PAUSE     -> RED    -> music stops completely

RULE THAT MATTERS MOST: only react on an actual state CHANGE.
    Call on_state_change(new_state) only when the state differs from
    the last one - calling it every loop iteration with an unchanged
    state must do nothing (no restart, no volume re-set). This class
    itself also guards against redundant calls (see the early return
    in on_state_change), so even if the caller accidentally calls it
    every iteration, behavior stays correct.
"""

import pygame

MUSIC_FILE = "assets/audio/soft_music.mp3"
NORMAL_VOLUME = 0.5
YELLOW_VOLUME = 0.2

# Maps the feedback loop's existing state names to the GREEN/YELLOW/RED
# concept requested here - no new states are introduced.
_STATE_TO_COLOR = {
    "READY": "GREEN",
    "HIGH_LOAD": "YELLOW",
    "PAUSE": "RED",
}


class MusicController:
    def __init__(self, music_file=MUSIC_FILE,
                 normal_volume=NORMAL_VOLUME, yellow_volume=YELLOW_VOLUME):
        pygame.mixer.init()
        pygame.mixer.music.load(music_file)
        self.normal_volume = normal_volume
        self.yellow_volume = yellow_volume
        self._current_color = None
        self._playing = False

    def start(self):
        """Call once, at the very start of the feedback loop (GREEN, normal volume, looping)."""
        self._apply_color("GREEN")

    def on_state_change(self, feedback_state):
        """
        Call this only when the feedback loop's state actually changed
        (e.g. inside the existing `if transition["changed"]:` block).
        feedback_state is one of "READY", "HIGH_LOAD", "PAUSE" - the
        same strings the state machine already produces.
        """
        color = _STATE_TO_COLOR.get(feedback_state)
        if color is None or color == self._current_color:
            return  # unknown state, or no real change - do nothing
        self._apply_color(color)

    def _apply_color(self, color):
        self._current_color = color

        if color == "GREEN":
            if not self._playing:
                pygame.mixer.music.play(loops=-1)
                self._playing = True
            pygame.mixer.music.set_volume(self.normal_volume)

        elif color == "YELLOW":
            if not self._playing:
                pygame.mixer.music.play(loops=-1)
                self._playing = True
            pygame.mixer.music.set_volume(self.yellow_volume)

        elif color == "RED":
            pygame.mixer.music.stop()
            self._playing = False