import os
import sys
import threading
import queue
import ctypes
from collections import OrderedDict
from typing import OrderedDict as TOrderedDict, Optional, Set, Callable

# NVDA Core Imports
from nvwave import WavePlayer, AudioPurpose
from logHandler import log
from synthDriverHandler import (
    SynthDriver as BaseSynthDriver,
    VoiceInfo,
    synthIndexReached,
    synthDoneSpeaking,
)
from speech.commands import IndexCommand, VolumeCommand, BreakCommand

# --- PATH CONFIGURATION ---
DRIVER_DIR = os.path.dirname(os.path.abspath(__file__))
SUPERTONIC_LIBS_PATH = os.path.join(DRIVER_DIR, "libs")

if SUPERTONIC_LIBS_PATH not in sys.path:
    sys.path.insert(0, SUPERTONIC_LIBS_PATH)

try:
    import numpy as np
    import requests
    from supertonic import TTS
    log.info("Supertonic: Library and dependencies loaded successfully.")
except ImportError as e:
    log.error(f"Supertonic: Critical error loading library: {e}")

# =========================================================================
# SYNTHESIS QUEUE THREAD
# =========================================================================

class _SynthQueueThread(threading.Thread):
    def __init__(self, driver: 'SynthDriver'):
        super().__init__()
        self.driver = driver
        self.daemon = True
        self.stop_event = threading.Event()
        self.cancel_event = threading.Event()

    def run(self):
        ctypes.windll.ole32.CoInitialize(None)
        log.info("Supertonic: Queue worker started.")
        while not self.stop_event.is_set():
            try:
                # Wait for an item from the queue (text, style, index)
                request = self.driver._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self.cancel_event.clear()
            text, style, index = request
            
            try:
                if self.cancel_event.is_set():
                    continue

                # Only perform synthesis if there is actual text
                if text and text.strip():
                    wav, duration = self.driver.tts_voice.synthesize(
                        text, 
                        voice_style=style,
                        speed=1.25,
                        total_steps=2,
                        verbose=False
                    )
                    
                    if not self.cancel_event.is_set():
                        # Apply volume and increase base strength (gain factor 3.0)
                        # self.driver._volume is 0-100
                        volume_factor = (self.driver._volume / 100.0) * 3.0
                        wav_amplified = wav * volume_factor
                        
                        # Prevent digital distortion (clipping)
                        wav_clamped = np.clip(wav_amplified, -1.0, 1.0)
                        
                        audio_int16 = (wav_clamped * 32767).astype(np.int16).tobytes()
                        if self.driver._player:
                            self.driver._player.feed(audio_int16)
                
                # Report the index back for navigation
                if index is not None and not self.cancel_event.is_set():
                    synthIndexReached.notify(synth=self.driver, index=index)
                
            except Exception as e:
                log.error(f"Supertonic: Error during synthesis: {e}")
            finally:
                self.driver._request_queue.task_done()
                synthDoneSpeaking.notify(synth=self.driver)
        
        ctypes.windll.ole32.CoUninitialize()

# =========================================================================
# MAIN SYNTHDRIVER CLASS
# =========================================================================

class SynthDriver(BaseSynthDriver):
    name = "supertonic"
    description = "Supertonic AI TTS"

    @classmethod
    def check(cls):
        return True

    _available_voices = OrderedDict([
        ("F1", "Female 1"), ("F2", "Female 2"), ("F3", "Female 3"), ("F4", "Female 4"), ("F5", "Female 5"),
        ("M1", "Male 1"), ("M2", "Male 2"), ("M3", "Male 3"), ("M4", "Male 4"), ("M5", "Male 5")
    ])

    supportedCommands = frozenset([IndexCommand, VolumeCommand, BreakCommand])
    supportedNotifications = frozenset([synthIndexReached, synthDoneSpeaking])

    supportedSettings = {
        BaseSynthDriver.VoiceSetting(),
        BaseSynthDriver.VolumeSetting(),
    }

    def __init__(self):
        self._current_voice_id = "F1"
        self._volume = 100
        self.tts_voice = None
        self._player = None
        self.style = None
        
        super(SynthDriver, self).__init__()
        
        self.model_path = os.path.join(DRIVER_DIR, "models")
        self._request_queue = queue.Queue()
        self._voice_loaded_event = threading.Event()

        self._loader_thread = threading.Thread(target=self._initialize_async)
        self._loader_thread.daemon = True
        self._loader_thread.start()

        self._worker_thread = _SynthQueueThread(driver=self)
        self._worker_thread.start()

    def _initialize_async(self):
        ctypes.windll.ole32.CoInitialize(None)
        try:
            config_path = os.path.join(self.model_path, "config.json")
            if not os.path.exists(config_path):
                log.error("Supertonic: Models not found. Please check the installation.")
                return

            self.tts_voice = TTS(model_dir=self.model_path, auto_download=False)
            self.style = self.tts_voice.get_voice_style(voice_name=self._current_voice_id)
            
            self._player = WavePlayer(
                channels=1,
                samplesPerSec=44100,
                bitsPerSample=16,
                purpose=AudioPurpose.SPEECH
            )
            self._voice_loaded_event.set()
            log.info("Supertonic: Fully loaded.")
        except Exception as e:
            log.error(f"Supertonic: Initialization failed: {e}")

    def _get_availableVoices(self) -> TOrderedDict[str, VoiceInfo]:
        voices = OrderedDict()
        for v_id, name in self._available_voices.items():
            voices[v_id] = VoiceInfo(v_id, name)
        return voices

    def _get_voice(self): return self._current_voice_id
    def _set_voice(self, value):
        if value in self._available_voices:
            self._current_voice_id = value
            if self.tts_voice:
                self.style = self.tts_voice.get_voice_style(voice_name=value)

    def _get_volume(self): return self._volume
    def _set_volume(self, value): self._volume = value

    def speak(self, speechSequence):
        if not self._voice_loaded_event.is_set():
            return

        text_parts = []
        last_index = None

        for item in speechSequence:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, IndexCommand):
                last_index = item.index
        
        combined_text = "".join(text_parts)
        
        if combined_text.strip() or last_index is not None:
            self._request_queue.put((combined_text, self.style, last_index))

    def cancel(self):
        if hasattr(self, '_worker_thread'):
            self._worker_thread.cancel_event.set()

        if self._player:
            try:
                self._player.stop()
            except Exception:
                pass

        try:
            while not self._request_queue.empty():
                self._request_queue.get_nowait()
        except queue.Empty:
            pass

    def terminate(self):
        if hasattr(self, '_worker_thread'):
            self._worker_thread.stop_event.set()
        if self._player:
            self._player.close()
        self.tts_voice = None