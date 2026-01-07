import os
import sys
import threading
import queue
import ctypes
from collections import OrderedDict
from typing import OrderedDict as TOrderedDict

# NVDA Core Imports
from nvwave import WavePlayer, AudioPurpose
from logHandler import log
import synthDriverHandler
from synthDriverHandler import (
    SynthDriver as BaseSynthDriver,
    VoiceInfo,
    synthIndexReached,
    synthDoneSpeaking,
)
from speech.commands import IndexCommand, VolumeCommand, BreakCommand

# --- PATH CONFIGURATION ---
DRIVER_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the 'libs' folder to the search path for Python modules
LIBS_DIR = os.path.join(DRIVER_DIR, "libs")
if LIBS_DIR not in sys.path:
    sys.path.insert(0, LIBS_DIR)

# Try importing dependencies and the helper module
try:
    import numpy as np
    # Import the V2 helper classes provided in helper.py
    from .helper import load_text_to_speech, load_voice_style, AVAILABLE_LANGS, chunk_text
    log.info("Supertonic 2: Dependencies (incl. numpy from libs) and helper loaded.")
except ImportError as e:
    log.error(f"Supertonic 2: Critical error loading dependencies: {e}")
    # Fallback languages if import fails
    AVAILABLE_LANGS = ["en", "ko", "es", "pt", "fr"]

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
        """
        Streaming worker: Processes text in small chunks for minimal latency.
        """
        ctypes.windll.ole32.CoInitialize(None)
        log.info("Supertonic 2: Streaming queue worker started.")
        while not self.stop_event.is_set():
            try:
                # Wait for request: (text, lang_code, style_obj, speed, index)
                request = self.driver._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self.cancel_event.clear()
            text, lang, style, speed, index = request
            
            try:
                if self.cancel_event.is_set():
                    continue

                if text and text.strip() and self.driver.tts_engine:
                    # Split text directly into small chunks for fast feedback (max 150 characters)
                    chunks = chunk_text(text, max_len=150)
                    
                    for chunk in chunks:
                        if self.cancel_event.is_set():
                            break
                        
                        # Convert the internal quality value to an integer (1-15)
                        steps_value = int(self.driver._quality)
                        
                        # Use the internal _infer method for direct control per chunk
                        # FIX: Languages must be passed as a list: [lang]
                        wav, _ = self.driver.tts_engine._infer(
                            [chunk], 
                            [lang], 
                            style, 
                            total_step=steps_value, 
                            speed=speed
                        )
                        
                        if not self.cancel_event.is_set() and wav is not None:
                            # Volume processing (Updated gain factor to 2.0)
                            volume_factor = (self.driver._volume / 100.0) * 2.0
                            wav_amplified = wav * volume_factor
                            
                            # Clip to prevent distortion
                            wav_clamped = np.clip(wav_amplified, -1.0, 1.0)
                            
                            # Convert to 16-bit PCM
                            audio_int16 = (wav_clamped * 32767).astype(np.int16).tobytes()
                            
                            if self.driver._player:
                                self.driver._player.feed(audio_int16)
                
                if index is not None and not self.cancel_event.is_set():
                    synthIndexReached.notify(synth=self.driver, index=index)
                
            except Exception as e:
                log.error(f"Supertonic 2: Synthesis error: {e}")
            finally:
                self.driver._request_queue.task_done()
                synthDoneSpeaking.notify(synth=self.driver)
        
        ctypes.windll.ole32.CoUninitialize()

# =========================================================================
# MAIN SYNTHDRIVER CLASS
# =========================================================================

class SynthDriver(BaseSynthDriver):
    name = "supertonic"
    description = "Supertonic 2 AI TTS"

    @classmethod
    def check(cls):
        return True

    # Available voices with names mapped to IDs
    _available_voices = OrderedDict([
        ("M1", "Alex"), ("M2", "James"), ("M3", "Robert"), ("M4", "Sam"), ("M5", "Daniel"),
        ("F1", "Sarah"), ("F2", "Lily"), ("F3", "Jessica"), ("F4", "Olivia"), ("F5", "Emily")
    ])

    # Language code translations
    _lang_names = OrderedDict([
        ("en", "English"),
        ("ko", "Korean"),
        ("es", "Spanish"),
        ("pt", "Portuguese"),
        ("fr", "French")
    ])

    # Options for the Quality Combobox (1-15)
    _quality_options = OrderedDict([(str(i), str(i)) for i in range(1, 16)])

    supportedCommands = frozenset([IndexCommand, VolumeCommand, BreakCommand])
    supportedNotifications = frozenset([synthIndexReached, synthDoneSpeaking])

    # Use DriverSetting for the Quality option to force a combobox.
    supportedSettings = (
        BaseSynthDriver.VoiceSetting(),
        BaseSynthDriver.VariantSetting(),
        BaseSynthDriver.VolumeSetting(),
        synthDriverHandler.DriverSetting("quality", "Quality (Steps 1-15)"),
    )

    def __init__(self):
        # Default starting voice: Sarah
        self._current_voice_id = "F1"
        self._current_lang = "en"
        # Standard values: Volume 50 and Quality 5
        self._volume = 50
        self._quality = "5" 
        
        self.tts_engine = None
        self._player = None
        self.current_style_obj = None
        
        super(SynthDriver, self).__init__()
        
        # Determine directory structure
        self.model_base_dir = os.path.join(DRIVER_DIR, "models")
        
        self._request_queue = queue.Queue()
        self._voice_loaded_event = threading.Event()

        # Start loading models in a separate thread
        self._loader_thread = threading.Thread(target=self._initialize_async)
        self._loader_thread.daemon = True
        self._loader_thread.start()

        # Start the speech worker
        self._worker_thread = _SynthQueueThread(driver=self)
        self._worker_thread.start()

    def _initialize_async(self):
        ctypes.windll.ole32.CoInitialize(None)
        try:
            onnx_dir = os.path.join(self.model_base_dir, "onnx")
            
            if not os.path.exists(os.path.join(onnx_dir, "tts.json")):
                 log.error(f"Supertonic 2: tts.json not found in {onnx_dir}")
                 return

            self.tts_engine = load_text_to_speech(onnx_dir, use_gpu=False)
            self._load_style(self._current_voice_id)
            
            self._player = WavePlayer(
                channels=1,
                samplesPerSec=44100, 
                bitsPerSample=16,
                purpose=AudioPurpose.SPEECH
            )
            
            self._voice_loaded_event.set()
            log.info("Supertonic 2: Synthesizer successfully loaded.")
        except Exception as e:
            log.error(f"Supertonic 2: Initialization failed: {e}")

    def _load_style(self, voice_id):
        try:
            style_path = os.path.join(self.model_base_dir, "voice_styles", f"{voice_id}.json")
            if os.path.exists(style_path):
                self.current_style_obj = load_voice_style([style_path])
                log.info(f"Supertonic 2: Voice {voice_id} loaded.")
            else:
                log.error(f"Supertonic 2: Voice file not found: {style_path}")
        except Exception as e:
            log.error(f"Supertonic 2: Error loading voice: {e}")

    def _get_availableVoices(self) -> TOrderedDict[str, VoiceInfo]:
        voices = OrderedDict()
        for v_id, name in self._available_voices.items():
            voices[v_id] = VoiceInfo(v_id, name)
        return voices

    def _get_voice(self): return self._current_voice_id
    def _set_voice(self, value):
        if value in self._available_voices:
            self._current_voice_id = value
            if self.tts_engine:
                self._load_style(value)

    def _get_availableVariants(self):
        return OrderedDict((code, VoiceInfo(code, name)) for code, name in self._lang_names.items())

    def _get_variant(self):
        return self._current_lang

    def _set_variant(self, value):
        if value in self._lang_names:
            self._current_lang = value

    def _get_availableQualitys(self):
        """Provides the dropdown/combobox display in NVDA."""
        return OrderedDict((key, VoiceInfo(key, label)) for key, label in self._quality_options.items())

    def _get_quality(self): return self._quality
    def _set_quality(self, value):
        if value in self._quality_options:
            self._quality = value

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
            # Speed locked at the recommended 1.05
            speed_factor = 1.05
            self._request_queue.put((
                combined_text, 
                self._current_lang, 
                self.current_style_obj, 
                speed_factor, 
                last_index
            ))

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
        self.tts_engine = None
