import os
import requests
from logHandler import log

# List of files
REPO_URL = "https://huggingface.co/Supertone/supertonic/resolve/main"
FILES_TO_DOWNLOAD = [
    "config.json", "onnx/duration_predictor.onnx", "onnx/text_encoder.onnx",
    "onnx/vector_estimator.onnx", "onnx/vocoder.onnx", "onnx/unicode_indexer.json",
    "onnx/tts.json", "onnx/tts.yml"
]
for i in range(1, 6):
    FILES_TO_DOWNLOAD.append(f"voice_styles/F{i}.json")
    FILES_TO_DOWNLOAD.append(f"voice_styles/M{i}.json")

def onInstall():
    """Called by NVDA during installation."""
    
    base_dir = os.path.dirname(__file__)
    addon_dir = os.path.join(base_dir, "synthDrivers", "supertonic", "models")
    
    log.info(f"Supertonic: Starting download to {addon_dir}")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # Use a Session object so we can explicitly close the connection
    with requests.Session() as session:
        session.headers.update(headers)
        
        try:
            for f_path in FILES_TO_DOWNLOAD:
                target = os.path.join(addon_dir, f_path)
                
                target_dir = os.path.dirname(target)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                
                if not os.path.exists(target):
                    log.info(f"Supertonic: Downloading {f_path}...")
                    with session.get(f"{REPO_URL}/{f_path}", stream=True, timeout=30) as r:
                        r.raise_for_status()
                        with open(target, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=1024*1024):
                                if chunk:
                                    f.write(chunk)
                                    f.flush()
                                    os.fsync(f.fileno()) # Force write to disk
            
            log.info("Supertonic: Download completed successfully.")
            
        except Exception as e:
            log.error(f"Supertonic: Critical error during download: {e}")
            # We do not raise the error here to prevent NVDA from crashing, 
            # but the files might be incomplete.

    # At the end of the 'with' blocks, all file handles and network connections are closed.
    # NVDA can now safely rename the directory upon restart.