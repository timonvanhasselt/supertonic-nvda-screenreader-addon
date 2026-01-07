import os
import requests
from logHandler import log

# --- CONFIGURATION V2 ---
REPO_URL = "https://huggingface.co/Supertone/supertonic-2/resolve/main"

FILES_TO_DOWNLOAD = [
    "onnx/tts.json",
    "onnx/unicode_indexer.json",
    "onnx/duration_predictor.onnx",
    "onnx/text_encoder.onnx", 
    "onnx/vector_estimator.onnx", 
    "onnx/vocoder.onnx"
]

for i in range(1, 6):
    FILES_TO_DOWNLOAD.append(f"voice_styles/F{i}.json")
    FILES_TO_DOWNLOAD.append(f"voice_styles/M{i}.json")

def onInstall():
    
    base_dir = os.path.dirname(__file__)
    # Path to the driver directory
    addon_dir = os.path.join(base_dir, "synthDrivers", "supertonic", "models")
    
    log.info(f"Supertonic V2: Starting download to {addon_dir}")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    
    with requests.Session() as session:
        session.headers.update(headers)
        
        try:
            for f_path in FILES_TO_DOWNLOAD:
                # Determine local target path (maintains the onnx/ and voice_styles/ folders)
                target = os.path.join(addon_dir, f_path)
                
                target_dir = os.path.dirname(target)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                
                if not os.path.exists(target):
                    log.info(f"Supertonic V2: Downloading {f_path}...")
                    url = f"{REPO_URL}/{f_path}"
                    
                    with session.get(url, stream=True, timeout=30) as r:
                        if r.status_code == 404:
                            log.error(f"Supertonic V2: File not found (404): {url}")
                            continue
                        r.raise_for_status()
                        with open(target, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=1024*1024):
                                if chunk:
                                    f.write(chunk)
                                    f.flush()
                                    os.fsync(f.fileno())
            
            log.info("Supertonic V2: Download process completed.")
            
        except Exception as e:
            log.error(f"Supertonic V2: Error during download: {e}")
