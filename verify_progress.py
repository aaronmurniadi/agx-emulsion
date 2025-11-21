from agx_emulsion.process.core.process import photo_process, photo_params
import numpy as np
import sys
import os

# Add current directory to sys.path
sys.path.append(os.getcwd())

def test_callback():
    print("Starting verification...")
    try:
        params = photo_params()
        # Use small image for speed
        image = np.random.rand(32, 32, 3)
        
        # Disable slow things
        params.settings.use_camera_lut = False
        params.settings.use_enlarger_lut = False
        params.settings.use_scanner_lut = False
        
        def cb(name, step, total):
            print(f"Callback received: Node={name}, Step={step}/{total}")
            
        print("Running process...")
        photo_process(image, params, progress_callback=cb)
        print("Verification finished.")
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_callback()
