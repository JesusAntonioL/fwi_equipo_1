from cx_Freeze import setup, Executable
import sys

# Dependencies are automatically detected, but it might need fine-tuning.
build_exe_options = {
    "packages": ["os", "cv2", "numpy", "PyQt6", "firebase_admin", "email", "smtplib", "ssl"],
    "include_files": [
        "wildfiredetection-72d0f-firebase-adminsdk-4dohx-79a6b85888.json",  # Include your Firebase credentials file
        # Add any other files your application needs, such as model files
    ],
    "excludes": ["tkinter"],  # Exclude unnecessary packages
}

# Base is set to "Win32GUI" for GUI applications on Windows
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="WildfireDetectionApp",
    version="0.1",
    description="Wildfire Detection Application",
    options={"build_exe": build_exe_options},
    executables=[Executable("WildFireUI_AirFlow.py", base=base)],  # Replace with your script's filename
)