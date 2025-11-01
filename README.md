Tracking Software README

A. Overview and Running Protocol
1. Outputs:
   - Per-individual CSV (and optional combined CSV), plus summary scripts.

2. File Organization:
   • main.py: main interface (launch the software)
   • tracker.py: tracking logic (IDs, CSV writing)
   • processor.py / contour.py: segmentation and contours (+ warm-up)
   • utils.py: display/conversion utilities
Appendix:
   • merge_tracks_gui.py: manual track merging (GUI)
   • more_metrics.py: additional post-processing metrics
   • summury_csv.py: global recap from per-ID CSVs

3. Launch:
   1) Start: python main.py
   2) Load a video (.mp4, .avi, …)
   3) Choose a preset, adjust sliders if needed, then Start

4. Key Parameters (UI):
   • Gray threshold: binarization threshold for detection
   • Area min/max: object size filter
   • Morph ellipse (odd): morphological denoising
   • Warm-up: start-of-video anti-noise stabilization
   • Background subtraction: None / KNN / MOG2 (+ combine mode)
   • Assignment: Greedy / Hungarian / Auto
   • Options: Only moving, Require motion to spawn, Single ID, etc.

5. Generated Outputs:
   5.1 Per-individual CSV (all presets)
   - One file per ID is created in the results/ folder.
   - Example: videoName_ID1.csv
   - Each frame produces one row with the following columns:
     • frame, id, raw_x, raw_y, flt_x, flt_y, w, h, area, delta_area

     Definitions:
     → raw_x, raw_y: raw coordinates (direct measurement on the detected contour)
     → flt_x, flt_y: filtered/tracked coordinates (tracker EMA smoothing)
     → w, h: width and height of the detected object
     → area: area of the detected contour
     → delta_area: absolute change in area between two consecutive frames (|area[n] − area[n−1]|)

   5.2 Combined CSV (optional)
   - A global CSV can be enabled; it concatenates rows for all IDs with the same column schema.

6. Visualization:
   • Display of contours (Rectangle / MinArea / Hull / Ellipse) and IDs over the video.

7. Best Practices:
   • Calibrate min/max area based on the individual type and image quality
   • Use flt_x/flt_y for analysis (trajectory is more stable than raw_*)
   • Enable warm-up (30–50 frames) to reduce false positives
   • Use summury_csv.py for a recap, then more_metrics.py if needed
   • If tracks need correction/merging, use merge_tracks_gui.py

8. Typical Workflow:
   • Load the video → Choose the preset → Start
   • Check the overlay and adjust if needed
   • Retrieve per-ID CSVs in results/
   • Run summury_csv.py for the recap.

B. Software Installation
Prerequisites:
   • System: Windows 10/11 (recommended) or Ubuntu 20.04+; macOS possible (MP4 encoding not tested).
   • Python: 3.10 to 3.12 (64-bit).
   • Internet access to install dependencies.

Steps:
1) Install Python 3.10–3.12 (64-bit)
   - On Windows, check "Add Python to PATH" during installation.
2) Copy the project folder onto the new PC (or clone from your repository).
3) Open a terminal in the project folder and create a virtual environment:
   python -m venv .venv
   (Windows) .venv\Scripts\activate
   (Linux/mac) source .venv/bin/activate
4) Install required libraries:
   pip install -r requirements_tracking.txt
5) (MP4/encoding) Install FFmpeg if needed:
   a. Windows: download the FFmpeg zip (ffmpeg.org), add ffmpeg\bin to PATH
   b. Ubuntu: sudo apt-get update && sudo apt-get install -y ffmpeg
6) Launch the application:
   python main.py

Main Dependencies (in requirements_tracking.txt):
   • opencv-python: video capture, image processing, MP4 encoding (via FFmpeg)
   • numpy: numerical computing
   • PySide6: graphical interface (Qt)
   • pyqtgraph: live plotting (optional)
   • pandas: summary scripts (summury_csv.py, more_metrics.py)

Quick Troubleshooting:
   • Unable to write MP4 video: check FFmpeg and write permissions for the folder.
   • Qt window won’t open / missing plugin: reinstall PySide6 (pip install --force-reinstall PySide6).
   • Low FPS: enable “Process at 0.75× scale (perf)”, reduce video size, or close heavy apps.
   • Empty CSV: verify that “Record CSV” is checked and that min_age_csv isn’t set too high (tracker.params).

