steps to run everything

- python -m venv venv
- .\venv\Scripts\Activate.ps1
<!-- - pip install "numpy<2.0.0,>=1.23.5" --force-reinstall1 -->
- pip install -r requirements.txt
- pip install opencv-python
- python -c "import cv2; print(cv2.__version__)"
- pip install ultralytics opencv-python pillow imagehash - install 
- python train_yolo.py   - use for running train yolo