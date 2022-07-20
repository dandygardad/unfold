# ``Unfold``
A Python project for **measuring distance between two ships** with **Stereo Camera**.

## Requirements
- Python 3.8 or newer
- PyTorch 1.7 or newer
- OpenCV
- Two cameras must be of the same model

## Install
```
git clone https://github.com/dandygardad/unfold.git

cd unfold

pip install -r requirements.txt
```

Rename `config.yaml.example` into `config.yaml` and edit the camera & detection config.

```
python main.py
```

## Features
- Live/Video source input
- Camera Calibration & Rectify
- Detect with Stereo YOLOv5 (Mini-version)
- Distance Measurement
- Root Mean Squared Error

---

🌸 from **Dandy Garda**
