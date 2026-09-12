"""Download the YuNet detector and SFace recognizer ONNX models from OpenCV Zoo."""

from __future__ import annotations

import sys
import urllib.request

from faceid import config

URLS = {
    config.DETECTOR_MODEL_PATH: "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    config.RECOGNIZER_MODEL_PATH: "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def main() -> int:
    for path, url in URLS.items():
        if path.exists() and path.stat().st_size > 1024:
            print(f"[skip] {path.name} already present")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[get ] {path.name} ...")
        urllib.request.urlretrieve(url, path)
        print(f"[ok  ] {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
