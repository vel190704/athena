# Test fixture attribution

## soccer_ball_kick.jpg

Source: [Soccer Ball about to be kicked.JPG](https://commons.wikimedia.org/wiki/File:Soccer_Ball_about_to_be_kicked.JPG),
Wikimedia Commons.

License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

Downsampled from the original 5184x2920 to 1280x720 (JPEG quality 90) to
keep the repository fixture small. Used in `test_ball_detector.py` as a
real photograph (not a synthetic idealization) to validate the YOLO
"sports ball" detection path — a `cv2.circle()`-drawn flat disc lacks the
lighting/texture signature a photo-trained COCO detector actually learned
to recognize, so this milestone's YOLO-path test specifically requires a
real photo rather than a synthetic one.
