# VirtualMouseInterface

## Requirements

- Python 3.x
- OpenCV
- Mediapipe
- Numpy
- PyAutoGUI
- CVZone

## Installation

1. Clone the repository:

```bash
git clone https://github.com/busraakbulut/VirtualMouseInterface.git
```
Navigate to the project directory:


   ```bash
   cd VirtualMouseInterface
   ```
Install the required dependencies:


   ```bash
   pip install opencv-python  mediapipe  numpy pyautogui cvzone
   ```
## Usage
Run the main script:

 ```bash
   python main.py
   ```
   Adjust the sensitivity level when prompted (choose a number between 1 and 3) to control the mouse movement speed.
   
   Once the application is running, it will use your device's webcam to track your facial landmarks. Keep your face visible to the camera for accurate tracking.
   
   Look in different directions to control the mouse cursor. The application will move the cursor based on your head movements.
   
   Blinking your eyes simulates mouse clicks. A quick blink will simulate a left-click, while a longer blink will simulate a right-click.
   
   Press 'q' to exit the application.
   
   ##License
   This project is licensed under the MIT License.
   
   ## Acknowledgements
   The Mediapipe library for facial landmark detection.
   The OpenCV library for computer vision and image processing.
   The Numpy library for numerical computations.
   The PyAutoGUI library for simulating mouse movements and clicks.
   The CVZone library for additional computer vision functionality.
   
   ## Contributing
   Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.
