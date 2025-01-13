# DeepFace Analysis
## Overview
This project provides a user-friendly GUI application for analyzing facial features using DeepFace. The application can analyze age, gender, race, and emotions from images or real-time video streams. It is developed using Python and Tkinter, and it leverages powerful models like DeepFace and RetinaFace for face detection and analysis.

### Features
1. **Image Analysis:** Upload an image and analyze facial features, including age, gender, race, and emotions.
2. **Real-Time Emotion Detection:** Use your webcam to detect emotions and approximate age in real-time.
3. **Integration with Other Models:** Access other multimodal models like:
   - Face Recognition Attendance System
   - English to Hindi-Gujarati Translator
   - Stock Market Price Checker
   - Enhanced Hand Detection Program
   - Hand Tracking & Brightness Control System
4. **User-Friendly Interface:** The application is simple and intuitive, designed for seamless interaction.

## Demonstration
Watch the full demonstration of the project on YouTube: [DeepFace Analysis Demo](https://youtu.be/86Xb8Kfm9fU?si=QC36LefljbtTxTra)

## How to Use
### Prerequisites
Ensure you have Python 3.8+ installed. Clone or download the repository to your local machine.

### Installation
This project is part of a larger multimodal system. The `requirements.txt` file contains dependencies for the entire system. If you only want to use this model, install the following minimal dependencies:

## Code Structure
- **`preprocess_image(img_path):`** Preprocesses the image for analysis.
- **`weighted_average_results(results):`** Aggregates results from multiple models using a weighted average.
- **`analyze_with_models(img_path):`** Runs analysis using various models and returns the results.
- **`face_analyze(img_path, result_text):`** Main function for analyzing faces and displaying results.
- **`upload_image(result_text):`** Handles image upload and starts the analysis.
- **`real_time_emotion_detection():`** Detects emotions and age in real-time using the webcam.
- **`run_other_model(script_name):`** Launches other models from the multimodal system.
- **`main():`** Initializes and runs the Tkinter GUI.

## Dependencies
- **DeepFace:** Used for facial analysis.
- **RetinaFace:** Used for face detection.
- **OpenCV:** Used for image and video processing.
- **NumPy:** Used for numerical computations.
- **Tkinter:** Used for the graphical user interface.

## Notes
- This project is part of a larger multimodal system. For a full exploration, download the entire multimodal system from the repository.
- For real-time emotion detection, ensure your webcam is connected and accessible.

## Contribution
Feel free to fork the repository, submit issues, or suggest improvements. Contributions are always welcome!

## License
This project is licensed under the MIT License. See the LICENSE file for details.

Developed by **A&J** as part of the Multimodal System.

