import tkinter as tk
from tkinter import filedialog, messagebox, Menu
import threading
import cv2
from deepface import DeepFace
import numpy as np
from retinaface import RetinaFace
import subprocess


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    normalized_img = gray_img / 255.0
    resized_img = cv2.resize(normalized_img, (224, 224))
    return resized_img


def weighted_average_results(results):
    avg_result = results[0].copy()
    weights = [1.0] * len(results)

    for i, res in enumerate(results[1:], start=1):
        avg_result["age"] += res["age"] * weights[i]
        if res["gender"] == "Woman":
            avg_result["gender"] = "Woman"
        for key in avg_result["race"]:
            avg_result["race"][key] += res["race"][key] * weights[i]
        for key in avg_result["emotion"]:
            avg_result["emotion"][key] += res["emotion"][key] * weights[i]

    total_weight = sum(weights)
    avg_result["age"] /= total_weight
    for key in avg_result["race"]:
        avg_result["race"][key] /= total_weight
    for key in avg_result["emotion"]:
        avg_result["emotion"][key] /= total_weight

    return avg_result


def analyze_with_models(img_path):
    results = []
    models = ["VGG-Face", "Facenet", "OpenFace", "DeepID", "ArcFace", "Dlib"]
    preprocessed_img = preprocess_image(img_path)
    for model in models:
        try:
            result = DeepFace.analyze(
                img_path=img_path,
                actions=("age", "gender", "race", "emotion"),
                enforce_detection=True,
                detector_backend="retinaface",
            )
            if isinstance(result, list):
                result = result[0]
            results.append(result)
        except Exception as e:
            print(f"Error Analyzing With Model {model}: {str(e)}")
    return results


def face_analyze(img_path: str, result_text) -> None:
    try:
        faces = RetinaFace.extract_faces(img_path)
        if not faces:
            raise ValueError(
                "Face Could Not Be Detected In The Image. Please Use A Clear Face Image."
            )

        results = analyze_with_models(img_path)

        if not results:
            raise Exception("No Results From Analysis!")

        result_dict = weighted_average_results(results)

        result_str = f'[+] Age: {round(result_dict["age"], 2)}\n'
        result_str += f'[+] Gender: {result_dict["gender"]}\n'
        result_str += f"[+] Race:\n"

        for k, v in result_dict["race"].items():
            result_str += f"{k} - {round(v, 2)}%\n"

        result_str += f"[+] Emotion:\n"
        for k, v in result_dict["emotion"].items():
            result_str += f"{k} - {round(v, 2)}%\n"

        result_str += "\n[INFO] Model: DeepFace Analysis\n"
        result_str += "[INFO] Developed By A&J"

        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, result_str)

    except Exception as _ex:
        messagebox.showerror("Error", f"An error occurred: {_ex}")


def upload_image(result_text):
    img_path = filedialog.askopenfilename(
        title="Select Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if img_path:
        threading.Thread(target=face_analyze, args=(img_path, result_text)).start()


def real_time_emotion_detection():
    def process_frame():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = RetinaFace.detect_faces(rgb_frame)

            for face in faces.values():
                x, y, w, h = face["facial_area"]
                face_roi = rgb_frame[y : y + h, x : x + w]
                result = DeepFace.analyze(
                    face_roi, actions=["age", "emotion"], enforce_detection=False
                )
                age = result[0]["age"]
                emotion = result[0]["dominant_emotion"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"{emotion}, {age}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Real-Time Emotion & Age Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    cap = cv2.VideoCapture(0)
    threading.Thread(target=process_frame).start()


def run_other_model(script_name):
    try:
        subprocess.run(["python", script_name], check=True)
    except Exception as e:
        messagebox.showerror("Error", f"Failed To Run {script_name}: {e}")


def main():
    root = tk.Tk()
    root.title("Multimodal Analysis Developed By A&J")
    root.geometry("1920x1080")
    root.configure(bg="#343f71")

    menubar = Menu(root)
    root.config(menu=menubar)

    file_menu = Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Open Image", command=lambda: upload_image(result_text))
    file_menu.add_command(label="Exit", command=root.quit)

    other_models_menu = Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Other Models", menu=other_models_menu)
    other_models_menu.add_command(
        label="Face Recognition Based Attendance System",
        command=lambda: run_other_model("SingleModel.py"),
    )
    other_models_menu.add_command(
        label="English To Hindi-Gujarati Translator",
        command=lambda: run_other_model("MultiModel2.py"),
    )
    other_models_menu.add_command(
        label="Stock Market Price Checker",
        command=lambda: run_other_model("MultiModel3.py"),
    )
    other_models_menu.add_command(
        label="Enhanced Hand Detection Program",
        command=lambda: run_other_model("MultiModel4.py"),
    )
    other_models_menu.add_command(
        label="Hand Tracking & Brightness Control System",
        command=lambda: run_other_model("MultiModel5.py"),
    )

    help_menu = Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(
        label="About",
        command=lambda: messagebox.showinfo(
            "About", "Multimodal Analysis v1.0\nDeveloped By A&J"
        ),
    )

    header = tk.Label(
        root,
        text="DeepFace Analysis",
        font=("Helvetica", 30, "bold"),
        fg="#ffffff",
        bg="#343f71",
    )
    header.pack(pady=20)

    instruction_frame = tk.Frame(root, bg="#343f71")
    instruction_frame.pack(pady=10)

    instruction_label = tk.Label(
        instruction_frame,
        text="Click 'Upload Image' To Analyze Face From An Image Or 'Start Real-Time' For Webcam Analysis.",
        font=("Helvetica", 18),
        fg="#ffffff",
        bg="#343f71",
    )
    instruction_label.pack()

    def animate_text():
        current_color = instruction_label.cget("fg")
        next_color = "#ff9900" if current_color == "#ffffff" else "#ffffff"
        instruction_label.config(fg=next_color)
        root.after(500, animate_text)

    animate_text()

    result_text = tk.Text(
        root,
        height=15,
        width=70,
        bg="#37475a",
        fg="#ffffff",
        font=("Courier", 15),
        wrap=tk.WORD,
    )
    result_text.pack(pady=20)

    button_frame = tk.Frame(root, bg="#343f71")
    button_frame.pack(pady=20)

    upload_button = tk.Button(
        button_frame,
        text="Upload Image For Face Analysis",
        command=lambda: upload_image(result_text),
        bg="#c0392b",
        fg="#ffffff",
        activebackground="#009500",
        activeforeground="#ffffff",
        font=("Helvetica", 15, "bold"),
        relief=tk.RAISED,
    )
    upload_button.grid(row=0, column=0, padx=10, pady=10)

    emotion_button = tk.Button(
        button_frame,
        text="Start Real-Time Emotion Detection",
        command=real_time_emotion_detection,
        bg="#c0392b",
        fg="#ffffff",
        activebackground="#009500",
        activeforeground="#ffffff",
        font=("Helvetica", 15, "bold"),
        relief=tk.RAISED,
    )
    emotion_button.grid(row=0, column=1, padx=10, pady=10)

    footer = tk.Label(
        root,
        text="Developed By A&J",
        font=("Helvetica", 15),
        fg="#ffea00",
        bg="#343f71",
    )
    footer.pack(side=tk.BOTTOM, pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
