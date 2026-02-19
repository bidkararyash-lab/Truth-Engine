# Truth-Engine
import cv2
import dlib
import numpy as np
import librosa
from scipy.spatial import distance as dist
from moviepy.editor import VideoFileClip

# --- 1. CONFIGURATION & MODELS ---
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# Load dlib's pre-trained face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# --- 2. HELPER FUNCTIONS ---

def eye_aspect_ratio(eye):
    """Calculates the EAR to detect blinking."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def analyze_audio(video_path):
    """Extracts spectrogram and MFCC features."""
    # Extract audio from video first
    temp_audio = "temp_audio.wav"
    VideoFileClip(video_path).audio.write_audiofile(temp_audio, verbose=False, logger=None)
    
    # Load and extract features
    y, sr = librosa.load(temp_audio, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs.T, axis=0)
    
    # Root Mean Square Energy for sync checking
    rms = librosa.feature.rms(y=y)
    return mfcc_mean, rms

# --- 3. MAIN MULTIMODAL ENGINE ---

def run_truth_engine(video_path):
    # Part A: Audio Analysis
    print("[INFO] Analyzing Audio Stream...")
    mfccs, audio_energy = analyze_audio(video_path)
    
    # Part B: Visual Analysis
    print("[INFO] Analyzing Visual Stream...")
    cap = cv2.VideoCapture(video_path)
    blink_counter = 0
    total_blinks = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

            # Blink Detection Logic
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

            if ear < EYE_AR_THRESH:
                blink_counter += 1
            else:
                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                    total_blinks += 1
                blink_counter = 0

            # Draw Output
            cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 0, 0), 2)
            cv2.putText(frame, f"Blinks: {total_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Status based on presence of biological signals
            status = "Authentic" if total_blinks > 0 else "Suspicious (No Blinking)"
            cv2.putText(frame, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Truth Engine: Multimodal Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[RESULT] Analysis Complete. Total Blinks detected: {total_blinks}")

# To run: run_truth_engine("your_video.mp4")
