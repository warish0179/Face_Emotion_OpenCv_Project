import cv2
from deepface import DeepFace

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

print("Starting stable tracking... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # FIX 1: Increased minNeighbors=10 and added minSize to STOP the vibration/flashing
    faces = face_cascade.detectMultiScale(
        gray_frame, 
        scaleFactor=1.2, 
        minNeighbors=10, 
        minSize=(100, 100)
    )

    # FIX 2: Only process the LARGEST face if multiple boxes appear
    if len(faces) > 0:
        # Sort faces by area (width * height) and pick the biggest one
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face

        face_roi = frame[y:y + h, x:x + w]

        try:
            # FIX 3: Force DeepFace to analyze your unique facial structure dynamically
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            
            # Mapping emotions to clear text + bright custom emojis
            emoji_dict = {
                'happy': ('HAPPY', '😊', (0, 255, 0)),       # Green
                'sad': ('SAD/CRYING', '😢', (255, 0, 0)),    # Blue
                'angry': ('ANGRY', '😠', (0, 0, 255)),        # Red
                'surprise': ('SURPRISED', '😮', (0, 255, 255)),# Yellow
                'neutral': ('NEUTRAL', '😐', (200, 200, 200)),# Gray
                'fear': ('FEAR', '😨', (255, 0, 255)),       # Magenta
                'disgust': ('DISGUST', '🤢', (0, 128, 0))     # Dark Green
            }
            
            emotion_name, emoji, text_color = emoji_dict.get(dominant_emotion, ('Analyzing...', '', (0, 255, 0)))
            display_text = f"{emotion_name} {emoji}"
            
        except Exception as e:
            display_text = "Analyzing..."
            text_color = (255, 255, 0)

        # Draw a solid, calm Cyan box around just ONE face
        cv2.rectangle(frame, (x, y), (x + w, y+h), (255, 255, 0), 2)
        
        # Display the emotion text and its unique emoji cleanly above your head
        cv2.putText(frame, display_text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    # Show the feed
    cv2.imshow('Real-time Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()