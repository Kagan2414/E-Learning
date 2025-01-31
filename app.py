# app.py
import cv2
import numpy as np
import os
import time
from flask import Flask, render_template, Response, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Constants
FACE_DATA_DIR = "user_faces/"
MAX_LOGIN_ATTEMPTS = 3
LOCK_TIME = 300  # 5 minutes
os.makedirs(FACE_DATA_DIR, exist_ok=True)

# Store failed login attempts
login_attempts = {}

# Initialize face detector and SIFT
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
sift = cv2.SIFT_create()

def preprocess_face(frame):
    """Detect face and return preprocessed face image"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 1:
            x, y, w, h = faces[0]
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            return face
        return None
    except Exception as e:
        print(f"Error in preprocess_face: {str(e)}")
        return None

def compute_face_features(face_img):
    """Compute SIFT features for a face image"""
    try:
        if face_img is None:
            return None, None
        keypoints, descriptors = sift.detectAndCompute(face_img, None)
        return keypoints, descriptors
    except Exception as e:
        print(f"Error in compute_face_features: {str(e)}")
        return None, None

def match_faces(desc1, desc2, threshold=0.75):
    """Match face features and return similarity score"""
    try:
        if desc1 is None or desc2 is None:
            return 0
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)
        
        return len(good_matches)
    except Exception as e:
        print(f"Error in match_faces: {str(e)}")
        return 0

def capture_face(username):
    """Capture and save user's face features"""
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Camera not accessible"

        best_face = None
        best_features = None
        
        # Try to get a good face for 5 seconds
        start_time = time.time()
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                continue
            
            face = preprocess_face(frame)
            if face is not None:
                _, features = compute_face_features(face)
                if features is not None and (best_features is None or features.shape[0] > best_features.shape[0]):
                    best_face = face
                    best_features = features

        if best_face is not None and best_features is not None:
            filename = secure_filename(f"{username}.npz")
            np.savez_compressed(os.path.join(FACE_DATA_DIR, filename),
                              face=best_face,
                              features=best_features)
            return True, "Face captured successfully"
        
        return False, "No suitable face detected"
    
    except Exception as e:
        return False, f"Error capturing face: {str(e)}"
    finally:
        if cap is not None:
            cap.release()

def recognize_face():
    """Recognize user's face and return username"""
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None, "Camera not accessible"

        best_match = 0
        best_match_user = None
        
        start_time = time.time()
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                continue
            
            current_face = preprocess_face(frame)
            if current_face is not None:
                _, current_features = compute_face_features(current_face)
                
                for user_file in os.listdir(FACE_DATA_DIR):
                    if not user_file.endswith('.npz'):
                        continue
                    
                    try:
                        data = np.load(os.path.join(FACE_DATA_DIR, user_file))
                        saved_features = data['features']
                        
                        match_score = match_faces(current_features, saved_features)
                        if match_score > best_match and match_score >= 10:
                            best_match = match_score
                            best_match_user = os.path.splitext(user_file)[0]
                    except Exception as e:
                        print(f"Error processing saved face {user_file}: {str(e)}")
                        continue

        if best_match_user:
            return best_match_user, "Login successful"
        return None, "Face not recognized"
    
    except Exception as e:
        return None, f"Error during recognition: {str(e)}"
    finally:
        if cap is not None:
            cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        if not username:
            flash('Username is required')
            return render_template('register.html')
        
        # Check if username already exists
        if any(file.startswith(secure_filename(username) + '.') for file in os.listdir(FACE_DATA_DIR)):
            flash('Username already exists')
            return render_template('register.html')
        
        success, message = capture_face(username)
        flash(message)
        if success:
            return redirect(url_for('index'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            ip = request.remote_addr
            if ip in login_attempts:
                attempts, timestamp = login_attempts[ip]
                if attempts >= MAX_LOGIN_ATTEMPTS:
                    if time.time() - timestamp < LOCK_TIME:
                        flash('Too many failed attempts. Please try again later.')
                        return render_template('login.html')
                    else:
                        login_attempts.pop(ip)

            user, message = recognize_face()
            
            if user:
                session['user'] = user
                session['last_activity'] = time.time()
                if ip in login_attempts:
                    login_attempts.pop(ip)
                return redirect(url_for('dashboard'))
            
            if ip in login_attempts:
                attempts, _ = login_attempts[ip]
                login_attempts[ip] = (attempts + 1, time.time())
            else:
                login_attempts[ip] = (1, time.time())
            
            flash(message)
            
        except Exception as e:
            flash(f"An error occurred during login: {str(e)}")
            
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        flash('Please login first')
        return redirect(url_for('login'))
    
    # Session timeout after 30 minutes
    if time.time() - session.get('last_activity', 0) > 1800:
        session.clear()
        flash('Session expired. Please login again.')
        return redirect(url_for('login'))
    
    session['last_activity'] = time.time()
    return render_template('dashboard.html', user=session['user'])

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully')
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host='0.0.0.0', port=port)