import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, redirect, session, url_for
from authlib.integrations.flask_client import OAuth
from functools import wraps
from urllib.parse import quote_plus, urlencode
import werkzeug.utils

# Import the facial recognition class
from facerec import FacialFeatureAnalyzer

load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

# Initialize the analyzer
analyzer = FacialFeatureAnalyzer()

# --- Auth0 setup ---
oauth = OAuth(app)
oauth.register(
    'auth0',
    client_id=os.environ.get("AUTH0_CLIENT_ID"),
    client_secret=os.environ.get("AUTH0_CLIENT_SECRET"),
    server_metadata_url=f'https://{os.environ.get("AUTH0_DOMAIN")}/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid profile email'}
)

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'profile' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/login')
def login():
    return oauth.auth0.authorize_redirect(
        redirect_uri=os.environ.get("AUTH0_CALLBACK_URL")
    )

@app.route('/callback')
def callback():
    print("Callback route reached!")
    try:
        token = oauth.auth0.authorize_access_token()
        print("Token fetched:", token)
        userinfo = oauth.auth0.userinfo(token=token)
        print("Userinfo fetched:", userinfo)
        session["profile"] = userinfo
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Authentication failed during callback: {e}")
        # Go to home (index) on error
        return redirect(url_for('home'))
    print("Callback completed, redirecting to upload_photo.")
    # FIX: Redirect to upload_photo as requested
    return redirect(url_for('upload_photo'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(
        f"https://{os.environ.get('AUTH0_DOMAIN')}/v2/logout?" +
        urlencode({
            "returnTo": os.environ.get("AUTH0_BASE_URL"),
            "client_id": os.environ.get("AUTH0_CLIENT_ID"),
        }, quote_via=quote_plus)
    )

# --- Firebase Config Helper Function ---
def get_firebase_config():
    """Builds and returns the Firebase config dictionary from environment variables."""
    return {
        "apiKey": os.environ.get("FIREBASE_API_KEY"),
        "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN"),
        "projectId": os.environ.get("FIREBASE_PROJECT_ID"),
        "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET"),
        "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID"),
        "appId": os.environ.get("FIREBASE_APP_ID"),
        "measurementId": os.environ.get("FIREBASE_MEASUREMENT_ID"),
    }

# MODIFIED: Pass Firebase config to all routes rendering 'index.html'
@app.route('/')
def home():
    user_data = session.get('profile')
    color_results = session.get('color_palette_results')
    # Pass Firebase config and App ID
    return render_template('index.html', user=user_data, color_results=color_results, 
                           firebase_config=get_firebase_config(), 
                           firebase_app_id=os.environ.get("FIREBASE_APP_ID"))

# MODIFIED: Pass Firebase config to all routes rendering 'index.html'
@app.route('/upload_photo', methods=['GET', 'POST'])
@requires_auth
def upload_photo():
    user_data = session.get('profile')
    color_results = session.get('color_palette_results')
    # Pass Firebase config and App ID
    return render_template('index.html', user=user_data, color_results=color_results,
                           firebase_config=get_firebase_config(),
                           firebase_app_id=os.environ.get("FIREBASE_APP_ID"))

# --- New Route for Face Analysis ---
@app.route('/analyze_face', methods=['POST'])
@requires_auth
def analyze_face():
    user_data = session.get('profile')
    file = request.files.get('file')
    
    if not file:
        return jsonify({"error": "No file part"}), 400

    # Ensure a directory for uploads exists (optional, but good practice)
    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)

    # Use a secure filename
    filename = werkzeug.utils.secure_filename(file.filename)
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)
    
    try:
        # 1. Process the image with the facial analyzer
        results = analyzer.analyze_face(filepath)
        
        if 'error' in results:
            os.remove(filepath)
            return jsonify({"error": results['error']}), 400
        
        # 2. Store the result in the session (simulating a database save per user)
        session['color_palette_results'] = results
        
        # 3. Clean up the uploaded file after analysis
        os.remove(filepath)
        
        print(f"Analysis complete for {user_data.get('email', 'N/A')}: {results['features']}")
        
        return jsonify({
            "success": True, 
            "message": "Facial analysis complete!",
            "results": results
        })

    except ValueError as ve:
        # Handle errors from the analyzer (e.g., file not loaded)
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"An unexpected error occurred during analysis: {e}"}), 500

if __name__ == '__main__':
    # Create the 'uploads' directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5000)