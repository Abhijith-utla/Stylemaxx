# filename: app.py (UPDATED)
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, redirect, session, url_for
from authlib.integrations.flask_client import OAuth
from functools import wraps
from urllib.parse import quote_plus, urlencode
import werkzeug.utils
import json # Import json for loading data
from flask import send_from_directory

# Import the facial recognition class
from facerec import FacialFeatureAnalyzer

# Import the new outfit matcher utility
from outfit_matcher import dynamic_outfit_match # <--- NEW IMPORT


load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

# Initialize the analyzer
analyzer = FacialFeatureAnalyzer()

# --- Auth0 setup (UNCHANGED) ---
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

# --- Firebase Config Helper Function (UNCHANGED) ---
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

# MODIFIED: Pass Firebase config to all routes rendering 'index.html' (UNCHANGED)
@app.route('/')
def home():
    user_data = session.get('profile')
    color_results = session.get('color_palette_results')
    # Pass Firebase config and App ID
    # Pass the real outfit results from session if available
    outfit_results = session.get('outfit_results') # <--- NEW: Get results
    return render_template('index.html', user=user_data, color_results=color_results, 
                           firebase_config=get_firebase_config(), 
                           firebase_app_id=os.environ.get("FIREBASE_APP_ID"),
                           outfit_results=outfit_results) # <--- NEW: Pass results

# MODIFIED: Pass Firebase config to all routes rendering 'index.html' (UNCHANGED)
@app.route('/upload_photo', methods=['GET', 'POST'])
@requires_auth
def upload_photo():
    user_data = session.get('profile')
    color_results = session.get('color_palette_results')
    outfit_results = session.get('outfit_results') # <--- NEW: Get results
    # Pass Firebase config and App ID
    return render_template('index.html', user=user_data, color_results=color_results,
                           firebase_config=get_firebase_config(),
                           firebase_app_id=os.environ.get("FIREBASE_APP_ID"),
                           outfit_results=outfit_results) # <--- NEW: Pass results

# --- Route for Face Analysis (UNCHANGED) ---
@app.route('/analyze_face', methods=['POST'])
@requires_auth
def analyze_face():
    # ... (unchanged logic for facial analysis) ...
    user_data = session.get('profile')
    file = request.files.get('file')
    
    if not file:
        return jsonify({"error": "No file part"}), 400

    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)

    filename = werkzeug.utils.secure_filename(file.filename)
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)
    
    try:
        results = analyzer.analyze_face(filepath)
        
        if 'error' in results:
            os.remove(filepath)
            return jsonify({"error": results['error']}), 400
        
        session['color_palette_results'] = results
        
        os.remove(filepath)
        
        print(f"Analysis complete for {user_data.get('email', 'N/A')}: {results['features']}")
        
        return jsonify({
            "success": True, 
            "message": "Facial analysis complete!",
            "results": results
        })

    except ValueError as ve:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"An unexpected error occurred during analysis: {e}"}), 500

# --- NEW Route for Outfit Generation ---
@app.route('/chef_up', methods=['POST'])
@requires_auth
def chef_up():
    data = request.get_json()
    item_image_url = data.get('imageUrl')
    item_type = data.get('itemType')
    item_id = data.get('itemId')

    if not item_image_url or not item_type:
        return jsonify({"error": "Missing item image URL or type."}), 400

    try:
        # Call the new ML function
        suggested_outfits = dynamic_outfit_match(item_image_url, item_type)

        # Check for error in the returned list
        if suggested_outfits and suggested_outfits[0].startswith('error:'):
            error_message = suggested_outfits[0].replace('error: ', '')
            return jsonify({"error": f"Outfit generation failed: {error_message}"}), 500
        
        # Store results in session
        # Storing the image paths and the original item ID
        session['outfit_results'] = {
            'original_item_id': item_id,
            'original_item_type': item_type,
            'suggested_outfits': suggested_outfits # List of 5 image paths
        }

        print(f"Outfit suggestions generated for item {item_id}. Suggested paths count: {len(suggested_outfits)}")

        return jsonify({
            "success": True, 
            "message": "Outfits cheffed up!",
            "results": suggested_outfits
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"An unexpected server error occurred during outfit generation: {e}"}), 500

# --- NEW: Configure a static route for the apparel dataset images ---
# This exposes the 'apparel_dataset' folder content under the URL prefix '/static/apparel/'
@app.route('/apparel_static/<path:filename>')
def serve_apparel_static(filename):
    # This uses Flask's built-in mechanism to serve files from a specified directory
    # It assumes the 'apparel_dataset' folder is in the root directory alongside app.py
    root_dir = os.path.join(app.root_path, 'apparel_dataset')
    return send_from_directory(root_dir, filename)

if __name__ == '__main__':
    # Create the 'uploads' directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5000)