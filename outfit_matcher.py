# filename: outfit_matcher.py
import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib
from urllib.request import urlopen, Request

# ----------------------------------------
# ORB Feature Extractor for tops/bottoms
# ----------------------------------------
def extract_orb_features(image_input):
    """
    Extracts ORB features from an image.
    :param image_input: File path or a file-like object (numpy array from decoding base64/URL).
    :return: Mean of ORB descriptors or None.
    """
    if isinstance(image_input, str) and image_input.startswith(('http', 'data:image')):
        # Handle Base64 (from frontend mock) or URL (real storage)
        try:
            # Simple handling for web/base64-like URLs
            if image_input.startswith('data:image'):
                import base64
                import re
                base64_str = re.sub('^data:image/.+;base64,', '', image_input)
                img_bytes = base64.b64decode(base64_str)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else: # Assuming it's a direct URL for a real-world scenario
                req = Request(image_input, headers={'User-Agent': 'Mozilla/5.0'})
                img_bytes = urlopen(req).read()
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error reading image from URL/Base64: {e}")
            return None
    elif isinstance(image_input, str):
        # Handle local file path
        img = cv2.imread(image_input, cv2.IMREAD_COLOR)
    else:
        print("Invalid image input type.")
        return None

    if img is None:
        return None
        
    img = cv2.resize(img, (224, 224))
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(32)
        
    return np.mean(descriptors, axis=0)


# ----------------------------------------
# Load ORB dataset (tops + bottoms)
# ----------------------------------------
# NOTE: In a real deployment, load this once globally in app.py or a cache.
# For simplicity, we keep it here for now.
def load_dataset(dataset_folder):
    data = []
    # Simplified loading to assume all ORB features are pre-calculated and stored.
    # The original logic of recalculating ORB on every call is very slow.
    # We will assume a simplified pre-calculated dataset structure for this integration.
    # For now, we keep the original slow logic to match the provided file structure.
    for pair_folder in os.listdir(dataset_folder):
        pair_path = os.path.join(dataset_folder, pair_folder)
        if not os.path.isdir(pair_path):
            continue
        top_path = os.path.join(pair_path, "top.jpg")
        bottom_path = os.path.join(pair_path, "bottom.jpg")
        if os.path.exists(top_path) and os.path.exists(bottom_path):
            top_feat = extract_orb_features(top_path)
            bottom_feat = extract_orb_features(bottom_path)
            if top_feat is not None and bottom_feat is not None:
                data.append({
                    "top_feat": top_feat,
                    "bottom_feat": bottom_feat,
                    "top_path": top_path,
                    "bottom_path": bottom_path
                })
    return data

# ----------------------------------------
# Find best matching item in dataset
# ----------------------------------------
def find_best_match(user_feat, dataset, target_feat_key):
    if user_feat is None:
        return None, None
    sims = [cosine_similarity(user_feat.reshape(1, -1),
                              entry[target_feat_key].reshape(1, -1))[0][0] for entry in dataset]
    best_idx = np.argmax(sims)
    return dataset[best_idx]["top_path"], dataset[best_idx]["bottom_path"]

# ----------------------------------------
# ResNet feature extractor for KNN
# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and freeze ResNet
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

def extract_resnet_features(image_input):
    """
    Extracts ResNet features from an image input (file path or base64).
    """
    img = None
    if isinstance(image_input, str) and image_input.startswith(('http', 'data:image')):
        # Handle Base64 (from frontend mock) or URL (real storage)
        try:
            if image_input.startswith('data:image'):
                import base64
                import re
                base64_str = re.sub('^data:image/.+;base64,', '', image_input)
                img_bytes = base64.b64decode(base64_str)
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            else: # Assuming it's a direct URL
                import io
                req = Request(image_input, headers={'User-Agent': 'Mozilla/5.0'})
                img_bytes = urlopen(req).read()
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            print(f"Error reading image from URL/Base64 for ResNet: {e}")
            return None
    elif isinstance(image_input, str):
        # Handle local file path
        img = Image.open(image_input).convert('RGB')

    if img is None:
        return None
        
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(img_t)
    return feat.cpu().numpy().flatten()


def build_files(folder_path):
    files = []
    # NOTE: The original logic will traverse your entire apparel_dataset folder.
    # For a real app, this list should be pre-calculated and cached.
    for root, _, filenames in os.walk(folder_path):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append(os.path.join(root, f))
    return files

def find_top5_neighbors(knn_model_path, suggested_item_img_path, folder_path, n=5):
    """
    Finds n-nearest neighbors (similar items) using the pre-trained KNN model.
    """
    try:
        # Load the pre-trained KNN model
        knn_model = joblib.load(knn_model_path)
    except FileNotFoundError:
        return [f"ERROR: KNN model not found at {knn_model_path}"]
    except Exception as e:
        return [f"ERROR: Failed to load KNN model: {e}"]
        
    try:
        # Build the list of files that the KNN index corresponds to
        files = build_files(folder_path)
        
        # Extract features from the suggested item (which is a file path in the dataset)
        feat = extract_resnet_features(suggested_item_img_path)
        
        # Find k+1 neighbors (k=5, plus the item itself which is index 0)
        distances, indices = knn_model.kneighbors([feat], n_neighbors=n+1)
        
        # Exclude the first one (the item itself) and get the paths
        neighbors = [files[i] for i in indices[0][1:]]
        return neighbors
        
    except Exception as e:
        return [f"ERROR: KNN search failed: {e}"]


# ----------------------------------------
# Dynamic outfit matching main function
# ----------------------------------------
def dynamic_outfit_match(user_img_url, user_item_type):
    """
    Generates a list of suggested outfit items based on the user's uploaded item.
    
    :param user_img_url: The image URL (or base64 string) of the user's item.
    :param user_item_type: The type of the user's item ('top' or 'bottom').
    :return: A list of suggested outfit paths.
    """
    
    # NOTE: We hardcode gender here as it's not provided by the frontend.
    # A real app would get this from user profile data.
    gender = "men" 
    gender_key = "men" if gender.lower() == "men" else "women"
    dataset_folder = "dataset"
    
    # NOTE: Load the dataset once in the Flask app startup and pass it in for better performance.
    dataset = load_dataset(dataset_folder)
    
    # 1. Extract features from the user's uploaded image
    user_feat = extract_orb_features(user_img_url)
    if user_feat is None:
        return ["error: Could not extract features from user image."]
        
    # 2. Find the best matching pre-paired outfit in the dataset
    if user_item_type.lower() == "top":
        target_feat_key = "top_feat"
        matched_top_path, suggested_bottom_path = find_best_match(user_feat, dataset, target_feat_key)
        suggested_item_path = suggested_bottom_path
        
        # 3. Find 5 similar items (bottoms) using KNN
        knn_model_path = f"{gender_key}_bottoms_knn.joblib"
        folder_path = f"apparel_dataset/{gender_key}/bottoms"
        suggested_outfits = find_top5_neighbors(knn_model_path, suggested_item_path, folder_path, n=5)

    elif user_item_type.lower() == "bottom":
        target_feat_key = "bottom_feat"
        matched_top_path, suggested_bottom_path = find_best_match(user_feat, dataset, target_feat_key)
        suggested_item_path = matched_top_path # Note: matched_top_path is the suggested top path
        
        # 3. Find 5 similar items (tops) using KNN
        knn_model_path = f"{gender_key}_tops_knn.joblib"
        folder_path = f"apparel_dataset/{gender_key}/tops"
        suggested_outfits = find_top5_neighbors(knn_model_path, suggested_item_path, folder_path, n=5)
        
    else:
        return ["error: Invalid clothing type."]

    # 4. Return the list of 5 suggested image paths
    return suggested_outfits