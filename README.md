# 👗 StyleMaxx – AI-Powered Outfit Recommendation Platform

StyleMaxx is a **full-stack AI web application** that blends fashion and technology to create a **personalized digital styling experience**.

The app allows users to:

* **Upload outfit images** (tops, bottoms, or full looks).
* **Authenticate securely** with OAuth (Google, GitHub, or Email).
* **Discover recommendations** powered by deep learning models that analyze color, texture, and clothing structure.

Unlike generic recommendation engines, StyleMaxx focuses on **individual style discovery**. It doesn’t just suggest “trending” items—it learns from each upload to suggest complementary looks, helping users mix, match, and expand their wardrobe intelligently.

---

## 🌟 Key Features

* **Personal Stylist in Your Browser** → Upload an image and instantly get curated style recommendations.
* **Smart Feature Extraction** → Uses ResNet50 to analyze visual attributes of outfits.
* **Style Clustering** → Groups similar looks using KNN + face recognition encodings for better personalization.
* **User Profiles** → Firebase securely manages logins and keeps track of each user’s style journey.
* **Scalable Data Handling** → Outfit images stored in Firebase Storage, metadata in Firestore, ML models running in Flask.

---

## 🚀 Tech Stack

### 🔹 Frontend – React

* **Interactive UI** built with React.js.
* Integrates with **Firebase Auth SDK** for seamless login.
* Displays recommendations from Flask backend in real time.

### 🔹 Backend – Flask + ML Models

* Exposes REST API endpoints for outfit processing.
* ML pipeline includes:

  * **Face recognition & clustering** (skin tone, dominant features).
  * **Outfit segmentation & deep feature extraction** with ResNet50.
  * **KNN similarity search** for intelligent outfit recommendations.

### 🔹 Firebase (Auth, Database, Storage)

* **Authentication (OAuth)** → Google, GitHub, Email/Password.
* **Firestore Database** → Saves user preferences & outfit metadata.
* **Storage** → Stores uploaded images, linked to user IDs.

### 🔹 OAuth Integration

* Users sign in with Google/GitHub/Email.
* Firebase issues **JWT tokens**.
* React forwards tokens to Flask API.
* Flask verifies them with Firebase Admin SDK before serving responses.

---

## 🔄 System Flow

1. **👤 User → React Frontend**

   * Uploads an outfit image.
   * Logs in with OAuth (Google/GitHub/Email).

2. **⚡ React → Firebase**

   * Handles login & token issuance.
   * Stores the uploaded image in Firebase Storage.

3. **🔗 React → Flask API**

   * Sends requests (with JWT) to the backend for ML inference.

4. **🧠 Flask → ML Models**

   * Extracts outfit features.
   * Finds similar looks with clustering + KNN.

5. **🗄️ Flask ↔ Firebase**

   * Stores metadata and references in Firestore.

6. **🎨 React Frontend**

   * Displays curated recommendations back to the user.

---

## 📦 Installation & Setup

### 1. Frontend (React)

```bash
cd frontend
npm install
npm start
```

### 2. Backend (Flask)

```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 3. Firebase Setup

* Create a Firebase Project.
* Enable:

  * **Authentication** (Google, GitHub, Email/Password)
  * **Firestore Database**
  * **Firebase Storage**
* Download your Firebase config and add it to the React app.

---

✨ With StyleMax, every user gets more than suggestions—they get **their own AI stylist** that evolves with their wardrobe.
