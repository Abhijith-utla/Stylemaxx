import cv2
import numpy as np
from collections import Counter
import face_recognition
from sklearn.cluster import KMeans

class FacialFeatureAnalyzer:
    def __init__(self):
        self.skin_tone_categories = {
            'Very Fair': [(255, 224, 189), (255, 239, 213)],
            'Fair': [(241, 194, 125), (255, 219, 172)],
            'Medium': [(224, 172, 105), (241, 194, 125)],
            'Olive': [(198, 134, 66), (224, 172, 105)],
            'Tan': [(141, 85, 36), (198, 134, 66)],
            'Brown': [(92, 51, 23), (141, 85, 36)],
            'Dark Brown': (58, 29, 15)
        }
    
    def load_image(self, image_path):
        """Load image from path"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image
    
    def get_dominant_colors(self, image_region, n_colors=3):
        """Extract dominant colors from an image region using K-means clustering"""
        pixels = image_region.reshape(-1, 3)
        pixels = pixels[~np.all(pixels == 0, axis=1)]  # Remove black pixels
        
        if len(pixels) == 0:
            return []
        
        kmeans = KMeans(n_clusters=min(n_colors, len(pixels)), random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts = Counter(labels)
        
        # Sort colors by frequency
        sorted_colors = [colors[i] for i in sorted(counts, key=counts.get, reverse=True)]
        return [(int(b), int(g), int(r)) for b, g, r in sorted_colors]
    
    def rgb_to_hex(self, rgb):
        """Convert RGB to HEX color code"""
        return '#{:02x}{:02x}{:02x}'.format(rgb[2], rgb[1], rgb[0])
    
    def classify_color(self, bgr_color, color_ranges):
        """Classify a color based on predefined ranges"""
        b, g, r = bgr_color
        
        for name, ranges in color_ranges.items():
            if isinstance(ranges, tuple) and len(ranges) == 3:
                continue
            for color_range in ranges if isinstance(ranges[0], list) else [ranges]:
                if isinstance(color_range, tuple):
                    rb, gb, bb = color_range
                    if abs(r - rb) < 40 and abs(g - gb) < 40 and abs(b - bb) < 40:
                        return name
        return "Unknown"
    
    def analyze_hair_color(self, dominant_colors):
        """Classify hair color based on dominant colors"""
        hair_colors = {
            'Black': [(0, 0, 0), (50, 50, 50)],
            'Dark Brown': [(40, 25, 15), (90, 60, 40)],
            'Brown': [(90, 60, 40), (160, 110, 70)],
            'Light Brown': [(160, 110, 70), (200, 150, 100)],
            'Blonde': [(200, 180, 140), (255, 240, 200)],
            'Red': [(100, 40, 20), (180, 80, 40)],
            'Auburn': [(120, 60, 30), (160, 90, 50)],
            'Grey': [(150, 150, 150), (200, 200, 200)],
            'White': [(200, 200, 200), (255, 255, 255)]
        }
        
        if not dominant_colors:
            return "Unknown"
        
        primary_color = dominant_colors[0]
        b, g, r = primary_color
        
        # Classify based on brightness and color ratios
        brightness = (r + g + b) / 3
        
        if brightness < 60:
            return "Black"
        elif brightness > 200 and abs(r - g) < 20 and abs(g - b) < 20:
            return "Grey/White"
        elif r > g and r > b and (r - g) > 30:
            return "Red/Auburn"
        elif brightness > 160:
            return "Blonde"
        elif brightness > 120:
            return "Light Brown"
        elif brightness > 80:
            return "Brown"
        else:
            return "Dark Brown"
    
    def analyze_eye_color(self, dominant_colors):
        """Classify eye color based on dominant colors"""
        if not dominant_colors:
            return "Unknown"
        
        b, g, r = dominant_colors[0]
        
        # Enhanced eye color classification
        if b > 140 and g > 100 and r < 100:
            return "Blue"
        elif b > 100 and g > 120 and b > r:
            return "Blue-Green"
        elif g > 120 and g > r and g > b:
            return "Green"
        elif g > 100 and r > 100 and (g - b) > 30:
            return "Hazel"
        elif r > 100 and g > 80 and b < 80:
            return "Brown"
        elif r > 120 and g > 100 and b > 80:
            return "Amber"
        elif r < 80 and g < 80 and b < 80:
            return "Dark Brown/Black"
        else:
            return "Brown"
    
    def analyze_skin_tone(self, dominant_colors):
        """Classify skin tone based on dominant colors"""
        if not dominant_colors:
            return "Unknown"
        
        b, g, r = dominant_colors[0]
        brightness = (r + g + b) / 3
        
        if brightness > 220:
            return "Very Fair"
        elif brightness > 190:
            return "Fair"
        elif brightness > 160:
            return "Medium"
        elif brightness > 130:
            return "Olive/Tan"
        elif brightness > 100:
            return "Brown"
        else:
            return "Dark Brown"
    
    def analyze_face(self, image_path):
        """Complete facial analysis including eyes, hair, skin tone, and color palette"""
        image = self.load_image(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face landmarks
        face_landmarks_list = face_recognition.face_landmarks(rgb_image)
        
        if not face_landmarks_list:
            return {"error": "No face detected in the image"}
        
        face_landmarks = face_landmarks_list[0]
        
        results = {
            'features': {},
            'dominant_colors': {},
            'color_palette': {},
            'recommendations': {}
        }
        
        # Analyze eyes
        if 'left_eye' in face_landmarks and 'right_eye' in face_landmarks:
            left_eye_points = np.array(face_landmarks['left_eye'])
            right_eye_points = np.array(face_landmarks['right_eye'])
            
            # Create masks for eyes
            eye_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(eye_mask, [left_eye_points], 255)
            cv2.fillPoly(eye_mask, [right_eye_points], 255)
            
            # Dilate to capture iris better
            kernel = np.ones((5, 5), np.uint8)
            eye_mask = cv2.dilate(eye_mask, kernel, iterations=1)
            
            eye_region = cv2.bitwise_and(image, image, mask=eye_mask)
            eye_colors = self.get_dominant_colors(eye_region, n_colors=3)
            
            results['features']['eye_color'] = self.analyze_eye_color(eye_colors)
            results['dominant_colors']['eyes'] = [self.rgb_to_hex(c) for c in eye_colors]
        
        # Analyze skin tone
        if 'chin' in face_landmarks and 'nose_bridge' in face_landmarks:
            face_points = (face_landmarks['chin'] + 
                          face_landmarks['nose_bridge'] + 
                          face_landmarks.get('left_cheek', []) +
                          face_landmarks.get('right_cheek', []))
            
            if face_points:
                face_points = np.array(face_points)
                skin_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                hull = cv2.convexHull(face_points)
                cv2.fillConvexPoly(skin_mask, hull, 255)
                
                skin_region = cv2.bitwise_and(image, image, mask=skin_mask)
                skin_colors = self.get_dominant_colors(skin_region, n_colors=3)
                
                results['features']['skin_tone'] = self.analyze_skin_tone(skin_colors)
                results['dominant_colors']['skin'] = [self.rgb_to_hex(c) for c in skin_colors]
        
        # Analyze hair (top portion of image above face)
        face_locations = face_recognition.face_locations(rgb_image)
        if face_locations:
            top, right, bottom, left = face_locations[0]
            hair_region = image[max(0, top - 100):top, left:right]
            
            if hair_region.size > 0:
                hair_colors = self.get_dominant_colors(hair_region, n_colors=3)
                results['features']['hair_color'] = self.analyze_hair_color(hair_colors)
                results['dominant_colors']['hair'] = [self.rgb_to_hex(c) for c in hair_colors]
        
        # Generate color palette recommendations
        results['color_palette'] = self.generate_color_palette(results['features'])
        
        return results
    
    def generate_color_palette(self, features):
        """Generate complementary color palette based on features"""
        palette = {
            'complementary_colors': [],
            'avoid_colors': [],
            'season': '',
            'best_metals': []
        }
        
        skin_tone = features.get('skin_tone', '')
        eye_color = features.get('eye_color', '')
        hair_color = features.get('hair_color', '')
        
        # Seasonal color analysis
        if 'Fair' in skin_tone or 'Very Fair' in skin_tone:
            if 'Blonde' in hair_color or 'Light' in hair_color:
                palette['season'] = 'Spring/Summer'
                palette['complementary_colors'] = ['Pastels', 'Soft Blues', 'Peach', 'Coral', 'Light Pink']
                palette['avoid_colors'] = ['Black', 'Very Dark Colors', 'Neon Colors']
                palette['best_metals'] = ['Gold', 'Rose Gold']
            else:
                palette['season'] = 'Winter'
                palette['complementary_colors'] = ['Bold Colors', 'Royal Blue', 'Emerald', 'Pure White', 'Black']
                palette['avoid_colors'] = ['Warm Browns', 'Orange', 'Rust']
                palette['best_metals'] = ['Silver', 'White Gold', 'Platinum']
        
        elif 'Medium' in skin_tone or 'Olive' in skin_tone:
            palette['season'] = 'Autumn/Spring'
            palette['complementary_colors'] = ['Warm Earth Tones', 'Olive Green', 'Rust', 'Gold', 'Coral']
            palette['avoid_colors'] = ['Icy Colors', 'Pale Pastels']
            palette['best_metals'] = ['Gold', 'Bronze', 'Copper']
        
        else:  # Tan, Brown, Dark Brown
            palette['season'] = 'All Seasons'
            palette['complementary_colors'] = ['Rich Jewel Tones', 'Deep Purple', 'Emerald', 'Burgundy', 'Gold']
            palette['avoid_colors'] = ['Pale Colors', 'Washed Out Tones']
            palette['best_metals'] = ['Gold', 'Rose Gold', 'Bronze']
        
        # Eye color enhancements
        if 'Blue' in eye_color:
            palette['eye_enhancing_colors'] = ['Orange', 'Copper', 'Warm Browns', 'Gold']
        elif 'Green' in eye_color:
            palette['eye_enhancing_colors'] = ['Purple', 'Burgundy', 'Plum', 'Rose']
        elif 'Brown' in eye_color or 'Hazel' in eye_color:
            palette['eye_enhancing_colors'] = ['Green', 'Purple', 'Blue', 'Gold']
        
        return palette
    
    def print_analysis(self, results):
        """Print formatted analysis results"""
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        print("\n" + "="*60)
        print("FACIAL FEATURE ANALYSIS")
        print("="*60)
        
        print("\n--- DETECTED FEATURES ---")
        for feature, value in results['features'].items():
            print(f"{feature.replace('_', ' ').title()}: {value}")
        
        print("\n--- DOMINANT COLORS (HEX) ---")
        for region, colors in results['dominant_colors'].items():
            print(f"\n{region.title()}:")
            for i, color in enumerate(colors, 1):
                print(f"  Color {i}: {color}")
        
        print("\n--- COLOR PALETTE RECOMMENDATIONS ---")
        palette = results['color_palette']
        print(f"\nSeason Type: {palette.get('season', 'N/A')}")
        print(f"\nBest Metals: {', '.join(palette.get('best_metals', []))}")
        
        print("\nComplementary Colors:")
        for color in palette.get('complementary_colors', []):
            print(f"  • {color}")
        
        if 'eye_enhancing_colors' in palette:
            print("\nEye-Enhancing Colors:")
            for color in palette.get('eye_enhancing_colors', []):
                print(f"  • {color}")
        
        print("\nColors to Avoid:")
        for color in palette.get('avoid_colors', []):
            print(f"  • {color}")
        
        print("\n" + "="*60)


# Example usage
if __name__ == "__main__":
    analyzer = FacialFeatureAnalyzer()
    
    # Analyze a face image
    image_path = "stylemax/image.png"  # Replace with your image path
    
    try:
        results = analyzer.analyze_face(image_path)
        analyzer.print_analysis(results)
        
        # You can also access specific results
        # print(f"\nEye Color: {results['features']['eye_color']}")
        # print(f"Hair Color: {results['features']['hair_color']}")
        # print(f"Skin Tone: {results['features']['skin_tone']}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Replace 'person_photo.jpg' with your actual image path")
        print("2. Ensure the image contains a clear, front-facing face")
        print("3. Have good lighting in the photo for accurate color detection")