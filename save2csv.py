import os
import cv2
import numpy as np
import csv
from sklearn.cluster import KMeans
import mediapipe as mp
import colorsys
from PIL import Image


# Custom colors mapping
custom_colors = {
    'Navy': (0, 0, 128),
    'Blue': (0, 0, 255),
    'Silver': (192, 192, 192),
    'Black': (0, 0, 0),
    'Grey': (128, 128, 128),
    'Green': (0, 128, 0),
    'Purple': (128, 0, 128),
    'White': (255, 255, 255),
    'Beige': (245, 245, 220),
    'Brown': (165, 42, 42),
    'Bronze': (205, 127, 50),
    'Teal': (0, 128, 128),
    'Copper': (184, 115, 51),
    'Pink': (255, 192, 203),
    'Off White': (255, 250, 240),
    'Maroon': (128, 0, 0),
    'Red': (255, 0, 0),
    'Khaki': (195, 176, 145),
    'Orange': (255, 165, 0),
    'Coffee Brown': (165, 42, 42),
    'Yellow': (255, 255, 0),
    'Charcoal': (54, 69, 79),
    'Gold': (255, 215, 0),
    'Steel': (70, 130, 180),
    'Tan': (210, 180, 140),
    'Multi': (255, 0, 255),
    'Magenta': (255, 0, 255),
    'Lavender': (230, 230, 250),
    'Sea Green': (46, 139, 87),
    'Cream': (255, 253, 208),
    'Peach': (255, 218, 185),
    'Olive': (128, 128, 0),
    'Skin': (255, 228, 181),
    'Burgundy': (128, 0, 32),
    'Grey Melange': (169, 169, 169),
    'Rust': (183, 65, 14),
    'Rose': (255, 0, 127),
    'Lime Green': (50, 205, 50),
    'Mauve': (224, 176, 255),
    'Turquoise Blue': (64, 224, 208),
    'Metallic': (211, 211, 211),
    'Mustard': (255, 255, 0),
    'Taupe': (139, 133, 137),
    'Nude': (222, 184, 135),
    'Mushroom Brown': (165, 42, 42),
    'Fluorescent Green': (0, 255, 0),
}

color_ranges = {
    "Bright Spring": [(255, 255, 255), (255, 0, 150), (255, 200, 200)],  # Bright whites, pinks, and light reds
    "True Spring": [(255, 200, 200), (255, 255, 255), (255, 255, 0)],  # Warm pinks, whites, and yellows
    "Light Spring": [(255, 255, 0), (255, 200, 200), (255, 150, 100)],  # Warm yellows, pinks, and light oranges
    "Light Summer": [(150, 150, 200), (100, 100, 150), (75, 75, 125)],  # Light lavenders, cool grays
    "True Summer": [(100, 100, 150), (75, 75, 100), (50, 50, 75)],  # Soft grays, lavenders
    "Soft Summer": [(50, 50, 75), (75, 75, 100), (100, 100, 150)],  # Cool grays, soft blues, lavenders
    "Soft Autumn": [(100, 50, 0), (150, 100, 50), (200, 150, 100)],  # Soft browns, oranges, and beiges
    "True Autumn": [(200, 150, 100), (150, 100, 50), (100, 50, 0)],  # Warm oranges, browns, and dark yellows
    "Dark Autumn": [(100, 25, 0), (100, 50, 0), (50, 25, 0)],  # Dark browns, burnt oranges
    "Dark Winter": [(0, 0, 100), (0, 50, 150), (150, 150, 200)],  # Cool blues, icy tones
    "True Winter": [(0, 50, 150), (0, 0, 100), (0, 0, 50)],  # Cool blues, purples, and dark grays
    "Bright Winter": [(255, 255, 255), (200, 0, 0), (150, 0, 0)],  # Bright whites, bright reds, dark reds
}

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation


# Function to apply selfie segmentation
def apply_selfie_segmentation(image, model_selection):
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=model_selection) as selfie_segmentation:
        results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (192, 192, 192)  # Default background color
        output_image = np.where(condition, image, bg_image)
        return output_image


# Function to detect faces in an image
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

# Function to crop the image to the first detected face
def crop_to_face(image, faces):
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image

# Function to detect dominant colors in an image and optionally lip color
def detect_colors(image, num_colors=20, detect_lip_color=False):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure image is in RGB format for KMeans
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(img)
    colors = kmeans.cluster_centers_

    # Filter out gray colors
    filtered_colors = []
    for color in colors:
        r, g, b = color
        if not (r == g == b or abs(r - g) < 15 and abs(r - b) < 15 and abs(g - b) < 15):
            filtered_colors.append(tuple(map(int, color)))

    # Optionally detect lip color (you may need to fine-tune the ROI for lips)
    if detect_lip_color:
        # Define ROI for lips
        # Example coordinates; adjust based on your specific use case
        lip_roi = image[150:250, 100:300]  # Example ROI coordinates

        # Convert ROI to RGB
        lip_roi_rgb = cv2.cvtColor(lip_roi, cv2.COLOR_BGR2RGB)
        lip_roi_rgb = lip_roi_rgb.reshape((lip_roi_rgb.shape[0] * lip_roi_rgb.shape[1], 3))

        # Apply KMeans clustering to detect dominant colors in lip ROI
        kmeans_lip = KMeans(n_clusters=1)
        kmeans_lip.fit(lip_roi_rgb)
        lip_color = tuple(map(int, kmeans_lip.cluster_centers_[0]))
        filtered_colors.append(lip_color)

    # Ensure exactly 20 colors are returned
    if len(filtered_colors) < 20:
        num_additional_colors = 20 - len(filtered_colors)
        additional_colors = np.random.randint(0, 256, size=(num_additional_colors, 3))
        filtered_colors.extend(map(tuple, additional_colors))

    return filtered_colors[:20]

# Function to convert RGB to HSL and return hue, saturation, and lightness
def rgb_to_hsl(r, g, b):
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    h = h * 360  # Convert hue to degrees
    s = s * 100  # Convert saturation to percentage
    l = l * 100  # Convert lightness to percentage
    return h, s, l


# Function to determine temperature from hues
def determine_temperature(hues):
    warm_hues = sum(1 for h in hues if 0 <= h <= 60 or 300 <= h <= 360)
    cool_hues = len(hues) - warm_hues
    return "Warm" if warm_hues >= cool_hues else "Cool"


# Function to determine overall depth
def determine_depth(lightness_values):
    average_lightness = sum(lightness_values) / len(lightness_values)
    if average_lightness < 20:
        return "Very Dark"
    elif 20 <= average_lightness < 40:
        return "Dark"
    elif 40 <= average_lightness < 60:
        return "Medium"
    elif 60 <= average_lightness < 80:
        return "Light"
    else:
        return "Very Light"

# Function to determine overall chroma
def determine_chroma(saturation_values):
    average_saturation = sum(saturation_values) / len(saturation_values)
    if average_saturation < 20:
        return "Low"
    elif 20 <= average_saturation < 40:
        return "Medium"
    else:
        return "High"


# Function to map colors to 12-season color theory
def map_to_season(hues, lightness_values, saturation_values):
    seasons = {
        'Bright Spring': 0,
        'True Spring': 0,
        'Light Spring': 0,
        'Light Summer': 0,
        'True Summer': 0,
        'Soft Summer': 0,
        'Soft Autumn': 0,
        'True Autumn': 0,
        'Deep Autumn': 0,
        'Deep Winter': 0,
        'True Winter': 0,
        'Bright Winter': 0
    }
    for h, l, s in zip(hues, lightness_values, saturation_values):
        # Determine the season based on hue, lightness, and saturation
        if 0 <= h < 45 or 330 <= h <= 360:  # Warm hues
            if s > 50 and l > 50:
                seasons['Bright Spring'] += 1
            elif s > 50 and l <= 50:
                seasons['True Spring'] += 1
            elif s <= 50 and l > 50:
                seasons['Light Spring'] += 1
        elif 45 <= h < 170:  # Cool hues
            if s <= 50 and l > 50:
                seasons['Light Summer'] += 1
            elif s <= 50 and l <= 50:
                seasons['True Summer'] += 1
            elif s > 50 and l <= 50:
                seasons['Soft Summer'] += 1
        elif 170 <= h < 260:  # Muted hues
            if s <= 50 and l <= 50:
                seasons['Soft Autumn'] += 1
            elif s > 50 and l <= 50:
                seasons['True Autumn'] += 1
            elif s > 50 and l > 50:
                seasons['Deep Autumn'] += 1
        elif 260 <= h < 330:  # Cool hues
            if s > 50 and l <= 50:
                seasons['Deep Winter'] += 1
            elif s > 50 and l > 50:
                seasons['True Winter'] += 1
            elif s <= 50 and l > 50:
                seasons['Bright Winter'] += 1
    # Determine the predominant season
    predominant_season = max(seasons, key=seasons.get)
    return predominant_season

# Function to recommend colors based on the season
def recommend_colors(season):
    color_ranges = {
        "Bright Spring": [(255, 255, 255), (255, 0, 150), (255, 200, 200)],  # Bright whites, pinks, and light reds
        "True Spring": [(255, 200, 200), (255, 255, 255), (255, 255, 0)],  # Warm pinks, whites, and yellows
        "Light Spring": [(255, 255, 0), (255, 200, 200), (255, 150, 100)],  # Warm yellows, pinks, and light oranges
        "Light Summer": [(150, 150, 200), (100, 100, 150), (75, 75, 125)],  # Light lavenders, cool grays
        "True Summer": [(100, 100, 150), (75, 75, 100), (50, 50, 75)],  # Soft grays, lavenders
        "Soft Summer": [(50, 50, 75), (75, 75, 100), (100, 100, 150)],  # Cool grays, soft blues, lavenders
        "Soft Autumn": [(100, 50, 0), (150, 100, 50), (200, 150, 100)],  # Soft browns, oranges, and beiges
        "True Autumn": [(200, 150, 100), (150, 100, 50), (100, 50, 0)],  # Warm oranges, browns, and dark yellows
        "Dark Autumn": [(100, 25, 0), (100, 50, 0), (50, 25, 0)],  # Dark browns, burnt oranges
        "Dark Winter": [(0, 0, 100), (0, 50, 150), (150, 150, 200)],  # Cool blues, icy tones
        "True Winter": [(0, 50, 150), (0, 0, 100), (0, 0, 50)],  # Cool blues, purples, and dark grays
        "Bright Winter": [(255, 255, 255), (200, 0, 0), (150, 0, 0)],  # Bright whites, bright reds, dark reds
    }



    palette = color_ranges.get(season, [])  # Return color range for the given season, or empty list if not found
    return palette


def classify_image(image_path):
    with open(image_path, 'rb') as uploaded_file:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Apply selfie segmentation
        segmented_image = apply_selfie_segmentation(image, 0)

        # Detect faces in the segmented image
        faces = detect_faces(segmented_image)

        # Crop image to the first detected face
        cropped_image = crop_to_face(segmented_image.copy(), faces)

        # Convert OpenCV images to PIL format for display
        pil_segmented_image = Image.fromarray(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
        if cropped_image is not None:
            pil_cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

            # Detect colors in the cropped image, excluding gray colors
            colors = detect_colors(cropped_image, num_colors=20, detect_lip_color=True)

            # Convert colors to HSL and extract hues, saturation, and lightness values
            hsl_colors = [rgb_to_hsl(r, g, b) for r, g, b in colors]
            hues = [h for h, s, l in hsl_colors]
            lightness_values = [l for h, s, l in hsl_colors]
            saturation_values = [s for h, s, l in hsl_colors]

            # Determine temperature
            temperature = determine_temperature(hues)

            # Determine overall depth
            depth = determine_depth(lightness_values)

            # Determine overall chroma
            chroma = determine_chroma(saturation_values)

            # Determine the season
            season = map_to_season(hues, lightness_values, saturation_values)

            palette = recommend_colors(season)

            return palette, season


# Main function to process all images in a folder and save results to a CSV file
def process_images_in_folder(folder_path):
    results = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            recommended_colors, season = classify_image(image_path)
            if season:
                # Join recommended colors into a single string
                colors_str = ", ".join([str(color) for color in recommended_colors])
                results.append([filename, season, colors_str])

    with open('recommended_colors.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image', 'Season', 'Colors'])
        csv_writer.writerows(results)

# Example usage
folder_path = "/home/niya/mycode/ReStyle/Interface/data/dataset/myntradataset/faces"
process_images_in_folder(folder_path)
