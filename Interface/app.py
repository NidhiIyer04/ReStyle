import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import colorsys

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
        "Deep Winter": [(0, 0, 50), (0, 50, 100), (100, 0, 50)],  # Dark blues, purples, and cool reds
        "Dark Winter": [(50, 0, 50), (0, 0, 100), (50, 50, 100)],  # Dark purples, blues, and cool grays
        "Cool Winter": [(0, 0, 100), (0, 50, 150), (150, 150, 200)],  # Cool blues, icy tones
        "Bright Winter": [(200, 0, 0), (0, 0, 0), (255, 255, 255)],  # Bright reds, blacks, whites
        "Deep Autumn": [(50, 25, 0), (100, 50, 0), (100, 25, 0)],  # Dark browns, burnt oranges
        "Warm Autumn": [(150, 75, 0), (200, 100, 0), (100, 25, 0)],  # Warm browns, oranges
        "Soft Autumn": [(150, 100, 50), (100, 50, 0), (50, 25, 0)],  # Soft browns, beiges
        "Cool Summer": [(0, 100, 200), (50, 150, 255), (0, 75, 150)],  # Cool blues, aquas
        "Soft Summer": [(100, 100, 150), (75, 75, 100), (50, 50, 75)],  # Soft grays, lavenders
        "Light Summer": [(150, 150, 200), (100, 100, 150), (75, 75, 125)],  # Light lavenders, cool grays
        "Bright Summer": [(255, 0, 150), (255, 50, 200), (255, 100, 255)],  # Bright
    }
# Main function
def main():
    # Streamlit UI
    st.title('Selfie Segmentation App')

    # Sidebar controls
    model_selection = st.sidebar.selectbox('Model Selection', options=[0, 1])

    # File uploader for image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Apply selfie segmentation
        segmented_image = apply_selfie_segmentation(image, model_selection)

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

            # Display the original image, segmented image, cropped image, and color palette
            st.subheader("Original Image")
            st.image(image, caption="Original Image", use_column_width=True)

            st.subheader("Segmented Image with Detected Face")
            st.image(pil_segmented_image, caption="Segmented Image with Detected Face", use_column_width=True)

            st.subheader("Cropped Image with Detected Face")
            st.image(pil_cropped_image, caption="Cropped Image with Detected Face", use_column_width=True)

            st.subheader("Color Palette")
            palette_col = st.columns(len(colors))
            for col, color in zip(palette_col, colors):
                color_square = np.zeros((100, 100, 3), dtype=np.uint8)
                color_square[:, :] = color
                col.image(color_square, use_column_width=True)

            st.subheader(f"Overall Temperature: {temperature}")
            st.subheader(f"Overall Depth: {depth}")
            st.subheader(f"Overall Chroma: {chroma}")
            st.subheader(f"Season: {season}")
            st.text(f"Congratulations!! You are a {season}. ")




if __name__ == '__main__':
    main()
