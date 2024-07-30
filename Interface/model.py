import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import colorsys
import streamlit as st
import pandas as pd
import subprocess
import time
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

def apply_selfie_segmentation(image, model_selection):
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=model_selection) as selfie_segmentation:
        results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (192, 192, 192)  # Default background color
        output_image = np.where(condition, image, bg_image)
        return output_image

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces


def crop_to_face(image, faces):
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image

def detect_colors(image, num_colors=20, detect_lip_color=False):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure image is in RGB format for KMeans
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(img)
    colors = kmeans.cluster_centers_

    filtered_colors = []
    for color in colors:
        r, g, b = color
        if not (r == g == b or abs(r - g) < 15 and abs(r - b) < 15 and abs(g - b) < 15):
            filtered_colors.append(tuple(map(int, color)))

    if detect_lip_color:

        lip_roi = image[150:250, 100:300]  # Example ROI coordinates

        lip_roi_rgb = cv2.cvtColor(lip_roi, cv2.COLOR_BGR2RGB)
        lip_roi_rgb = lip_roi_rgb.reshape((lip_roi_rgb.shape[0] * lip_roi_rgb.shape[1], 3))

        kmeans_lip = KMeans(n_clusters=1)
        kmeans_lip.fit(lip_roi_rgb)
        lip_color = tuple(map(int, kmeans_lip.cluster_centers_[0]))
        filtered_colors.append(lip_color)

    if len(filtered_colors) < 20:
        num_additional_colors = 20 - len(filtered_colors)
        additional_colors = np.random.randint(0, 256, size=(num_additional_colors, 3))
        filtered_colors.extend(map(tuple, additional_colors))

    return filtered_colors[:20]


def rgb_to_hsl(r, g, b):
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    h = h * 360  
    s = s * 100  
    l = l * 100  
    return h, s, l

def determine_temperature(hues):
    warm_hues = sum(1 for h in hues if 0 <= h <= 60 or 300 <= h <= 360)
    cool_hues = len(hues) - warm_hues
    return "Warm" if warm_hues >= cool_hues else "Cool"

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

def determine_chroma(saturation_values):
    average_saturation = sum(saturation_values) / len(saturation_values)
    if average_saturation < 20:
        return "Low"
    elif 20 <= average_saturation < 40:
        return "Medium"
    else:
        return "High"

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
        if 0 <= h < 45 or 330 <= h <= 360:  
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

    predominant_season = max(seasons, key=seasons.get)
    return predominant_season

def recommend_colors(season):
    color_map = {
        'Bright Spring': 'red-yellow-blue',
        'True Spring': 'yellow-green-blue',
        'Light Spring': 'pink-beige-blue',
        'Light Summer': 'pastel-blue-green',
        'True Summer': 'blue-green-pink',
        'Soft Summer': 'grey-blue-green',
        'Soft Autumn': 'earth-tone',
        'True Autumn': 'orange-brown-green',
        'Deep Autumn': 'dark-brown-green',
        'Deep Winter': 'black-grey-blue',
        'True Winter': 'black-white-red',
        'Bright Winter': 'bright-blue-red-white'
    }
    colors = color_map.get(season, 'black')
    color_query = '+'.join(colors.split('-'))
    url = f"https://www.myntra.com/{colors}?rawQuery={color_query}"

    color_ranges = {
        'Bright Spring': [(255, 0, 0), (255, 255, 0), (0, 0, 255)],
        'True Spring': [(255, 255, 0), (0, 255, 0), (0, 0, 255)],
        'Light Spring': [(255, 192, 203), (255, 228, 225), (0, 191, 255)],
        'Light Summer': [(173, 216, 230), (144, 238, 144), (152, 251, 152)],
        'True Summer': [(70, 130, 180), (0, 255, 255), (255, 192, 203)],
        'Soft Summer': [(128, 128, 128), (192, 192, 192), (0, 128, 128)],
        'Soft Autumn': [(139, 69, 19), (160, 82, 45), (205, 133, 63)],
        'True Autumn': [(255, 69, 0), (139, 69, 19), (0, 128, 0)],
        'Deep Autumn': [(101, 67, 33), (139, 69, 19), (0, 100, 0)],
        'Deep Winter': [(0, 0, 0), (169, 169, 169), (0, 0, 139)],
        'True Winter': [(0, 0, 0), (255, 255, 255), (255, 0, 0)],
        'Bright Winter': [(0, 0, 255), (255, 0, 0), (255, 255, 255)]
    }
    palette = color_ranges.get(season, [])
    
    return [url, palette]
def open_url(url):
    webbrowser.open(url)
def main():
    # Streamlit UI
    st.title('ReStyle')

    uploaded_file = st.file_uploader("Upload an image of yourself", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image (optional)
        #st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Step 2: Display a video with autoplay
        video_file = "./animation.mp4"  # Replace with the path to your video file

        # Check if the file exists
        try:
            with open(video_file, "rb") as video:
                st.video(video.read())
            
            # Simulate the video playing duration (example: 7 seconds)
        except:
            time.sleep(7)

        # Simulate the video playing duration (example: 7 seconds)
        #time.sleep(7)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        segmented_image = apply_selfie_segmentation(image, 0)

        faces = detect_faces(segmented_image)

        cropped_image = crop_to_face(segmented_image.copy(), faces)

        pil_segmented_image = Image.fromarray(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
        if cropped_image is not None:
            pil_cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

            colors = detect_colors(cropped_image, num_colors=20, detect_lip_color=True)

            hsl_colors = [rgb_to_hsl(r, g, b) for r, g, b in colors]
            hues = [h for h, s, l in hsl_colors]
            lightness_values = [l for h, s, l in hsl_colors]
            saturation_values = [s for h, s, l in hsl_colors]
            
            temperature = determine_temperature(hues)
            depth = determine_depth(lightness_values)
            chroma = determine_chroma(saturation_values)
            season = map_to_season(hues, lightness_values, saturation_values)

            recommended_colors_url, recommended_colors_palette = recommend_colors(season)
            st.markdown(
                f'<a href="{recommended_colors_url}" target="_blank" class="button">Check out the recommended products on Myntra</a>',
                unsafe_allow_html=True
            )

            # Add custom CSS to style the link as a button
            st.markdown(
                """
                <style>
                .button {
                    display: inline-block;
                    padding: 10px 20px;
                    font-size: 16px;
                    font-weight: bold;
                    color: white;
                    background-color: #000000;
                    border-radius: 5px;
                    text-align: center;
                    text-decoration: none;
                }
                .button:hover {
                    background-color: #000000;
                }
                </style>
                """,
                unsafe_allow_html=True
            )



if __name__ == '__main__':
    main()
