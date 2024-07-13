import streamlit as st
import pandas as pd
import os
import base64

# Function to read CSV with custom parser
def read_csv_with_custom_parser(csv_path):
    df = pd.read_csv(csv_path)
    return df

# Sidebar filters
st.sidebar.header('Filters')

# Try to read CSV file with custom parser
try:
    csv_path = 'data/dataset/myntradataset/styles.csv'
    df = read_csv_with_custom_parser(csv_path)
except Exception as e:
    st.error(f"Error reading CSV file: {e}")
    st.stop()

# Filters
filters = {}
for col in df.columns:
    if col in ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage']:
        filters[col] = st.sidebar.multiselect(
            col,
            df[col].dropna().unique(),
            default=df[col].dropna().unique()
        )

# Apply filters function
def apply_filters(df, filters):
    filtered_df = df.copy()
    for col, vals in filters.items():
        filtered_df = filtered_df[filtered_df[col].isin(vals)]
    return filtered_df

# "Apply Filters" button
if st.sidebar.button('Apply Filters'):
    filtered_df = apply_filters(df, filters)
    st.write(f'Total products: {filtered_df.shape[0]}')

    # Display clothes horizontally with product image and details using custom HTML and CSS
    html_content = """
    <style>
        .product-container {
            display: flex;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid #ccc; /* Add border between products */
        }
        .product-image {
            max-width: 200px; /* Adjust image width as needed */
            margin-right: 20px; /* Add spacing between image and details */
        }
        .product-details {
            flex: 1; /* Expand to fill remaining space */
        }
    </style>
    <div class="products-container">
        <!-- Products will be dynamically inserted here -->
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

    images_path = 'data/dataset/myntradataset/images'
    for index, row in filtered_df.iterrows():
        image_path = os.path.join(images_path, f"{row['id']}.jpg")
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode()
            st.markdown(f"""
                <div class="product-container">
                    <img src="data:image/jpeg;base64,{image_base64}" class="product-image">
                    <div class="product-details">
                        <h3>{row["productDisplayName"]} - {row["articleType"]}</h3>
                        <p><strong>Gender:</strong> {row["gender"]}</p>
                        <p><strong>Master Category:</strong> {row["masterCategory"]}</p>
                        <p><strong>Sub Category:</strong> {row["subCategory"]}</p>
                        <p><strong>Base Colour:</strong> {row["baseColour"]}</p>
                        <p><strong>Season:</strong> {row["season"]}</p>
                        <p><strong>Year:</strong> {row["year"]}</p>
                        <p><strong>Usage:</strong> {row["usage"]}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.write("Image not available.")

# Display the initial sidebar with filters
else:
    st.sidebar.write('Click "Apply Filters" to filter products.')
