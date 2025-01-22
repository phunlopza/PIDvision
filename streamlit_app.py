import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import openai
import os

# Set OpenAI API Key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit App Title
st.title("Plant Project P&ID Analysis Tool")

# File Upload Section
legend_file = st.file_uploader("Upload P&ID Legend (PDF/Image)", type=["pdf", "png", "jpg"])
pid_file = st.file_uploader("Upload P&ID Diagram (PDF/Image)", type=["pdf", "png", "jpg"])

if legend_file and pid_file:
    st.success("Files uploaded successfully!")
else:
    st.warning("Please upload both the legend and the P&ID diagram.")

# Function to Convert PDFs to Images
def convert_pdf_to_images(pdf_file):
    try:
        images = convert_from_bytes(pdf_file.read())
        return images
    except Exception as e:
        st.error(f"Error converting PDF to images: {e}")
        return None

# Function to Extract Text Using Tesseract
def extract_text(file):
    try:
        if file.type == "application/pdf":
            images = convert_pdf_to_images(file)
            if images:
                text = ""
                for page_image in images:
                    text += pytesseract.image_to_string(page_image)
                return text
        else:
            image = Image.open(file)
            text = pytesseract.image_to_string(image)
            return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None

# Function to Detect Symbols Using OpenCV
def detect_symbols(file):
    try:
        if file.type == "application/pdf":
            images = convert_pdf_to_images(file)
            if images:
                first_page = images[0]
                image = np.array(first_page)
        else:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        return edges
    except Exception as e:
        st.error(f"Error processing P&ID: {e}")
        return None

# Function to Generate Process Description Using OpenAI
def generate_description(legend, components):
    try:
        prompt = f"Using the following legend:\n{legend}\nDescribe the process shown in the P&ID with these components:\n{components}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error generating description: {e}")
        return None

# Extract Legend Text
legend_text = None
if legend_file:
    legend_text = extract_text(legend_file)
    if legend_text:
        st.text_area("Extracted Legend Text:", legend_text, height=200)

# Detect Symbols in P&ID
edges = None
if pid_file:
    edges = detect_symbols(pid_file)
    if edges is not None:
        st.image(edges, caption="Detected Edges in P&ID", use_column_width=True)

# Generate Process Description
description = None
if pid_file and legend_file and legend_text:
    # Example components (replace with actual analysis results in production)
    components = "Pump, Heat Exchanger, Control Valve"
    description = generate_description(legend_text, components)
    if description:
        st.text_area("Generated Process Description:", description, height=200)

# Provide a Download Button for Process Description
if description:
    st.download_button(
        label="Download Process Description",
        data=description,
        file_name="process_description.txt",
        mime="text/plain",
    )
