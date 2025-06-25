# anime.py (updated with Hugging Face option)
import streamlit as st
import os
from PIL import Image
import numpy as np
import cv2
from transformers import pipeline

class AnimeStyleConverter:
    def __init__(self, use_huggingface=False):
        self.use_huggingface = use_huggingface
        if use_huggingface:
            try:
                self.model = pipeline("image-to-image", model="AK391/animegan2-pytorch")
            except:
                self.model = None
        else:
            self.model = None
    
    def resize_image_if_needed(self, image, max_dimension=1024):
        """Resize image if any dimension exceeds max_dimension while maintaining aspect ratio"""
        width, height = image.size
        
        if width <= max_dimension and height <= max_dimension:
            return image
        
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
            
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    def convert_to_anime(self, image):
        """Convert image to anime style"""
        try:
            processed_image = self.resize_image_if_needed(image, max_dimension=1024)
            
            if self.use_huggingface and self.model:
                # Use Hugging Face model
                anime_image = self.model(processed_image)
                anime_image = anime_image[0]  # Get the first image from the output
            else:
                # Use OpenCV method
                anime_image = self._apply_anime_effect(processed_image)
            
            return anime_image, None
            
        except Exception as e:
            return None, f"Error during conversion: {str(e)}"
    
    def _apply_anime_effect(self, image):
        """Apply anime/cartoon effect using OpenCV"""
        img_array = np.array(image.convert('RGB'))
        
        # Step 1: Edge-preserving smoothing
        smooth = cv2.edgePreservingFilter(img_array, flags=1, sigma_s=60, sigma_r=0.4)
        
        # Step 2: Enhance edges
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, blockSize=9, C=9)
        
        # Step 3: Color quantization
        data = smooth.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        quantized = quantized.reshape(img_array.shape)
        
        # Step 4: Combine with edges
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        anime = cv2.bitwise_and(quantized, edges)
        
        return Image.fromarray(anime)

# Streamlit UI
st.title("Photo to Anime Converter")

use_hf = st.checkbox("Use Hugging Face Model (better quality but slower)", value=False)
uploaded_file = st.file_uploader("Upload photo", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Photo", use_container_width=True)
    
    if st.button("Convert to Anime"):
        converter = AnimeStyleConverter(use_huggingface=use_hf)
        with st.spinner("Converting to anime style..."):
            anime_img, error = converter.convert_to_anime(image)
            
            if error:
                st.error(f"Error: {error}")
            else:
                st.image(anime_img, caption="Anime Version", use_container_width=True)
                
                # Download button
                st.download_button(
                    label="Download Anime Image",
                    data=cv2.imencode('.png', np.array(anime_img))[1].tobytes(),
                    file_name="anime_version.png",
                    mime="image/png"
                )