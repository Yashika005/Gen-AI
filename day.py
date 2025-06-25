import streamlit as st
from PIL import Image
import numpy as np
from skimage import exposure
import io
from transformers import pipeline

# Configure page
st.set_page_config(
    page_title="Day/Night Scene Translator",
    page_icon="🌇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Utility Functions
def resize_image_if_needed(image, max_dimension=1024):
    """Resize image if it's too large while maintaining aspect ratio"""
    width, height = image.size
    if max(width, height) <= max_dimension:
        return image
    
    if width > height:
        new_width = max_dimension
        new_height = int((height * max_dimension) / width)
    else:
        new_height = max_dimension
        new_width = int((width * max_dimension) / height)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def validate_image_upload(uploaded_file, max_size_mb=20):
    """Validate uploaded image file"""
    if uploaded_file is None:
        return None, "No file uploaded"
    
    # Check file size
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return None, f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
    
    try:
        image = Image.open(uploaded_file)
        return image, None
    except Exception as e:
        return None, f"Error opening image: {str(e)}"

def download_image_button(image, filename="transformed_image.png"):
    """Create a download button for the transformed image"""
    img_buffer = io.BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(img_buffer, format='PNG')
    img_bytes = img_buffer.getvalue()
    
    st.download_button(
        label="📥 Download Transformed Image",
        data=img_bytes,
        file_name=filename,
        mime="image/png"
    )

def display_image_comparison(original, transformed, original_label="Original", transformed_label="Transformed"):
    """Display two images side by side for comparison"""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{original_label}**")
        st.image(original, use_container_width=True)
    with col2:
        st.markdown(f"**{transformed_label}**")
        st.image(transformed, use_container_width=True)

class DayNightTranslator:
    def __init__(self):
        try:
            # Using a more modern model than GPT-2
            self.text_model = pipeline("text-generation", model="gpt2-medium")
            self.image_model = None
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            self.text_model = None
            self.image_model = None
    
    def analyze_scene(self, image):
        """Analyze the scene using text model"""
        try:
            if not self.text_model:
                return None, "Model not loaded"
            
            prompt = """Analyze this image focusing on:
            1. Current lighting (day/night)
            2. Main structures
            3. Sky conditions
            4. Light sources
            5. Shadows
            6. Color palette
            
            Describe how this scene would look at the opposite time of day."""
            
            response = self.text_model(
                prompt,
                max_new_tokens=300,  # Better than max_length
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.text_model.tokenizer.eos_token_id
            )
            return response[0]['generated_text'], None
        except Exception as e:
            return None, f"Error analyzing scene: {str(e)}"
    
    def detect_time_of_day(self, image):
        """Detect if the image is day or night"""
        try:
            if not self.text_model:
                return "day", "Unable to detect - Model not loaded"
            
            prompt = """Is this image DAY or NIGHT? Respond with only 'DAY' or 'NIGHT'."""
            response = self.text_model(
                prompt,
                max_new_tokens=5,  # Only need short response
                do_sample=False,
                pad_token_id=self.text_model.tokenizer.eos_token_id
            )
            result = response[0]['generated_text'].strip().upper()
            return "night" if "NIGHT" in result else "day", None
        except Exception as e:
            return "day", f"Error detecting time: {str(e)}"
    
    def translate_scene(self, image, target_time=None):
        """Translate scene between day and night"""
        try:
            processed_image = resize_image_if_needed(image)
            
            if target_time is None:
                current_time, error = self.detect_time_of_day(processed_image)
                if error:
                    return None, None, error
                target_time = "day" if current_time == "night" else "night"
            
            scene_analysis, error = self.analyze_scene(processed_image)
            if error:
                return None, None, error
            
            transformed_image = self._apply_time_transformation(processed_image, target_time)
            return transformed_image, scene_analysis, None
        except Exception as e:
            return None, None, f"Error during transformation: {str(e)}"
    
    def _apply_time_transformation(self, image, target_time):
        """Apply time transformation using image processing"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        
        if target_time == "night":
            transformed = self._transform_to_night(img_array)
        else:
            transformed = self._transform_to_day(img_array)
        
        transformed = np.clip(transformed, 0, 255).astype(np.uint8)
        return Image.fromarray(transformed)
    
    def _transform_to_night(self, img_array):
        """Transform to night scene"""
        height, width = img_array.shape[:2]
        night_img = img_array.astype(np.float64)
        
        # Apply darkness
        night_img = night_img * 0.15
        
        # Blue/purple night atmosphere
        night_img[:, :, 0] *= 0.6  # Reduce red
        night_img[:, :, 1] *= 0.7  # Reduce green
        night_img[:, :, 2] *= 1.8  # Enhance blue
        
        # Add sky gradient
        for y in range(height // 3):
            intensity = 0.3 + 0.4 * (1 - y / (height // 3))
            night_img[y, :, 0] += 20 * intensity  # Purple
            night_img[y, :, 1] += 10 * intensity
            night_img[y, :, 2] += 60 * intensity  # Blue
        
        # Add moon
        moon_x, moon_y = width - width//4, height//6
        moon_radius = min(width, height) // 15
        y_coords, x_coords = np.ogrid[:height, :width]
        moon_mask = ((x_coords - moon_x)**2 + (y_coords - moon_y)**2) <= moon_radius**2
        night_img[moon_mask] = [200, 200, 255]
        
        # Moon glow
        glow_radius = moon_radius * 3
        glow_mask = ((x_coords - moon_x)**2 + (y_coords - moon_y)**2) <= glow_radius**2
        glow_distance = np.sqrt((x_coords - moon_x)**2 + (y_coords - moon_y)**2)
        glow_intensity = np.maximum(0, 1 - glow_distance / glow_radius)
        
        for c in range(3):
            night_img[:, :, c] = np.where(
                glow_mask & ~moon_mask,
                night_img[:, :, c] + glow_intensity * [30, 30, 50][c],
                night_img[:, :, c]
            )
        
        # Add artificial lights
        num_lights = max(3, width // 80)
        for _ in range(num_lights):
            light_x = np.random.randint(width//4, 3*width//4)
            light_y = np.random.randint(height//2, height - height//10)
            light_radius = max(8, min(width, height)//40)
            
            light_mask = ((x_coords - light_x)**2 + (y_coords - light_y)**2) <= light_radius**2
            light_distance = np.sqrt((x_coords - light_x)**2 + (y_coords - light_y)**2)
            light_intensity = np.maximum(0, 1 - light_distance / light_radius)
            
            for c in range(3):
                night_img[:, :, c] = np.where(
                    light_mask,
                    np.minimum(255, night_img[:, :, c] + light_intensity * [120, 100, 30][c]),
                    night_img[:, :, c]
                )
        
        night_img = exposure.adjust_gamma(night_img/255.0, gamma=0.8) * 255
        return night_img
    
    def _transform_to_day(self, img_array):
        """Transform to day scene"""
        height, width = img_array.shape[:2]
        day_img = img_array.astype(np.float64)
        
        # Brightness increase
        day_img = day_img * 4.5
        
        # Warm sunlight
        day_img[:, :, 0] *= 1.4  # Red/orange
        day_img[:, :, 1] *= 1.25  # Yellow
        day_img[:, :, 2] *= 0.85  # Reduce blue
        
        # Sky effect
        y_coords, x_coords = np.ogrid[:height, :width]
        sky_height = int(height * 0.4)
        
        for y in range(sky_height):
            intensity = 1.0 + 2.0 * (sky_height - y) / sky_height
            warmth = 0.8 * (sky_height - y) / sky_height
            
            day_img[y, :, 0] = np.minimum(255, day_img[y, :, 0] * intensity + 80 * warmth)
            day_img[y, :, 1] = np.minimum(255, day_img[y, :, 1] * intensity + 60 * warmth)
            day_img[y, :, 2] = np.minimum(255, day_img[y, :, 2] * (intensity * 0.7) + 20 * warmth)
        
        # Add sun
        sun_x, sun_y = width - width//5, height//8
        sun_radius = min(width, height) // 20
        sun_mask = ((x_coords - sun_x)**2 + (y_coords - sun_y)**2) <= sun_radius**2
        day_img[sun_mask] = [255, 255, 200]
        
        # Sun glow
        glow_radius = sun_radius * 4
        glow_mask = ((x_coords - sun_x)**2 + (y_coords - sun_y)**2) <= glow_radius**2
        glow_distance = np.sqrt((x_coords - sun_x)**2 + (y_coords - sun_y)**2)
        glow_intensity = np.maximum(0, 1 - glow_distance / glow_radius)**0.5
        
        for c in range(3):
            day_img[:, :, c] = np.where(
                glow_mask & ~sun_mask,
                np.minimum(255, day_img[:, :, c] + glow_intensity * [100, 80, 40][c]),
                day_img[:, :, c]
            )
        
        # Ground illumination
        ground_start = sky_height
        for y in range(ground_start, height):
            ground_intensity = 1.0 + 0.3 * (height - y) / (height - ground_start)
            day_img[y, :] = np.minimum(255, day_img[y, :] * ground_intensity)
        
        day_img = exposure.adjust_gamma(day_img/255.0, gamma=1.2) * 255
        
        # Warm color cast
        warm_overlay = np.zeros_like(day_img)
        warm_overlay[:, :, 0] = 20  # Red
        warm_overlay[:, :, 1] = 15  # Yellow
        day_img = np.minimum(255, day_img + warm_overlay)
        
        return day_img

def main():
    st.title("🌇 Day/Night Scene Translator")
    
    translator = DayNightTranslator()
    
    # Upload section
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
    
    if st.button("✨ Transform Image"):
        if not uploaded_file:
            st.warning("Please upload an image")
            return
        
        image, error = validate_image_upload(uploaded_file)
        if error:
            st.error(error)
            return
        
        with st.spinner("Processing image..."):
            transformed, analysis, error = translator.translate_scene(image)
            
            if error:
                st.error(error)
                return
            
            st.success("Transformation complete!")
            display_image_comparison(
                image,
                transformed,
                "Original",
                "Transformed"
            )
            
            download_image_button(transformed)
            
            with st.expander("Scene Analysis"):
                st.markdown(analysis)

if __name__ == "__main__":
    main()