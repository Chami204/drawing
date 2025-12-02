import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import streamlit as st
from PIL import Image
import time
import math

class ProfileMatcher:
    def __init__(self, template_root):
        self.template_root = template_root
        self.templates = {}
        self.reference_size = 300
        self.pixels_to_mm_ratio = None
        # Don't load templates in __init__ anymore

    def load_templates(self):
        """Pre-load all template images - only when needed"""
        if self.templates:  # Already loaded
            return
            
        st.write("📂 Loading templates...")
        start_time = time.time()

        for class_name in os.listdir(self.template_root):
            class_path = os.path.join(self.template_root, class_name)
            if os.path.isdir(class_path):
                class_images = []
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            standardized, measurements = self.scale_normalize_with_measurements(img)
                            class_images.append({
                                'original': img,
                                'standardized': standardized,
                                'filename': os.path.basename(img_path),
                                'class': class_name,
                                'measurements': measurements
                            })
                if class_images:
                    self.templates[class_name] = class_images

        st.success(f"✅ Loaded {sum(len(v) for v in self.templates.values())} templates from {len(self.templates)} classes in {time.time()-start_time:.2f} seconds")

    def detect_profile_contour(self, image):
        """Detect the main profile contour"""
        _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        main_contour = max(contours, key=cv2.contourArea)
        return main_contour

    def calculate_measurements(self, contour, scale_factor=1.0):
        """Calculate various measurements from the profile contour"""
        if contour is None:
            return {}
        
        x, y, w, h = cv2.boundingRect(contour)
        
        height_px = h
        width_px = w
        perimeter_px = cv2.arcLength(contour, True)
        area_px = cv2.contourArea(contour)
        
        aspect_ratio = w / h if h > 0 else 0
        compactness = (perimeter_px ** 2) / (4 * math.pi * area_px) if area_px > 0 else 0
        
        contour_points = contour.reshape(-1, 2)
        upper_half_points = [point for point in contour_points if point[1] < y + h/2]
        if upper_half_points:
            nose_point = min(upper_half_points, key=lambda p: p[0])
            chin_point = max(contour_points, key=lambda p: p[1])
            face_height_px = abs(chin_point[1] - nose_point[1])
        else:
            face_height_px = height_px
        
        if self.pixels_to_mm_ratio:
            height_mm = height_px * self.pixels_to_mm_ratio
            width_mm = width_px * self.pixels_to_mm_ratio
            perimeter_mm = perimeter_px * self.pixels_to_mm_ratio
            area_mm2 = area_px * (self.pixels_to_mm_ratio ** 2)
            face_height_mm = face_height_px * self.pixels_to_mm_ratio
        else:
            height_mm = width_mm = perimeter_mm = area_mm2 = face_height_mm = None
        
        measurements = {
            'pixels': {
                'height': height_px,
                'width': width_px,
                'perimeter': perimeter_px,
                'area': area_px,
                'face_height': face_height_px,
                'aspect_ratio': aspect_ratio,
                'compactness': compactness
            },
            'millimeters': {
                'height': height_mm,
                'width': width_mm,
                'perimeter': perimeter_mm,
                'area': area_mm2,
                'face_height': face_height_mm
            } if self.pixels_to_mm_ratio else None
        }
        
        return measurements

    def scale_normalize_with_measurements(self, image):
        """Normalize image scale while preserving aspect ratio"""
        contour = self.detect_profile_contour(image)
        
        if contour is None:
            standardized = cv2.resize(image, (self.reference_size, self.reference_size))
            measurements = self.calculate_measurements(contour)
            return standardized, measurements

        x, y, w, h = cv2.boundingRect(contour)
        profile_region = image[y:y+h, x:x+w]
        
        measurements = self.calculate_measurements(contour)
        
        scale_factor = min(self.reference_size/w, self.reference_size/h)
        new_width = int(w * scale_factor)
        new_height = int(h * scale_factor)
        resized = cv2.resize(profile_region, (new_width, new_height))

        pad_width = (self.reference_size - new_width) // 2
        pad_height = (self.reference_size - new_height) // 2
        padded = cv2.copyMakeBorder(resized,
                                   pad_height, pad_height,
                                   pad_width, pad_width,
                                   cv2.BORDER_CONSTANT, value=255)

        standardized = cv2.resize(padded, (self.reference_size, self.reference_size))
        
        return standardized, measurements

    def preprocess_user_image(self, image):
        """Prepare user image for comparison"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        standardized, measurements = self.scale_normalize_with_measurements(gray)
        return standardized, measurements

    def find_similar_profiles(self, user_image, max_matches=5):
        """Find matching profiles with scale normalization"""
        # Ensure templates are loaded before matching
        self.load_templates()
        
        start_time = time.time()
        processed_user, user_measurements = self.preprocess_user_image(user_image)

        matches = []
        for class_name, template_list in self.templates.items():
            for template in template_list:
                similarity = ssim(processed_user, template['standardized'], full=True)[0]
                matches.append({
                    'similarity': similarity,
                    'class': class_name,
                    'image': template['original'],
                    'processed': template['standardized'],
                    'filename': template['filename'],
                    'measurements': template['measurements']
                })

        matches.sort(key=lambda x: x['similarity'], reverse=True)

        results = []
        seen_classes = set()
        for match in matches:
            if match['class'] not in seen_classes:
                results.append(match)
                seen_classes.add(match['class'])
                if len(results) >= max_matches:
                    break

        st.write(f"⏱️ Matching completed in {time.time()-start_time:.2f} seconds")
        return processed_user, results, user_measurements

def display_measurements(measurements, title="Measurement Analysis"):
    """Display measurements in a clean format"""
    st.subheader(f"📏 {title}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📐 Pixel Measurements:**")
        pixels = measurements['pixels']
        st.write(f"• Height: {pixels['height']:.1f} px")
        st.write(f"• Width: {pixels['width']:.1f} px")
        st.write(f"• Face Height: {pixels['face_height']:.1f} px")
        st.write(f"• Perimeter: {pixels['perimeter']:.1f} px")
        st.write(f"• Area: {pixels['area']:.1f} px²")
        st.write(f"• Aspect Ratio: {pixels['aspect_ratio']:.2f}")
        st.write(f"• Compactness: {pixels['compactness']:.2f}")
    
    with col2:
        if measurements['millimeters']:
            st.markdown("**📏 Real-world Measurements:**")
            mm = measurements['millimeters']
            st.write(f"• Height: {mm['height']:.1f} mm")
            st.write(f"• Width: {mm['width']:.1f} mm")
            st.write(f"• Face Height: {mm['face_height']:.1f} mm")
            st.write(f"• Perimeter: {mm['perimeter']:.1f} mm")
            st.write(f"• Area: {mm['area']:.1f} mm²")
        else:
            st.markdown("**ℹ️ Scale Information:**")
            st.write("To get real-world measurements, set the pixels-to-mm ratio in sidebar.")

def display_results_streamlit(user_img, processed_user, matches, user_measurements):
    """Display results in Streamlit with measurements"""
    
    st.subheader("📊 Profile Matching Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(user_img, caption="Original Input", use_column_width=True)
    with col2:
        st.image(processed_user, caption="Normalized Input", use_column_width=True)
    
    display_measurements(user_measurements, "Your Profile Measurements")
    
    if matches:
        st.subheader("🎯 Best Match Analysis")
        best_match = matches[0]
        
        col1, col2 = st.columns(2)
        with col1:
            match_img = Image.fromarray(best_match['processed'])
            st.image(match_img, use_column_width=True)
            st.metric(
                label=f"Best Match: {best_match['class']}",
                value=f"{best_match['similarity']:.3f}"
            )
            st.caption(f"File: {best_match['filename']}")
        
        with col2:
            display_measurements(best_match['measurements'], f"Reference Measurements ({best_match['class']})")
            
            st.markdown("**📊 Comparison with Your Profile:**")
            user_pixels = user_measurements['pixels']
            match_pixels = best_match['measurements']['pixels']
            
            height_diff = user_pixels['height'] - match_pixels['height']
            area_diff = user_pixels['area'] - match_pixels['area']
            
            st.write(f"• Height difference: {height_diff:+.1f} px")
            st.write(f"• Area difference: {area_diff:+.1f} px²")
            st.write(f"• Aspect ratio similarity: {1 - abs(user_pixels['aspect_ratio'] - match_pixels['aspect_ratio']):.3f}")
    
    st.subheader(f"🏆 Top {len(matches)} Matches")
    
    cols = st.columns(len(matches))
    for idx, (col, match) in enumerate(zip(cols, matches)):
        with col:
            match_img = Image.fromarray(match['processed'])
            st.image(match_img, use_column_width=True)
            st.metric(
                label=f"Match {idx+1}: {match['class']}",
                value=f"{match['similarity']:.3f}"
            )
            st.caption(f"File: {match['filename']}")
    
    st.subheader("📋 Detailed Results")
    
    results_data = []
    for i, match in enumerate(matches, 1):
        measurements = match['measurements']['pixels']
        results_data.append({
            "Rank": i,
            "Class": match['class'],
            "Similarity": f"{match['similarity']:.3f}",
            "Height (px)": f"{measurements['height']:.1f}",
            "Area (px²)": f"{measurements['area']:.1f}",
            "Filename": match['filename']
        })
    
    st.table(results_data)

def main():
    st.set_page_config(
        page_title="Profile Matcher with Measurements",
        page_icon="📏",
        layout="wide"
    )
    
    st.title("📏 Profile Image Matching System")
    st.markdown("Upload a profile image to find similar matches and get detailed measurements.")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    max_matches = st.sidebar.slider("Maximum matches", 1, 10, 5)
    
    st.sidebar.header("📏 Measurement Settings")
    pixels_to_mm = st.sidebar.number_input(
        "Pixels to mm ratio (optional)", 
        min_value=0.0, 
        max_value=10.0, 
        value=0.0,
        help="Set scale for real measurements (e.g., 0.5 = 2 pixels = 1 mm)"
    )
    
    # Initialize matcher only once using session state
    if 'matcher' not in st.session_state:
        TEMPLATE_PATH = "trained_data"
        st.session_state.matcher = ProfileMatcher(TEMPLATE_PATH)
        st.info("🔧 Profile matcher initialized. Ready to load templates when needed.")
    
    # Apply pixels-to-mm ratio if changed
    if pixels_to_mm > 0:
        st.session_state.matcher.pixels_to_mm_ratio = pixels_to_mm
    
    # File upload
    st.header("📤 Upload Profile Image")
    
    uploaded_file = st.file_uploader(
        "Choose a profile image", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a side profile image for analysis"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.info("Ready to analyze!")
            if st.button("🚀 Find Matches & Analyze", type="primary"):
                with st.spinner("Analyzing profile and finding matches..."):
                    try:
                        user_img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        processed_user, matches, user_measurements = st.session_state.matcher.find_similar_profiles(
                            user_img_cv, 
                            max_matches=max_matches
                        )
                        
                        processed_user_pil = Image.fromarray(processed_user)
                        user_img_pil = Image.fromarray(cv2.cvtColor(user_img_cv, cv2.COLOR_BGR2RGB))
                        
                        display_results_streamlit(user_img_pil, processed_user_pil, matches, user_measurements)
                        
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                        st.info("Please try with a different image or check the image format.")
    
    # Show template status
    with st.sidebar:
        st.header("📊 System Status")
        if st.session_state.matcher.templates:
            template_count = sum(len(v) for v in st.session_state.matcher.templates.values())
            class_count = len(st.session_state.matcher.templates)
            st.success(f"✅ Templates loaded: {template_count} images, {class_count} classes")
        else:
            st.info("📁 Templates not loaded yet - will load when you analyze an image")

    # Instructions
    with st.expander("ℹ️ Instructions"):
        st.markdown("""
        **How to use:**
        1. Upload a clear side profile image
        2. Click 'Find Matches & Analyze'
        3. View matching results and measurements
        
        **Measurements include:**
        - Height, Width, Area, Perimeter
        - Face Height (nose to chin)
        - Aspect Ratio & Compactness
        - Real-world measurements (if scale set)
        
        **For best results:**
        - Use clear, well-lit profile images
        - Ensure good contrast between subject and background
        - Set pixels-to-mm ratio for real measurements
        """)

if __name__ == "__main__":
    main()
