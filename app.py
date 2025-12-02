import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import streamlit as st
from PIL import Image
import time
import math

class OptimizedProfileMatcher:
    def __init__(self, template_root):
        self.template_root = template_root
        self.templates = {}
        self.reference_size = 300
        self.pixels_to_mm_ratio = None
        self.template_hashes = {}  # Store hashes for quick exact match detection

    def load_templates(self):
        """Fast template loading with hash computation"""
        if self.templates:
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
                            standardized = self.fast_normalize(img)
                            
                            # Compute hash for quick exact match detection
                            img_hash = self.compute_image_hash(standardized)
                            
                            # Compute basic features (fast)
                            measurements = self.fast_measurements(standardized)
                            
                            class_images.append({
                                'original': img,
                                'standardized': standardized,
                                'filename': os.path.basename(img_path),
                                'class': class_name,
                                'measurements': measurements,
                                'hash': img_hash
                            })
                            
                            # Store hash for quick lookup
                            self.template_hashes[img_hash] = {
                                'class': class_name,
                                'filename': img_file,
                                'standardized': standardized
                            }
                if class_images:
                    self.templates[class_name] = class_images

        st.success(f"✅ Loaded {sum(len(v) for v in self.templates.values())} templates from {len(self.templates)} classes in {time.time()-start_time:.2f} seconds")

    def compute_image_hash(self, image):
        """Compute perceptual hash for quick exact match detection"""
        # Resize to 8x8 for faster hashing
        small = cv2.resize(image, (8, 8))
        
        # Compute average
        avg = np.mean(small)
        
        # Create binary hash
        hash_value = 0
        for i in range(8):
            for j in range(8):
                if small[i, j] > avg:
                    hash_value |= 1 << (i * 8 + j)
        
        return hash_value

    def fast_normalize(self, image):
        """Fast image normalization"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Simple threshold and contour detection
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return cv2.resize(gray, (self.reference_size, self.reference_size))
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Crop and resize
        cropped = gray[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (self.reference_size, self.reference_size), interpolation=cv2.INTER_AREA)
        
        return resized

    def fast_measurements(self, standardized_img):
        """Fast measurements computation"""
        _, thresh = cv2.threshold(standardized_img, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'pixels': {
                    'height': 0, 'width': 0, 'area': 0,
                    'perimeter': 0, 'aspect_ratio': 0
                },
                'millimeters': None
            }
        
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        
        height_px = h
        width_px = w
        area_px = cv2.contourArea(contour)
        perimeter_px = cv2.arcLength(contour, True)
        aspect_ratio = w / h if h > 0 else 0
        
        if self.pixels_to_mm_ratio:
            height_mm = height_px * self.pixels_to_mm_ratio
            width_mm = width_px * self.pixels_to_mm_ratio
            area_mm2 = area_px * (self.pixels_to_mm_ratio ** 2)
            perimeter_mm = perimeter_px * self.pixels_to_mm_ratio
        else:
            height_mm = width_mm = area_mm2 = perimeter_mm = None
        
        return {
            'pixels': {
                'height': height_px,
                'width': width_px,
                'area': area_px,
                'perimeter': perimeter_px,
                'aspect_ratio': aspect_ratio
            },
            'millimeters': {
                'height': height_mm,
                'width': width_mm,
                'area': area_mm2,
                'perimeter': perimeter_mm
            } if self.pixels_to_mm_ratio else None
        }

    def fast_similarity(self, img1, img2):
        """Optimized similarity calculation with exact match priority"""
        # Check for exact match first (using hash)
        hash1 = self.compute_image_hash(img1)
        hash2 = self.compute_image_hash(img2)
        
        # If hashes are identical, it's an exact match
        if hash1 == hash2:
            return 1.0  # Perfect match
        
        # Compute SSIM (faster than multi-feature approach)
        try:
            # Downsample for faster computation if needed
            if img1.shape[0] > 150:
                img1_small = cv2.resize(img1, (150, 150))
                img2_small = cv2.resize(img2, (150, 150))
                similarity = ssim(img1_small, img2_small)
            else:
                similarity = ssim(img1, img2)
            
            # Add edge similarity bonus (fast computation)
            edges1 = cv2.Canny(img1, 50, 150)
            edges2 = cv2.Canny(img2, 50, 150)
            
            edge_match = np.sum(edges1 & edges2) / max(np.sum(edges1), np.sum(edges2)) if max(np.sum(edges1), np.sum(edges2)) > 0 else 0
            
            # Combined score with edge bonus
            final_score = 0.8 * similarity + 0.2 * edge_match
            
            return min(final_score, 1.0)
        except:
            return ssim(img1, img2)

    def find_similar_profiles(self, user_image, max_matches=5):
        """Optimized matching with exact match priority"""
        self.load_templates()
        
        start_time = time.time()
        
        # Fast preprocessing
        if len(user_image.shape) == 3:
            user_gray = cv2.cvtColor(user_image, cv2.COLOR_BGR2GRAY)
        else:
            user_gray = user_image.copy()
        
        user_standardized = self.fast_normalize(user_gray)
        user_hash = self.compute_image_hash(user_standardized)
        
        # Check for exact match first
        exact_matches = []
        if user_hash in self.template_hashes:
            exact_match = self.template_hashes[user_hash]
            exact_matches.append({
                'similarity': 1.0,
                'class': exact_match['class'],
                'filename': exact_match['filename'],
                'standardized': exact_match['standardized'],
                'measurements': self.fast_measurements(exact_match['standardized']),
                'is_exact_match': True
            })
        
        # If we found exact matches, just return them
        if exact_matches:
            user_measurements = self.fast_measurements(user_standardized)
            st.write(f"✅ Found exact match in {time.time()-start_time:.2f} seconds!")
            return user_standardized, exact_matches[:max_matches], user_measurements
        
        # Otherwise, do regular matching (optimized)
        matches = []
        
        # Pre-compute user features once
        user_edges = cv2.Canny(user_standardized, 50, 150)
        user_edge_count = np.sum(user_edges > 0)
        
        for class_name, template_list in self.templates.items():
            for template in template_list:
                # Fast similarity computation
                similarity = self.fast_similarity(user_standardized, template['standardized'])
                
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

        user_measurements = self.fast_measurements(user_standardized)
        
        st.write(f"⏱️ Matching completed in {time.time()-start_time:.2f} seconds")
        return user_standardized, results, user_measurements

def display_measurements(measurements, title="Measurements"):
    """Fast measurement display"""
    st.subheader(f"📏 {title}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📐 Pixel Measurements:**")
        pixels = measurements['pixels']
        st.write(f"• Height: {pixels['height']:.1f} px")
        st.write(f"• Width: {pixels['width']:.1f} px")
        st.write(f"• Area: {pixels['area']:.1f} px²")
        st.write(f"• Perimeter: {pixels['perimeter']:.1f} px")
        st.write(f"• Aspect Ratio: {pixels['aspect_ratio']:.2f}")
    
    with col2:
        if measurements['millimeters']:
            st.markdown("**📏 Real-world Measurements:**")
            mm = measurements['millimeters']
            st.write(f"• Height: {mm['height']:.1f} mm")
            st.write(f"• Width: {mm['width']:.1f} mm")
            st.write(f"• Area: {mm['area']:.1f} mm²")

def display_results_fast(user_img, processed_user, matches, user_measurements):
    """Fast results display"""
    
    st.subheader("⚡ Quick Matching Results")
    
    # Show exact match notification
    if matches and matches[0].get('is_exact_match', False):
        st.success(f"🎯 EXACT MATCH FOUND: {matches[0]['class']}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(user_img, caption="Original Input", use_column_width=True)
    with col2:
        st.image(processed_user, caption="Normalized", use_column_width=True)
    
    # Quick measurements
    with st.expander("📐 Your Measurements"):
        display_measurements(user_measurements)
    
    if matches:
        # Best match
        st.subheader("🏆 Best Match")
        best_match = matches[0]
        
        col1, col2 = st.columns(2)
        with col1:
            match_img = Image.fromarray(best_match['processed'])
            st.image(match_img, use_column_width=True)
            
            similarity_value = best_match['similarity']
            if similarity_value == 1.0:
                st.success(f"Exact Match: {best_match['class']}")
            else:
                st.metric(f"Best Match: {best_match['class']}", f"{similarity_value:.3f}")
            
            st.caption(f"File: {best_match['filename']}")
        
        with col2:
            if not best_match.get('is_exact_match', False):
                display_measurements(best_match['measurements'], "Match Measurements")
                
                # Quick comparison
                user_pixels = user_measurements['pixels']
                match_pixels = best_match['measurements']['pixels']
                
                st.markdown("**📊 Quick Comparison:**")
                height_diff = user_pixels['height'] - match_pixels['height']
                area_diff = user_pixels['area'] - match_pixels['area']
                st.write(f"• Height diff: {height_diff:+.1f} px")
                st.write(f"• Area diff: {area_diff:+.1f} px²")
    
        # All matches grid
        st.subheader(f"📋 Top {len(matches)} Matches")
        
        cols = st.columns(len(matches))
        for idx, (col, match) in enumerate(zip(cols, matches)):
            with col:
                match_img = Image.fromarray(match['processed'])
                st.image(match_img, use_column_width=True)
                
                similarity = match['similarity']
                if similarity == 1.0:
                    st.success(f"Exact")
                else:
                    st.metric(f"Match {idx+1}", f"{similarity:.3f}")
                
                st.caption(f"{match['class']}")

def main():
    st.set_page_config(
        page_title="Fast Profile Matcher",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ Fast Profile Matching System")
    st.markdown("Optimized for speed with exact match detection")
    
    # Simple configuration
    st.sidebar.header("⚙️ Quick Settings")
    max_matches = st.sidebar.slider("Matches to show", 1, 10, 3)
    
    # Initialize matcher
    if 'matcher' not in st.session_state:
        TEMPLATE_PATH = "trained_data"
        if os.path.exists(TEMPLATE_PATH):
            st.session_state.matcher = OptimizedProfileMatcher(TEMPLATE_PATH)
            st.info("⚡ Fast matcher initialized")
        else:
            st.error(f"Folder '{TEMPLATE_PATH}' not found!")
            return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload profile image", 
        type=['png', '.jpg', '.jpeg'],
        help="Upload image for fast matching"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded", use_column_width=True)
        
        with col2:
            if st.button("⚡ Find Matches", type="primary"):
                with st.spinner("Matching..."):
                    try:
                        start_time = time.time()
                        
                        user_img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        processed_user, matches, user_measurements = st.session_state.matcher.find_similar_profiles(
                            user_img_cv, 
                            max_matches=max_matches
                        )
                        
                        total_time = time.time() - start_time
                        
                        processed_user_pil = Image.fromarray(processed_user)
                        user_img_pil = Image.fromarray(cv2.cvtColor(user_img_cv, cv2.COLOR_BGR2RGB))
                        
                        display_results_fast(user_img_pil, processed_user_pil, matches, user_measurements)
                        
                        st.sidebar.success(f"Total time: {total_time:.2f} seconds")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    # Quick status
    with st.sidebar:
        if hasattr(st.session_state, 'matcher'):
            if st.session_state.matcher.templates:
                template_count = sum(len(v) for v in st.session_state.matcher.templates.values())
                st.success(f"✅ {template_count} templates ready")
            else:
                st.info("Templates will load on first match")

    # Simple instructions
    with st.expander("ℹ️ Quick Info"):
        st.markdown("""
        **⚡ Features:**
        - Exact match detection (identical images)
        - Fast SSIM-based matching
        - Quick measurements
        - Optimized for speed
        
        **🎯 How it works:**
        1. Computes image hash for exact matches
        2. If exact match found: returns immediately (1.0 score)
        3. Otherwise: fast SSIM + edge matching
        4. Results in seconds, not minutes
        """)

if __name__ == "__main__":
    main()


