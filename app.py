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
        self.template_hashes = {}

    def load_templates(self):
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
                            img_hash = self.compute_image_hash(standardized)
                            
                            measurements = self.fast_measurements(standardized)
                            
                            class_images.append({
                                'original': img,
                                'standardized': standardized,
                                'filename': os.path.basename(img_path),
                                'class': class_name,
                                'measurements': measurements,
                                'hash': img_hash
                            })
                            
                            self.template_hashes[img_hash] = {
                                'class': class_name,
                                'filename': img_file,
                                'standardized': standardized
                            }
                if class_images:
                    self.templates[class_name] = class_images

        st.success(f"✅ Loaded {sum(len(v) for v in self.templates.values())} templates from {len(self.templates)} classes in {time.time()-start_time:.2f} seconds")

    def compute_image_hash(self, image):
        small = cv2.resize(image, (8, 8))
        avg = np.mean(small)
        hash_value = 0
        for i in range(8):
            for j in range(8):
                if small[i, j] > avg:
                    hash_value |= 1 << (i * 8 + j)
        return hash_value

    def fast_normalize(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return cv2.resize(gray, (self.reference_size, self.reference_size))
        
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        cropped = gray[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (self.reference_size, self.reference_size), interpolation=cv2.INTER_AREA)
        
        return resized

    def fast_measurements(self, standardized_img):
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
        hash1 = self.compute_image_hash(img1)
        hash2 = self.compute_image_hash(img2)
        
        if hash1 == hash2:
            return 1.0
        
        try:
            if img1.shape[0] > 150:
                img1_small = cv2.resize(img1, (150, 150))
                img2_small = cv2.resize(img2, (150, 150))
                similarity = ssim(img1_small, img2_small)
            else:
                similarity = ssim(img1, img2)
            
            edges1 = cv2.Canny(img1, 50, 150)
            edges2 = cv2.Canny(img2, 50, 150)
            
            edge_match = np.sum(edges1 & edges2) / max(np.sum(edges1), np.sum(edges2)) if max(np.sum(edges1), np.sum(edges2)) > 0 else 0
            
            final_score = 0.8 * similarity + 0.2 * edge_match
            
            return min(final_score, 1.0)
        except:
            return ssim(img1, img2)

    def find_similar_profiles(self, user_image, max_matches=5):
        self.load_templates()
        
        start_time = time.time()
        
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
        
        if exact_matches:
            user_measurements = self.fast_measurements(user_standardized)
            st.write(f"✅ Found exact match in {time.time()-start_time:.2f} seconds!")
            return user_standardized, exact_matches[:max_matches], user_measurements
        
        matches = []
        user_edges = cv2.Canny(user_standardized, 50, 150)
        user_edge_count = np.sum(user_edges > 0)
        
        for class_name, template_list in self.templates.items():
            for template in template_list:
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

def display_measurements(measurements, title="📏 Match Measurements"):
    """Display measurements in a clean format"""
    st.subheader(title)
    
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
            st.write(f"• Perimeter: {mm['perimeter']:.1f} mm")

def display_results_with_selection(user_img, processed_user, matches, user_measurements, matcher):
    """Display results with interactive selection"""
    
    st.subheader("📊 Profile Matching Results")
    
    # Display input images
    col1, col2 = st.columns(2)
    with col1:
        st.image(user_img, caption="Original Input", use_column_width=True)
    with col2:
        st.image(processed_user, caption="Normalized", use_column_width=True)
    
    # Show exact match notification
    if matches and matches[0].get('is_exact_match', False):
        st.success(f"🎯 EXACT MATCH FOUND: {matches[0]['class']}")
    
    # Display user measurements
    with st.expander("📏 Your Measurements", expanded=False):
        display_measurements(user_measurements, "Your Profile")
    
    # 1. FIRST: Show all similar matches for selection
    st.subheader("🎯 Select a Match (Click on an image)")
    st.info("Click on any image below to select it as your best match")
    
    # Initialize selection in session state
    if 'selected_match_idx' not in st.session_state:
        st.session_state.selected_match_idx = 0  # Default to first match
    
    # Display matches in columns with clickable selection
    cols = st.columns(len(matches))
    selected_match = None
    
    for idx, (col, match) in enumerate(zip(cols, matches)):
        with col:
            # Create a clickable container for each match
            match_img = Image.fromarray(match['processed'])
            
            # Check if this match is currently selected
            is_selected = st.session_state.selected_match_idx == idx
            
            # Display image with border if selected
            if is_selected:
                col.markdown(f"<div style='border: 3px solid #4CAF50; padding: 5px; border-radius: 5px;'>", unsafe_allow_html=True)
            
            # Display the image
            col.image(match_img, use_column_width=True)
            
            if is_selected:
                col.markdown("</div>", unsafe_allow_html=True)
                selected_match = match
            
            # Add click functionality
            if col.button(f"Select Match {idx+1}", key=f"select_{idx}"):
                st.session_state.selected_match_idx = idx
                st.rerun()  # Refresh to show updated selection
            
            # Show match info
            similarity_value = match['similarity']
            if similarity_value == 1.0 and match.get('is_exact_match', False):
                col.success(f"Exact Match")
            else:
                col.metric(f"Similarity", f"{similarity_value:.3f}")
            
            col.caption(f"Class: {match['class']}")
            col.caption(f"File: {match['filename']}")
            
            if is_selected:
                col.success("✅ Selected")
    
    # 2. THEN: Show the selected match as best match with calculations
    if selected_match is None and len(matches) > 0:
        selected_match = matches[st.session_state.selected_match_idx]
    
    if selected_match:
        st.markdown("---")
        
        # Show the selected match as "Best Match"
        st.subheader(f"🏆 Best Match: {selected_match['class']}")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Show the selected match image
            selected_img = Image.fromarray(selected_match['processed'])
            st.image(selected_img, caption=f"Selected as Best Match", use_column_width=True)
            
            # Show similarity score
            similarity_value = selected_match['similarity']
            if similarity_value == 1.0 and selected_match.get('is_exact_match', False):
                st.success("🎯 Perfect Match (100% identical)")
            else:
                st.metric("Similarity Score", f"{similarity_value:.3f}")
            
            st.caption(f"File: {selected_match['filename']}")
            st.caption(f"Rank: {st.session_state.selected_match_idx + 1} of {len(matches)}")
        
        with col2:
            # 3. Show the relevant calculations for the selected image
            display_measurements(selected_match['measurements'], f"Selected Match")
        
        with col3:
            # Quick comparison with user image
            st.markdown("**📊 Quick Comparison**")
            user_pixels = user_measurements['pixels']
            match_pixels = selected_match['measurements']['pixels']
            
            height_diff = user_pixels['height'] - match_pixels['height']
            width_diff = user_pixels['width'] - match_pixels['width']
            area_diff = user_pixels['area'] - match_pixels['area']
            
            st.write(f"Height diff: {height_diff:+.1f} px")
            st.write(f"Width diff: {width_diff:+.1f} px")
            st.write(f"Area diff: {area_diff:+.1f} px²")
            
            # Aspect ratio similarity
            aspect_similarity = 1 - abs(user_pixels['aspect_ratio'] - match_pixels['aspect_ratio'])
            st.write(f"Aspect similarity: {aspect_similarity:.3f}")
        
        # 4. Show all matches summary table
        st.subheader("📋 All Matches Summary")
        
        summary_data = []
        for i, match in enumerate(matches, 1):
            is_selected = i-1 == st.session_state.selected_match_idx
            measurements = match['measurements']['pixels']
            
            summary_data.append({
                "Rank": i,
                "Class": match['class'],
                "Similarity": f"{match['similarity']:.3f}",
                "Height (px)": f"{measurements['height']:.1f}",
                "Area (px²)": f"{measurements['area']:.1f}",
                "Selected": "✅" if is_selected else ""
            })
        
        import pandas as pd
        df = pd.DataFrame(summary_data)
        st.table(df)

def main():
    st.set_page_config(
        page_title="Interactive Profile Matcher",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Interactive Profile Matching System")
    st.markdown("Select any match by clicking on the image")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    max_matches = st.sidebar.slider("Matches to show", 1, 10, 5)
    
    # Initialize matcher
    if 'matcher' not in st.session_state:
        TEMPLATE_PATH = "trained_data"
        if os.path.exists(TEMPLATE_PATH):
            st.session_state.matcher = OptimizedProfileMatcher(TEMPLATE_PATH)
            st.info("🎯 Interactive matcher initialized")
        else:
            st.error(f"Folder '{TEMPLATE_PATH}' not found!")
            return
    
    # Reset selection when new analysis starts
    if 'reset_selection' not in st.session_state:
        st.session_state.reset_selection = True
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload profile image", 
        type=['png', '.jpg', '.jpeg'],
        help="Upload image for matching"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded", use_column_width=True)
        
        with col2:
            if st.button("🔍 Find Matches", type="primary"):
                with st.spinner("Finding matches..."):
                    try:
                        # Reset selection for new analysis
                        st.session_state.selected_match_idx = 0
                        
                        start_time = time.time()
                        user_img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        processed_user, matches, user_measurements = st.session_state.matcher.find_similar_profiles(
                            user_img_cv, 
                            max_matches=max_matches
                        )
                        
                        total_time = time.time() - start_time
                        
                        processed_user_pil = Image.fromarray(processed_user)
                        user_img_pil = Image.fromarray(cv2.cvtColor(user_img_cv, cv2.COLOR_BGR2RGB))
                        
                        display_results_with_selection(user_img_pil, processed_user_pil, matches, user_measurements, st.session_state.matcher)
                        
                        st.sidebar.success(f"Total time: {total_time:.2f} seconds")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    # Status
    with st.sidebar:
        if hasattr(st.session_state, 'matcher'):
            if st.session_state.matcher.templates:
                template_count = sum(len(v) for v in st.session_state.matcher.templates.values())
                st.success(f"✅ {template_count} templates loaded")
        
        st.markdown("---")
        st.markdown("**🎯 Instructions:**")
        st.markdown("1. Upload an image")
        st.markdown("2. Click 'Find Matches'")
        st.markdown("3. Click on any match image to select it")
        st.markdown("4. View selected match details below")

    with st.expander("ℹ️ How to use"):
        st.markdown("""
        **🎯 Interactive Features:**
        1. **Clickable Images**: Click on any match to select it
        2. **Dynamic Updates**: Measurements update based on selection
        3. **Exact Match Detection**: Identical images show as 100% match
        4. **Comparison View**: Side-by-side with your image
        
        **📊 Selection Workflow:**
        1. All matches shown first for selection
        2. Click any image to select it as "Best Match"
        3. Selected match details shown immediately
        4. Compare with your image measurements
        """)

if __name__ == "__main__":
    main()
