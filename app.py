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
                            standardized = self.fast_normalize(img, preserve_aspect=True)
                            img_hash = self.compute_image_hash(standardized)
                            
                            class_images.append({
                                'original': img,
                                'standardized': standardized,
                                'filename': os.path.basename(img_path),
                                'class': class_name,
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

    def fast_normalize(self, image, preserve_aspect=True):
        """Normalize image while preserving original proportions"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # If no contour found, return original size resized to fit
            h, w = gray.shape
            scale = self.reference_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            return cv2.resize(gray, (new_w, new_h))
        
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract the profile region
        profile_region = gray[y:y+h, x:x+w]
        
        if preserve_aspect:
            # Preserve original aspect ratio - fit within reference_size
            scale = self.reference_size / max(w, h)
            new_width = int(w * scale)
            new_height = int(h * scale)
            
            # Resize preserving aspect ratio
            resized = cv2.resize(profile_region, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Create a canvas of reference_size x reference_size with white background
            result = np.ones((self.reference_size, self.reference_size), dtype=np.uint8) * 255
            
            # Calculate position to center the resized image
            x_offset = (self.reference_size - new_width) // 2
            y_offset = (self.reference_size - new_height) // 2
            
            # Place the resized image in the center
            result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            
            return result
        else:
            # Original behavior - stretch to square
            return cv2.resize(profile_region, (self.reference_size, self.reference_size), interpolation=cv2.INTER_AREA)

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
            
            final_score = 0.35 * similarity + 0.65 * edge_match
            
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
        
        user_standardized = self.fast_normalize(user_gray, preserve_aspect=True)
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
                'is_exact_match': True
            })
        
        if exact_matches:
            st.write(f"✅ Found exact match in {time.time()-start_time:.2f} seconds!")
            return user_standardized, exact_matches[:max_matches]
        
        matches = []
        
        for class_name, template_list in self.templates.items():
            for template in template_list:
                similarity = self.fast_similarity(user_standardized, template['standardized'])
                
                matches.append({
                    'similarity': similarity,
                    'class': class_name,
                    'image': template['original'],
                    'processed': template['standardized'],
                    'filename': template['filename']
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
        return user_standardized, results

def display_results_with_selection(user_img, processed_user, matches):
    """Display results with interactive selection"""
    
    st.subheader("📊 Profile Matching Results")
    
    # Display input images
    col1, col2 = st.columns(2)
    with col1:
        st.image(user_img, caption="Original Input", use_column_width=True)
    with col2:
        st.image(processed_user, caption="Normalized (Preserved Aspect Ratio)", use_column_width=True)
    
    # Show exact match notification
    if matches and matches[0].get('is_exact_match', False):
        st.success(f"🎯 EXACT MATCH FOUND: {matches[0]['class']}")
    
    # 1. FIRST: Show all similar matches for selection
    st.subheader("🎯 Select a Match (Click on an image)")
    st.info("Click on any image below to select it as your best match")
    
    # Initialize selection in session state
    if 'selected_match_idx' not in st.session_state:
        st.session_state.selected_match_idx = 0  # Default to first match
    
    # Use callback functions for button clicks
    def set_selected_idx(idx):
        st.session_state.selected_match_idx = idx
    
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
                selected_match = match
            
            # Display the image
            col.image(match_img, use_column_width=True)
            
            if is_selected:
                col.markdown("</div>", unsafe_allow_html=True)
            
            # Add click functionality
            col.button(
                f"Select Match {idx+1}", 
                key=f"select_{idx}",
                on_click=set_selected_idx,
                args=(idx,)
            )
            
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
    
    # 2. THEN: Show the selected match as best match
    if selected_match is None and len(matches) > 0:
        selected_match = matches[st.session_state.selected_match_idx]
    
    if selected_match:
        st.markdown("---")
        
        # Show the selected match as "Best Match"
        st.subheader(f"🏆 Best Match: {selected_match['class']}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show the selected match image
            selected_img = Image.fromarray(selected_match['processed'])
            
            # Get original image dimensions if available
            if 'image' in selected_match and selected_match['image'] is not None:
                original_img = selected_match['image']
                if isinstance(original_img, np.ndarray):
                    h, w = original_img.shape[:2]
                    st.caption(f"Original size: {w}×{h} pixels")
            
            st.image(selected_img, caption=f"Selected as Best Match (Preserved Aspect Ratio)", use_column_width=True)
            
            # Show similarity score
            similarity_value = selected_match['similarity']
            if similarity_value == 1.0 and selected_match.get('is_exact_match', False):
                st.success("🎯 Perfect Match (100% identical)")
            else:
                st.metric("Similarity Score", f"{similarity_value:.3f}")
            
            st.caption(f"File: {selected_match['filename']}")
            st.caption(f"Rank: {st.session_state.selected_match_idx + 1} of {len(matches)}")
        
        with col2:
            # Show match details
            st.markdown("**📊 Match Details**")
            st.write(f"**Class:** {selected_match['class']}")
            st.write(f"**Similarity:** {selected_match['similarity']:.3f}")
            st.write(f"**Filename:** {selected_match['filename']}")
            
            if selected_match.get('is_exact_match', False):
                st.success("**Type:** Exact Match")
            else:
                st.info("**Type:** Similar Match")
        
        # 4. Show all matches summary table
        st.subheader("📋 All Matches Summary")
        
        import pandas as pd
        summary_data = []
        for i, match in enumerate(matches, 1):
            is_selected = i-1 == st.session_state.selected_match_idx
            
            summary_data.append({
                "Rank": i,
                "Class": match['class'],
                "Similarity": f"{match['similarity']:.3f}",
                "Type": "Exact" if match.get('is_exact_match', False) else "Similar",
                "Selected": "✅" if is_selected else ""
            })
        
        df = pd.DataFrame(summary_data)
        st.table(df)

def main():
    st.set_page_config(
        page_title="Profile Matcher",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 ALU SCAN - The Aluminum Profile Search Engine")
    st.markdown("Upload an image to find similar profiles")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    max_matches = st.sidebar.slider("Matches to show", 1, 10, 5)
    
    # Initialize matcher in session state
    if 'matcher' not in st.session_state:
        TEMPLATE_PATH = "trained_data"
        if os.path.exists(TEMPLATE_PATH):
            st.session_state.matcher = OptimizedProfileMatcher(TEMPLATE_PATH)
            st.info("🔍 Profile matcher initialized")
        else:
            st.error(f"Folder '{TEMPLATE_PATH}' not found!")
            return
    
    # Initialize session state variables
    if 'processed_user' not in st.session_state:
        st.session_state.processed_user = None
    if 'matches' not in st.session_state:
        st.session_state.matches = None
    if 'user_img_pil' not in st.session_state:
        st.session_state.user_img_pil = None
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload profile image", 
        type=['png', '.jpg', '.jpeg'],
        help="Upload image for matching",
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded", use_column_width=True)
        
        with col2:
            # Check if we need to analyze or re-analyze
            analyze_needed = True
            if st.session_state.analysis_done:
                # Check if this is the same file (by comparing first few bytes)
                current_file_bytes = uploaded_file.getvalue()[:100]
                if 'last_file_bytes' in st.session_state:
                    if current_file_bytes == st.session_state.last_file_bytes:
                        analyze_needed = False
            
            if analyze_needed:
                if st.button("🔍 Find Matches", type="primary"):
                    with st.spinner("Finding matches..."):
                        try:
                            # Store file bytes for comparison
                            st.session_state.last_file_bytes = uploaded_file.getvalue()[:100]
                            
                            # Reset selection
                            st.session_state.selected_match_idx = 0
                            
                            start_time = time.time()
                            user_img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                            
                            # Perform analysis
                            processed_user, matches = st.session_state.matcher.find_similar_profiles(
                                user_img_cv, 
                                max_matches=max_matches
                            )
                            
                            total_time = time.time() - start_time
                            
                            # Convert images for display
                            processed_user_pil = Image.fromarray(processed_user)
                            user_img_pil = Image.fromarray(cv2.cvtColor(user_img_cv, cv2.COLOR_BGR2RGB))
                            
                            # Store everything in session state
                            st.session_state.processed_user = processed_user_pil
                            st.session_state.matches = matches
                            st.session_state.user_img_pil = user_img_pil
                            st.session_state.analysis_done = True
                            st.session_state.analysis_time = total_time
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            else:
                # Show re-analyze option
                if st.button("🔄 Re-analyze", type="secondary"):
                    st.session_state.analysis_done = False
                    st.rerun()
    
    # Display results if analysis is done
    if st.session_state.analysis_done and st.session_state.matches is not None:
        # Show analysis time in sidebar
        if 'analysis_time' in st.session_state:
            st.sidebar.success(f"Analysis time: {st.session_state.analysis_time:.2f} seconds")
        
        # Display the results with selection
        display_results_with_selection(
            st.session_state.user_img_pil, 
            st.session_state.processed_user, 
            st.session_state.matches
        )
    
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
        st.markdown("4. Selected match shown as Best Match")

    with st.expander("ℹ️ How to use"):
        st.markdown("""
        **🔍 Features:**
        1. **Clickable Images**: Click on any match to select it
        2. **Exact Match Detection**: Identical images show as 100% match
        3. **Preserved Aspect Ratio**: Images shown in original proportions
        4. **Fast Matching**: Optimized for speed
        
        **📊 Workflow:**
        1. All matches shown first for selection
        2. Click any image to select it as "Best Match"
        3. Selected match shown with details
        4. No measurements - focus on visual similarity
        """)

if __name__ == "__main__":
    main()







