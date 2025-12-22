import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import streamlit as st
from PIL import Image
import time
import math
import pickle
import hashlib
import gc

class OptimizedProfileMatcher:
    def __init__(self, template_root):
        self.template_root = template_root
        self.templates = {}
        self.template_hashes = {}
        self.reference_size = 300
        self.cache_dir = ".matcher_cache"
        
        # Create cache directory
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Load only metadata initially
        self.templates_loaded = False
        self.template_count = 0
        self.shape_types = {}

    def load_templates_lazy(self):
        """Load template metadata without loading full images into memory"""
        if self.templates_loaded:
            return
            
        st.write("📂 Loading template metadata...")
        start_time = time.time()
        
        metadata_file = os.path.join(self.cache_dir, "templates_metadata.pkl")
        
        # Try to load from cache
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'rb') as f:
                    self.templates = pickle.load(f)
                    self.template_hashes = pickle.load(f)
                    self.shape_types = pickle.load(f)
                self.template_count = sum(len(v) for v in self.templates.values())
                st.success(f"✅ Loaded {self.template_count} template metadata from cache")
                self.templates_loaded = True
                return
            except:
                pass
        
        # Build metadata
        self.templates = {}
        self.template_hashes = {}
        self.shape_types = {}
        
        for class_name in os.listdir(self.template_root):
            class_path = os.path.join(self.template_root, class_name)
            if os.path.isdir(class_path):
                class_images = []
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        
                        # Generate unique ID for each template
                        template_id = f"{class_name}_{img_file}"
                        
                        # Store only metadata, not full images
                        class_images.append({
                            'path': img_path,
                            'filename': img_file,
                            'class': class_name,
                            'id': template_id,
                            'cached_file': os.path.join(self.cache_dir, f"{hashlib.md5(template_id.encode()).hexdigest()}.npz")
                        })
                        
                        # Load and process to get hash and shape type
                        try:
                            # Load image temporarily to compute hash
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                standardized = self.fast_normalize(img, preserve_aspect=True)
                                img_hash = self.compute_image_hash(standardized)
                                shape_type = self.detect_shape_type(standardized)
                                
                                # Save processed image to cache file
                                np.savez_compressed(
                                    os.path.join(self.cache_dir, f"{hashlib.md5(template_id.encode()).hexdigest()}.npz"),
                                    processed=standardized,
                                    original_shape=img.shape,
                                    hash=img_hash,
                                    shape_type=shape_type
                                )
                                
                                self.template_hashes[img_hash] = {
                                    'class': class_name,
                                    'filename': img_file,
                                    'path': img_path,
                                    'id': template_id
                                }
                                
                                # Track shape type distribution
                                self.shape_types[shape_type] = self.shape_types.get(shape_type, 0) + 1
                                
                                # Free memory
                                del img, standardized
                                gc.collect()
                        except Exception as e:
                            st.warning(f"Could not process {img_path}: {e}")
                            continue
                
                if class_images:
                    self.templates[class_name] = class_images
        
        self.template_count = sum(len(v) for v in self.templates.values())
        
        # Save metadata to cache
        try:
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.templates, f)
                pickle.dump(self.template_hashes, f)
                pickle.dump(self.shape_types, f)
        except:
            pass
        
        st.success(f"✅ Processed {self.template_count} templates in {time.time()-start_time:.2f} seconds")
        self.templates_loaded = True

    def load_template_image(self, template_info):
        """Load a specific template image from cache or file"""
        try:
            # Try to load from cache first
            cache_file = template_info.get('cached_file')
            if cache_file and os.path.exists(cache_file):
                data = np.load(cache_file)
                return {
                    'processed': data['processed'],
                    'shape_type': str(data['shape_type']),
                    'hash': int(data['hash'])
                }
            
            # Fallback: load from original file
            img = cv2.imread(template_info['path'], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                standardized = self.fast_normalize(img, preserve_aspect=True)
                return {
                    'processed': standardized,
                    'shape_type': self.detect_shape_type(standardized),
                    'hash': self.compute_image_hash(standardized)
                }
        except Exception as e:
            st.warning(f"Error loading template {template_info['filename']}: {e}")
        
        return None

    def compute_image_hash(self, image):
        """Compute a simple hash for quick comparison"""
        # Use smaller size for hash computation to save memory
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
        
        # Use adaptive thresholding for better contour detection
        if gray.size > 0:
            gray = cv2.medianBlur(gray, 5)
        
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
        
        # Extract the profile region with some padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(gray.shape[1] - x, w + 2 * padding)
        h = min(gray.shape[0] - y, h + 2 * padding)
        
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

    def detect_shape_type(self, image):
        """Detect if shape is primarily curved or straight-edged"""
        if image is None or image.size == 0:
            return 'unknown'
        
        try:
            _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 'unknown'
            
            cnt = max(contours, key=cv2.contourArea)
            
            # Calculate circularity
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            # Calculate straightness by approximating polygon
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            vertices = len(approx)
            
            # High circularity and few vertices indicate curved shape
            if circularity > 0.7 and vertices < 8:
                return 'curved'
            else:
                return 'straight'
        except:
            return 'unknown'

    def compare_shape_descriptors_lightweight(self, img1, img2):
        """Lightweight shape descriptor comparison"""
        try:
            # Use simpler metrics to save memory
            if img1 is None or img2 is None or img1.size == 0 or img2.size == 0:
                return 0.5
            
            # 1. Compare aspect ratios
            _, thresh1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
            _, thresh2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
            
            # Get bounding boxes
            contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours1 or not contours2:
                return 0.5
            
            cnt1 = max(contours1, key=cv2.contourArea)
            cnt2 = max(contours2, key=cv2.contourArea)
            
            rect1 = cv2.minAreaRect(cnt1)
            rect2 = cv2.minAreaRect(cnt2)
            
            # Get width and height from rect
            w1, h1 = rect1[1]
            w2, h2 = rect2[1]
            
            # Handle zero dimensions
            if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
                return 0.5
            
            aspect1 = max(w1, h1) / min(w1, h1)
            aspect2 = max(w2, h2) / min(w2, h2)
            
            aspect_similarity = 1.0 / (1.0 + abs(aspect1 - aspect2))
            
            # 2. Compare solidity (area/convex hull area)
            hull1 = cv2.convexHull(cnt1)
            hull2 = cv2.convexHull(cnt2)
            
            hull_area1 = cv2.contourArea(hull1)
            hull_area2 = cv2.contourArea(hull2)
            
            if hull_area1 > 0 and hull_area2 > 0:
                solidity1 = area1 / hull_area1 if 'area1' in locals() else cv2.contourArea(cnt1) / hull_area1
                solidity2 = area2 / hull_area2 if 'area2' in locals() else cv2.contourArea(cnt2) / hull_area2
                solidity_similarity = 1.0 / (1.0 + abs(solidity1 - solidity2))
            else:
                solidity_similarity = 0.5
            
            # Combine metrics
            return 0.7 * aspect_similarity + 0.3 * solidity_similarity
        except:
            return 0.5

    def fast_similarity_lightweight(self, img1, img2):
        """Memory-efficient similarity computation"""
        try:
            # First check hash for exact match
            hash1 = self.compute_image_hash(img1)
            hash2 = self.compute_image_hash(img2)
            
            if hash1 == hash2:
                return 1.0
            
            # Use smaller images for SSIM to save memory
            if max(img1.shape[0], img1.shape[1]) > 100:
                size = 100
                img1_small = cv2.resize(img1, (size, size))
                img2_small = cv2.resize(img2, (size, size))
                
                # Downsample for SSIM
                ssim_size = 64
                img1_ssim = cv2.resize(img1_small, (ssim_size, ssim_size))
                img2_ssim = cv2.resize(img2_small, (ssim_size, ssim_size))
                
                similarity = ssim(img1_ssim, img2_ssim, data_range=255)
            else:
                similarity = ssim(img1, img2, data_range=255)
            
            # Lightweight shape comparison
            shape_similarity = self.compare_shape_descriptors_lightweight(img1, img2)
            
            # Simple edge detection on small images
            edges1 = cv2.Canny(cv2.resize(img1, (64, 64)), 50, 150)
            edges2 = cv2.Canny(cv2.resize(img2, (64, 64)), 50, 150)
            
            if np.sum(edges1) > 0 and np.sum(edges2) > 0:
                edge_match = np.sum(edges1 & edges2) / max(np.sum(edges1), np.sum(edges2))
            else:
                edge_match = 0
            
            # Check shape types
            is_circular1 = self.detect_shape_type(img1) == 'curved'
            is_circular2 = self.detect_shape_type(img2) == 'curved'
            
            if is_circular1 != is_circular2:
                # Different shape types - emphasize shape similarity
                final_score = 0.2 * similarity + 0.1 * edge_match + 0.7 * shape_similarity
            else:
                # Same shape type
                final_score = 0.4 * similarity + 0.2 * edge_match + 0.4 * shape_similarity
            
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            # Fallback to simple similarity
            return 0.5

    def find_similar_profiles(self, user_image, max_matches=5):
        """Find similar profiles with memory optimization"""
        # Load metadata first
        self.load_templates_lazy()
        
        start_time = time.time()
        
        # Process user image
        if len(user_image.shape) == 3:
            user_gray = cv2.cvtColor(user_image, cv2.COLOR_BGR2GRAY)
        else:
            user_gray = user_image.copy()
        
        user_standardized = self.fast_normalize(user_gray, preserve_aspect=True)
        user_hash = self.compute_image_hash(user_standardized)
        user_shape_type = self.detect_shape_type(user_standardized)
        
        # Check for exact match
        exact_matches = []
        if user_hash in self.template_hashes:
            exact_match = self.template_hashes[user_hash]
            exact_matches.append({
                'similarity': 1.0,
                'class': exact_match['class'],
                'filename': exact_match['filename'],
                'path': exact_match['path'],
                'id': exact_match['id'],
                'is_exact_match': True,
                'shape_type': user_shape_type
            })
        
        if exact_matches:
            st.write(f"✅ Found exact match in {time.time()-start_time:.2f} seconds!")
            # Load the exact match image
            for match in exact_matches:
                template_data = self.load_template_image({'path': match['path'], 'filename': match['filename']})
                if template_data:
                    match['processed'] = template_data['processed']
            return user_standardized, exact_matches[:max_matches]
        
        matches = []
        processed_count = 0
        
        # Process templates in batches to manage memory
        batch_size = 20
        total_templates = self.template_count
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for class_name, template_list in self.templates.items():
            for template in template_list:
                processed_count += 1
                
                # Update progress
                if processed_count % 10 == 0:
                    progress = processed_count / total_templates
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {processed_count}/{total_templates} templates...")
                
                # Load template image
                template_data = self.load_template_image(template)
                if not template_data:
                    continue
                
                # Apply shape type penalty
                shape_penalty = 0.5 if user_shape_type != template_data['shape_type'] else 1.0
                
                # Compute similarity
                similarity = self.fast_similarity_lightweight(user_standardized, template_data['processed'])
                adjusted_similarity = similarity * shape_penalty
                
                matches.append({
                    'similarity': adjusted_similarity,
                    'raw_similarity': similarity,
                    'class': class_name,
                    'image_path': template['path'],
                    'processed': template_data['processed'],
                    'filename': template['filename'],
                    'shape_type': template_data['shape_type'],
                    'user_shape_type': user_shape_type,
                    'id': template['id']
                })
                
                # Clear template data to free memory
                del template_data
                
                # Force garbage collection periodically
                if processed_count % batch_size == 0:
                    gc.collect()
        
        progress_bar.empty()
        status_text.empty()
        
        # Sort matches
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Get top unique classes
        results = []
        seen_classes = set()
        for match in matches:
            if match['class'] not in seen_classes:
                results.append(match)
                seen_classes.add(match['class'])
                if len(results) >= max_matches:
                    break
        
        st.write(f"⏱️ Matching completed in {time.time()-start_time:.2f} seconds")
        
        # Show shape distribution
        if results:
            shape_counts = {}
            for result in results:
                shape_type = result.get('shape_type', 'unknown')
                shape_counts[shape_type] = shape_counts.get(shape_type, 0) + 1
            
            if shape_counts:
                shape_info = ", ".join([f"{k}: {v}" for k, v in shape_counts.items()])
                st.info(f"User shape: {user_shape_type}, Top matches: {shape_info}")
        
        # Clear memory
        gc.collect()
        
        return user_standardized, results

def display_results_with_selection(user_img, processed_user, matches):
    """Display results with memory-efficient approach"""
    st.subheader("📊 Profile Matching Results")
    
    # Display input images
    col1, col2 = st.columns(2)
    with col1:
        st.image(user_img, caption="Original Input", use_column_width=True)
    with col2:
        # Convert numpy array to PIL for display
        if isinstance(processed_user, np.ndarray):
            processed_pil = Image.fromarray(processed_user)
        else:
            processed_pil = processed_user
        st.image(processed_pil, caption="Normalized (Preserved Aspect Ratio)", use_column_width=True)
    
    # Show shape info
    if matches and 'user_shape_type' in matches[0]:
        user_shape = matches[0]['user_shape_type']
        shape_emoji = "🔵" if user_shape == 'curved' else "📏"
        st.info(f"{shape_emoji} Detected shape type: **{user_shape.capitalize()}** profile")
    
    # Initialize selection
    if 'selected_match_idx' not in st.session_state:
        st.session_state.selected_match_idx = 0
    
    # Display matches in tabs to save memory
    tab_names = [f"Match {i+1}" for i in range(len(matches))]
    tabs = st.tabs(tab_names)
    
    for idx, (tab, match) in enumerate(zip(tabs, matches)):
        with tab:
            col1, col2 = tab.columns([2, 1])
            
            with col1:
                # Display match image
                match_img = Image.fromarray(match['processed'])
                tab.image(match_img, caption=f"Match {idx+1}: {match['class']}", use_column_width=True)
            
            with col2:
                # Match info
                similarity = match['similarity']
                raw_similarity = match.get('raw_similarity', similarity)
                
                tab.metric("Similarity", f"{similarity:.3f}")
                if raw_similarity != similarity:
                    tab.caption(f"Raw: {raw_similarity:.3f}")
                
                tab.write(f"**Class:** {match['class']}")
                tab.write(f"**File:** {match['filename']}")
                
                shape_type = match.get('shape_type', 'unknown')
                shape_emoji = "🔵" if shape_type == 'curved' else "📏"
                tab.write(f"**Shape:** {shape_emoji} {shape_type.capitalize()}")
                
                if match.get('is_exact_match', False):
                    tab.success("🎯 Exact Match!")
                
                # Selection button
                if st.session_state.selected_match_idx == idx:
                    tab.success("✅ Currently Selected")
                else:
                    if tab.button(f"Select Match {idx+1}", key=f"select_{idx}"):
                        st.session_state.selected_match_idx = idx
                        st.rerun()
    
    # Show selected match details
    selected_idx = st.session_state.selected_match_idx
    if 0 <= selected_idx < len(matches):
        selected_match = matches[selected_idx]
        
        st.markdown("---")
        st.subheader(f"🏆 Selected Best Match: {selected_match['class']}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_img = Image.fromarray(selected_match['processed'])
            st.image(selected_img, caption=f"Best Match: {selected_match['class']}", use_column_width=True)
            
            # Show comparison with user image
            comparison_col1, comparison_col2 = st.columns(2)
            with comparison_col1:
                st.image(user_img, caption="Your Image", width=150)
            with comparison_col2:
                st.image(selected_img, caption="Match", width=150)
        
        with col2:
            st.markdown("**📊 Match Details**")
            st.write(f"**Rank:** {selected_idx + 1} of {len(matches)}")
            st.write(f"**Class:** {selected_match['class']}")
            st.write(f"**Similarity:** {selected_match['similarity']:.3f}")
            if selected_match.get('raw_similarity') and selected_match['raw_similarity'] != selected_match['similarity']:
                st.write(f"**Raw Similarity:** {selected_match['raw_similarity']:.3f}")
                st.caption("(Adjusted for shape compatibility)")
            
            shape_type = selected_match.get('shape_type', 'unknown')
            user_shape = selected_match.get('user_shape_type', 'unknown')
            
            if shape_type == user_shape:
                st.success(f"✅ Shape compatible ({shape_type})")
            else:
                st.warning(f"⚠️ Different shapes (You: {user_shape}, Match: {shape_type})")
            
            st.write(f"**Filename:** {selected_match['filename']}")
            
            if selected_match.get('is_exact_match', False):
                st.success("**Type:** Exact Match")
            else:
                st.info("**Type:** Similar Match")

def main():
    st.set_page_config(
        page_title="ALU SCAN - Memory Optimized",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 ALU SCAN - Memory Optimized")
    st.markdown("Upload an image to find similar aluminum profiles")
    
    # Memory usage warning
    with st.expander("⚠️ Memory Optimization Tips"):
        st.markdown("""
        **This version is optimized for memory usage:**
        1. **Lazy Loading**: Templates loaded only when needed
        2. **Caching**: Processed templates cached to disk
        3. **Batch Processing**: Templates processed in small batches
        4. **Downsampling**: Uses smaller images for computation
        5. **Garbage Collection**: Manual memory management
        
        **If still getting memory errors:**
        1. Reduce the number of templates
        2. Make template images smaller (max 500x500 pixels)
        3. Use PNG format instead of JPEG
        4. Restart the app if memory gets too high
        """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    max_matches = st.sidebar.slider("Matches to show", 1, 10, 3, 
                                   help="Fewer matches = less memory usage")
    
    # Clear cache button
    if st.sidebar.button("🗑️ Clear Cache"):
        cache_dir = ".matcher_cache"
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
            st.sidebar.success("Cache cleared!")
            st.rerun()
    
    # Initialize matcher
    if 'matcher' not in st.session_state:
        TEMPLATE_PATH = "trained_data"
        if os.path.exists(TEMPLATE_PATH):
            with st.spinner("Initializing matcher..."):
                st.session_state.matcher = OptimizedProfileMatcher(TEMPLATE_PATH)
            st.sidebar.success("Matcher initialized")
        else:
            st.error(f"❌ Template folder '{TEMPLATE_PATH}' not found!")
            st.info("Please create a 'trained_data' folder with your template images")
            return
    
    # File upload
    uploaded_file = st.file_uploader(
        "📤 Upload profile image", 
        type=['png', '.jpg', '.jpeg'],
        help="Supported formats: PNG, JPG, JPEG"
    )
    
    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        if st.button("🔍 Find Similar Profiles", type="primary"):
            with st.spinner("Analyzing and finding matches..."):
                try:
                    # Convert to OpenCV format
                    user_img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Find matches
                    processed_user, matches = st.session_state.matcher.find_similar_profiles(
                        user_img_cv, 
                        max_matches=max_matches
                    )
                    
                    # Convert to PIL for display
                    processed_user_pil = Image.fromarray(processed_user)
                    user_img_pil = image
                    
                    # Store in session state
                    st.session_state.processed_user = processed_user_pil
                    st.session_state.matches = matches
                    st.session_state.user_img_pil = user_img_pil
                    st.session_state.analysis_done = True
                    
                    # Force garbage collection
                    gc.collect()
                    
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    st.info("Try uploading a smaller image or reducing template count")
    
    # Display results if available
    if st.session_state.get('analysis_done', False):
        display_results_with_selection(
            st.session_state.user_img_pil,
            st.session_state.processed_user,
            st.session_state.matches
        )
        
        # Clear memory button
        if st.button("🧹 Clear Memory"):
            keys_to_clear = ['processed_user', 'matches', 'user_img_pil', 'analysis_done']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            gc.collect()
            st.success("Memory cleared! You can upload a new image.")
            st.rerun()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**ℹ️ About this app:**")
    st.sidebar.markdown("""
    - **Memory Optimized** version
    - **Shape-Aware** matching
    - **Template Caching** for speed
    - **Batch Processing** to reduce memory
    """)

if __name__ == "__main__":
    main()
