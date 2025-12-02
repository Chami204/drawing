import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import streamlit as st
from PIL import Image
import time
import math
import pandas as pd
from scipy import ndimage

class EnhancedProfileMatcher:
    def __init__(self, template_root):
        self.template_root = template_root
        self.templates = {}
        self.reference_size = 300
        self.pixels_to_mm_ratio = None
        self.shape_features_cache = {}
        self.original_images = {}  # Store original reference images
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
                
                # Look for original image first
                original_img = None
                for img_file in os.listdir(class_path):
                    if 'original' in img_file.lower() and img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        original_path = os.path.join(class_path, img_file)
                        original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
                        self.original_images[class_name] = original_img
                        break
                
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            standardized, measurements, shape_features = self.scale_normalize_with_analysis(img)
                            class_images.append({
                                'original': img,
                                'standardized': standardized,
                                'filename': os.path.basename(img_path),
                                'class': class_name,
                                'measurements': measurements,
                                'shape_features': shape_features,
                                'is_original': 'original' in img_file.lower()
                            })
                if class_images:
                    self.templates[class_name] = class_images

        st.success(f"✅ Loaded {sum(len(v) for v in self.templates.values())} templates from {len(self.templates)} classes in {time.time()-start_time:.2f} seconds")

    def detect_profile_contour(self, image):
        """Enhanced contour detection with better thresholding"""
        # Use adaptive thresholding for better edge detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback to simple thresholding
            _, thresh_simple = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh_simple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Get the largest contour
        main_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour to reduce noise
        epsilon = 0.001 * cv2.arcLength(main_contour, True)
        approximated = cv2.approxPolyDP(main_contour, epsilon, True)
        
        return approximated

    def analyze_shape_features(self, contour):
        """Enhanced shape analysis for better differentiation"""
        if contour is None or len(contour) < 5:
            return {
                'circularity': 0,
                'rectangularity': 0,
                'angularity': 0,
                'curvature_index': 0,
                'straight_edge_ratio': 0,
                'corner_count': 0,
                'symmetry_score': 0,
                'compactness': 0
            }
        
        # Basic measurements
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity
        circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # Bounding rectangle analysis
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)
        rectangularity = area / box_area if box_area > 0 else 0
        
        # Corner detection using Harris or approximation
        corner_count = self.detect_corners(contour)
        
        # Curvature analysis
        curvature_index = self.calculate_curvature_index(contour)
        
        # Straight edge detection
        straight_edge_ratio = self.detect_straight_edges(contour)
        
        # Angularity (how angular vs curved)
        angularity = corner_count / (len(contour) / 10) if len(contour) > 0 else 0
        
        # Symmetry analysis
        symmetry_score = self.calculate_symmetry(contour)
        
        # Compactness
        compactness = (perimeter ** 2) / (4 * math.pi * area) if area > 0 else 0
        
        return {
            'circularity': circularity,  # 1.0 = perfect circle
            'rectangularity': rectangularity,  # 1.0 = perfect rectangle
            'angularity': min(angularity, 1.0),  # Higher = more angular
            'curvature_index': curvature_index,  # Higher = more curved
            'straight_edge_ratio': straight_edge_ratio,  # Ratio of straight edges
            'corner_count': corner_count,
            'symmetry_score': symmetry_score,
            'compactness': compactness,
            'shape_type': self.classify_shape(circularity, rectangularity, angularity, curvature_index)
        }
    
    def detect_corners(self, contour):
        """Detect corners using Douglas-Peucker algorithm"""
        if len(contour) < 5:
            return 0
            
        # Simplify contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Count vertices
        return len(approx)
    
    def calculate_curvature_index(self, contour):
        """Calculate how curved the shape is"""
        if len(contour) < 10:
            return 0
            
        contour = contour.reshape(-1, 2)
        curvatures = []
        
        for i in range(1, len(contour)-1):
            p1 = contour[i-1]
            p2 = contour[i]
            p3 = contour[i+1]
            
            # Calculate curvature using cross product method
            v1 = p2 - p1
            v2 = p3 - p2
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                # Normalize vectors
                v1_norm = v1 / np.linalg.norm(v1)
                v2_norm = v2 / np.linalg.norm(v2)
                
                # Calculate angle between vectors
                dot_product = np.dot(v1_norm, v2_norm)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                curvatures.append(angle)
        
        if curvatures:
            avg_curvature = np.mean(curvatures)
            # Normalize to 0-1 range
            return min(avg_curvature / (math.pi/2), 1.0)
        return 0
    
    def detect_straight_edges(self, contour):
        """Detect ratio of straight edges in contour"""
        if len(contour) < 10:
            return 0
            
        contour = contour.reshape(-1, 2)
        straight_segments = 0
        total_segments = len(contour) - 1
        
        for i in range(len(contour)-2):
            p1 = contour[i]
            p2 = contour[i+1]
            p3 = contour[i+2]
            
            # Check if points are approximately collinear
            area = abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2)
            if area < 2:  # Threshold for collinearity
                straight_segments += 1
        
        return straight_segments / total_segments if total_segments > 0 else 0
    
    def calculate_symmetry(self, contour):
        """Calculate symmetry score of the shape"""
        if len(contour) < 10:
            return 0
            
        contour = contour.reshape(-1, 2)
        
        # Find centroid
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return 0
            
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Calculate distances from centroid
        distances = [np.linalg.norm(np.array([cx, cy]) - np.array(p)) for p in contour]
        
        # Check symmetry by comparing opposite points
        symmetry_errors = []
        for i in range(len(contour)):
            opposite_idx = (i + len(contour)//2) % len(contour)
            dist1 = distances[i]
            dist2 = distances[opposite_idx]
            symmetry_errors.append(abs(dist1 - dist2) / max(dist1, dist2) if max(dist1, dist2) > 0 else 0)
        
        avg_error = np.mean(symmetry_errors) if symmetry_errors else 0
        return 1 - min(avg_error, 1.0)
    
    def classify_shape(self, circularity, rectangularity, angularity, curvature_index):
        """Classify shape based on features"""
        if circularity > 0.9:
            return "Circle/Ellipse"
        elif rectangularity > 0.85:
            return "Rectangle"
        elif angularity > 0.7:
            return "Angular/Polygonal"
        elif curvature_index > 0.6:
            return "Curved/Organic"
        else:
            return "Complex/Mixed"

    def calculate_measurements(self, contour, scale_factor=1.0):
        """Enhanced measurements with shape-specific calculations"""
        if contour is None:
            return {}
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Basic measurements
        height_px = h
        width_px = w
        perimeter_px = cv2.arcLength(contour, True)
        area_px = cv2.contourArea(contour)
        
        # Aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        
        # Compactness
        compactness = (perimeter_px ** 2) / (4 * math.pi * area_px) if area_px > 0 else 0
        
        # Find extreme points for face measurements
        contour_points = contour.reshape(-1, 2)
        
        # Find nose (leftmost point in upper half)
        upper_half = [p for p in contour_points if p[1] < y + h/2]
        if upper_half:
            nose_point = min(upper_half, key=lambda p: p[0])
        else:
            nose_point = min(contour_points, key=lambda p: p[0])
        
        # Find chin (lowest point)
        chin_point = max(contour_points, key=lambda p: p[1])
        
        # Face height
        face_height_px = abs(chin_point[1] - nose_point[1])
        
        # Calculate major and minor axes for ellipses
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            ellipse_angle = ellipse[2]
        else:
            major_axis = max(w, h)
            minor_axis = min(w, h)
            ellipse_angle = 0
        
        # Apply scale conversion
        if self.pixels_to_mm_ratio:
            height_mm = height_px * self.pixels_to_mm_ratio
            width_mm = width_px * self.pixels_to_mm_ratio
            perimeter_mm = perimeter_px * self.pixels_to_mm_ratio
            area_mm2 = area_px * (self.pixels_to_mm_ratio ** 2)
            face_height_mm = face_height_px * self.pixels_to_mm_ratio
            major_axis_mm = major_axis * self.pixels_to_mm_ratio
            minor_axis_mm = minor_axis * self.pixels_to_mm_ratio
        else:
            height_mm = width_mm = perimeter_mm = area_mm2 = face_height_mm = None
            major_axis_mm = minor_axis_mm = None
        
        measurements = {
            'pixels': {
                'height': height_px,
                'width': width_px,
                'perimeter': perimeter_px,
                'area': area_px,
                'face_height': face_height_px,
                'aspect_ratio': aspect_ratio,
                'compactness': compactness,
                'major_axis': major_axis,
                'minor_axis': minor_axis,
                'ellipse_angle': ellipse_angle
            },
            'millimeters': {
                'height': height_mm,
                'width': width_mm,
                'perimeter': perimeter_mm,
                'area': area_mm2,
                'face_height': face_height_mm,
                'major_axis': major_axis_mm,
                'minor_axis': minor_axis_mm
            } if self.pixels_to_mm_ratio else None
        }
        
        return measurements

    def scale_normalize_with_analysis(self, image):
        """Enhanced normalization with aspect ratio preservation"""
        contour = self.detect_profile_contour(image)
        
        # Calculate measurements and shape features
        measurements = self.calculate_measurements(contour)
        shape_features = self.analyze_shape_features(contour)
        
        if contour is None:
            standardized = cv2.resize(image, (self.reference_size, self.reference_size))
            return standardized, measurements, shape_features

        x, y, w, h = cv2.boundingRect(contour)
        profile_region = image[y:y+h, x:x+w]
        
        # Preserve aspect ratio better
        target_ratio = self.reference_size / max(w, h)
        new_width = int(w * target_ratio)
        new_height = int(h * target_ratio)
        
        # Resize
        resized = cv2.resize(profile_region, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create new canvas
        standardized = np.ones((self.reference_size, self.reference_size), dtype=np.uint8) * 255
        
        # Center the resized image
        start_x = (self.reference_size - new_width) // 2
        start_y = (self.reference_size - new_height) // 2
        standardized[start_y:start_y+new_height, start_x:start_x+new_width] = resized
        
        return standardized, measurements, shape_features

    def preprocess_user_image(self, image):
        """Enhanced preprocessing with shape analysis"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        standardized, measurements, shape_features = self.scale_normalize_with_analysis(gray)
        return standardized, measurements, shape_features

    def enhanced_similarity(self, user_img, template_img, user_shape_features, template_shape_features):
        """Enhanced similarity calculation with shape features"""
        # Base SSIM score
        ssim_score = ssim(user_img, template_img, full=True)[0]
        
        # Shape feature similarity
        shape_similarity = self.calculate_shape_similarity(user_shape_features, template_shape_features)
        
        # Combine scores (weighted average)
        final_score = 0.7 * ssim_score + 0.3 * shape_similarity
        
        return final_score
    
    def calculate_shape_similarity(self, features1, features2):
        """Calculate similarity between shape features"""
        if not features1 or not features2:
            return 0.5
        
        # Compare key shape features
        comparisons = []
        
        # Circularity similarity
        circ_diff = 1 - abs(features1.get('circularity', 0) - features2.get('circularity', 0))
        comparisons.append(circ_diff)
        
        # Angularity similarity
        ang_diff = 1 - abs(features1.get('angularity', 0) - features2.get('angularity', 0))
        comparisons.append(ang_diff)
        
        # Curvature similarity
        curv_diff = 1 - abs(features1.get('curvature_index', 0) - features2.get('curvature_index', 0))
        comparisons.append(curv_diff)
        
        # Straight edge ratio similarity
        straight_diff = 1 - abs(features1.get('straight_edge_ratio', 0) - features2.get('straight_edge_ratio', 0))
        comparisons.append(straight_diff)
        
        # Corner count similarity (normalized)
        max_corners = max(features1.get('corner_count', 0), features2.get('corner_count', 0))
        if max_corners > 0:
            corner_diff = 1 - abs(features1.get('corner_count', 0) - features2.get('corner_count', 0)) / max_corners
        else:
            corner_diff = 1
        comparisons.append(corner_diff)
        
        # Average all comparisons
        return np.mean(comparisons)

    def find_similar_profiles(self, user_image, max_matches=5):
        """Enhanced matching with shape features"""
        # Ensure templates are loaded
        self.load_templates()
        
        start_time = time.time()
        processed_user, user_measurements, user_shape_features = self.preprocess_user_image(user_image)

        matches = []
        for class_name, template_list in self.templates.items():
            for template in template_list:
                # Use enhanced similarity
                similarity = self.enhanced_similarity(
                    processed_user, 
                    template['standardized'],
                    user_shape_features,
                    template['shape_features']
                )
                
                matches.append({
                    'similarity': similarity,
                    'class': class_name,
                    'image': template['original'],
                    'processed': template['standardized'],
                    'filename': template['filename'],
                    'measurements': template['measurements'],
                    'shape_features': template['shape_features'],
                    'is_original': template.get('is_original', False)
                })

        matches.sort(key=lambda x: x['similarity'], reverse=True)

        # Get best matches
        results = []
        seen_classes = set()
        for match in matches:
            if match['class'] not in seen_classes:
                results.append(match)
                seen_classes.add(match['class'])
                if len(results) >= max_matches:
                    break

        st.write(f"⏱️ Matching completed in {time.time()-start_time:.2f} seconds")
        return processed_user, results, user_measurements, user_shape_features

def display_shape_analysis(shape_features, title="Shape Analysis"):
    """Display detailed shape analysis"""
    st.subheader(f"🔷 {title}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📐 Shape Characteristics:**")
        
        # Shape type
        shape_type = shape_features.get('shape_type', 'Unknown')
        st.metric("Shape Type", shape_type)
        
        # Circularity gauge
        circularity = shape_features.get('circularity', 0)
        st.progress(circularity)
        st.caption(f"Circularity: {circularity:.3f} (1.0 = perfect circle)")
        
        # Rectangularity gauge
        rectangularity = shape_features.get('rectangularity', 0)
        st.progress(rectangularity)
        st.caption(f"Rectangularity: {rectangularity:.3f} (1.0 = perfect rectangle)")
    
    with col2:
        st.markdown("**📊 Shape Metrics:**")
        
        st.write(f"• **Angularity**: {shape_features.get('angularity', 0):.3f}")
        st.write(f"• **Curvature Index**: {shape_features.get('curvature_index', 0):.3f}")
        st.write(f"• **Straight Edge Ratio**: {shape_features.get('straight_edge_ratio', 0):.3f}")
        st.write(f"• **Corner Count**: {shape_features.get('corner_count', 0)}")
        st.write(f"• **Symmetry Score**: {shape_features.get('symmetry_score', 0):.3f}")
        st.write(f"• **Compactness**: {shape_features.get('compactness', 0):.3f}")

def display_measurements_with_comparison(user_measurements, match_measurements, title="Detailed Measurements"):
    """Display measurements with comparison"""
    st.subheader(f"📏 {title}")
    
    # Create comparison table
    comparison_data = []
    user_pixels = user_measurements['pixels']
    match_pixels = match_measurements['pixels']
    
    metrics = [
        ('Height', 'height', 'px'),
        ('Width', 'width', 'px'),
        ('Area', 'area', 'px²'),
        ('Perimeter', 'perimeter', 'px'),
        ('Face Height', 'face_height', 'px'),
        ('Aspect Ratio', 'aspect_ratio', ''),
        ('Compactness', 'compactness', ''),
        ('Major Axis', 'major_axis', 'px'),
        ('Minor Axis', 'minor_axis', 'px')
    ]
    
    for display_name, key, unit in metrics:
        user_val = user_pixels.get(key, 0)
        match_val = match_pixels.get(key, 0)
        
        if user_val and match_val:
            diff = user_val - match_val
            diff_percent = (diff / match_val * 100) if match_val != 0 else 0
            
            comparison_data.append({
                'Metric': display_name,
                'Your Value': f"{user_val:.1f} {unit}",
                'Match Value': f"{match_val:.1f} {unit}",
                'Difference': f"{diff:+.1f} {unit}",
                'Difference %': f"{diff_percent:+.1f}%"
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        st.table(df)
    
    # Display shape comparison
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📐 Your Shape Characteristics:**")
        for key in ['aspect_ratio', 'compactness']:
            if key in user_pixels:
                st.write(f"• {key.replace('_', ' ').title()}: {user_pixels[key]:.3f}")
    
    with col2:
        st.markdown("**📐 Match Shape Characteristics:**")
        for key in ['aspect_ratio', 'compactness']:
            if key in match_pixels:
                st.write(f"• {key.replace('_', ' ').title()}: {match_pixels[key]:.3f}")

def display_interactive_results(user_img, processed_user, matches, user_measurements, user_shape_features):
    """Display interactive results with selection capability"""
    
    st.subheader("📊 Profile Matching Results")
    
    # Display input images
    col1, col2 = st.columns(2)
    with col1:
        st.image(user_img, caption="Original Input", use_column_width=True)
    with col2:
        st.image(processed_user, caption="Normalized Input", use_column_width=True)
    
    # Display user shape analysis
    display_shape_analysis(user_shape_features, "Your Profile Shape Analysis")
    
    # Display user measurements
    st.subheader("📏 Your Profile Measurements")
    display_measurements_simple(user_measurements)
    
    # Interactive match selection
    st.subheader(f"🏆 Top {len(matches)} Matches")
    st.info("Click on an image to select it for detailed comparison")
    
    # Store selection in session state
    if 'selected_match' not in st.session_state:
        st.session_state.selected_match = None
    
    # Display matches in columns
    cols = st.columns(len(matches))
    selected_idx = None
    
    for idx, (col, match) in enumerate(zip(cols, matches)):
        with col:
            # Convert to PIL for display
            match_img = Image.fromarray(match['processed'])
            
            # Create a button for each match
            if col.button(f"Select Match {idx+1}", key=f"select_{idx}"):
                st.session_state.selected_match = match
            
            st.image(match_img, use_column_width=True)
            
            # Highlight if selected
            if st.session_state.selected_match and st.session_state.selected_match['filename'] == match['filename']:
                selected_idx = idx
                st.success(f"✅ Selected: {match['class']}")
            
            st.metric(
                label=f"Match {idx+1}: {match['class']}",
                value=f"{match['similarity']:.3f}"
            )
            st.caption(f"File: {match['filename']}")
            st.caption(f"Shape: {match['shape_features'].get('shape_type', 'Unknown')}")
    
    # If a match is selected, show detailed comparison
    if st.session_state.selected_match:
        st.markdown("---")
        st.subheader("🎯 Selected Match Detailed Analysis")
        
        selected_match = st.session_state.selected_match
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(Image.fromarray(selected_match['processed']), 
                    caption=f"Selected: {selected_match['class']}", 
                    use_column_width=True)
            
            st.metric(
                label=f"Similarity Score",
                value=f"{selected_match['similarity']:.3f}",
                delta=f"Rank: {selected_idx+1 if selected_idx is not None else 'N/A'}"
            )
        
        with col2:
            # Display shape analysis of selected match
            display_shape_analysis(selected_match['shape_features'], 
                                 f"Selected Match Shape Analysis")
        
        # Display detailed measurements comparison
        display_measurements_with_comparison(user_measurements, 
                                           selected_match['measurements'],
                                           "Detailed Measurement Comparison")
        
        # Show if this is the original reference image
        if selected_match.get('is_original', False):
            st.success("🎉 This is the ORIGINAL reference image for this class!")
        
        # Load and display original reference image if available
        class_name = selected_match['class']
        if hasattr(st.session_state.matcher, 'original_images') and class_name in st.session_state.matcher.original_images:
            st.subheader("📁 Original Reference Image")
            original_img = st.session_state.matcher.original_images[class_name]
            original_pil = Image.fromarray(original_img)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_pil, caption=f"Original Reference: {class_name}", use_column_width=True)
            
            with col2:
                # Show original vs selected comparison
                selected_pil = Image.fromarray(selected_match['processed'])
                st.image(selected_pil, caption=f"Selected Match", use_column_width=True)
                
                # Calculate similarity between selected and original
                if selected_match['filename'] != 'original':
                    # Convert to arrays for comparison
                    selected_array = np.array(selected_pil.convert('L'))
                    original_resized = cv2.resize(original_img, (selected_array.shape[1], selected_array.shape[0]))
                    
                    similarity = ssim(selected_array, original_resized, full=True)[0]
                    st.metric("Similarity to Original", f"{similarity:.3f}")
    
    # Display all matches in a table
    st.subheader("📋 All Matches Summary")
    
    results_data = []
    for i, match in enumerate(matches, 1):
        shape_type = match['shape_features'].get('shape_type', 'Unknown')
        results_data.append({
            "Rank": i,
            "Class": match['class'],
            "Similarity": f"{match['similarity']:.3f}",
            "Shape Type": shape_type,
            "Circularity": f"{match['shape_features'].get('circularity', 0):.3f}",
            "Angularity": f"{match['shape_features'].get('angularity', 0):.3f}",
            "Selected": "✅" if st.session_state.selected_match and 
                             st.session_state.selected_match['filename'] == match['filename'] else ""
        })
    
    df = pd.DataFrame(results_data)
    st.dataframe(df, use_container_width=True)

def display_measurements_simple(measurements, title="Measurements"):
    """Display basic measurements"""
    st.write(f"**{title}:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📐 Pixel Measurements:**")
        pixels = measurements['pixels']
        st.write(f"• Height: {pixels['height']:.1f} px")
        st.write(f"• Width: {pixels['width']:.1f} px")
        st.write(f"• Area: {pixels['area']:.1f} px²")
        st.write(f"• Perimeter: {pixels['perimeter']:.1f} px")
    
    with col2:
        if measurements['millimeters']:
            st.markdown("**📏 Real-world Measurements:**")
            mm = measurements['millimeters']
            st.write(f"• Height: {mm['height']:.1f} mm")
            st.write(f"• Width: {mm['width']:.1f} mm")
            st.write(f"• Area: {mm['area']:.1f} mm²")
        else:
            st.info("Set pixels-to-mm ratio in sidebar for real measurements")

def main():
    st.set_page_config(
        page_title="Enhanced Profile Matcher",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Enhanced Profile Image Matching System")
    st.markdown("Upload a profile image to find similar matches with detailed shape analysis and measurements.")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    max_matches = st.sidebar.slider("Maximum matches", 3, 10, 5)
    
    st.sidebar.header("📏 Measurement Settings")
    pixels_to_mm = st.sidebar.number_input(
        "Pixels to mm ratio", 
        min_value=0.0, 
        max_value=10.0, 
        value=0.264,  # Default: 0.264 mm per pixel (approx 96 DPI)
        help="Set scale for real measurements (e.g., 0.264 = ~96 DPI)"
    )
    
    st.sidebar.header("🔧 Advanced Settings")
    enable_shape_analysis = st.sidebar.checkbox("Enable enhanced shape analysis", value=True)
    show_original_images = st.sidebar.checkbox("Show original reference images", value=True)
    
    # Initialize matcher only once using session state
    if 'matcher' not in st.session_state:
        TEMPLATE_PATH = "trained_data"
        st.session_state.matcher = EnhancedProfileMatcher(TEMPLATE_PATH)
        st.info("🔧 Enhanced profile matcher initialized. Ready to load templates when needed.")
    
    # Apply settings
    if pixels_to_mm > 0:
        st.session_state.matcher.pixels_to_mm_ratio = pixels_to_mm
    
    # File upload
    st.header("📤 Upload Profile Image")
    
    uploaded_file = st.file_uploader(
        "Choose a profile image", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a profile image for analysis"
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
                        # Convert to OpenCV format
                        user_img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        
                        # Find matches
                        processed_user, matches, user_measurements, user_shape_features = \
                            st.session_state.matcher.find_similar_profiles(
                                user_img_cv, 
                                max_matches=max_matches
                            )
                        
                        # Convert for display
                        processed_user_pil = Image.fromarray(processed_user)
                        user_img_pil = Image.fromarray(cv2.cvtColor(user_img_cv, cv2.COLOR_BGR2RGB))
                        
                        # Display interactive results
                        display_interactive_results(
                            user_img_pil, 
                            processed_user_pil, 
                            matches, 
                            user_measurements,
                            user_shape_features
                        )
                        
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
            
            # Show shape distribution
            if st.session_state.matcher.templates:
                shape_counts = {}
                for class_name, templates in st.session_state.matcher.templates.items():
                    for template in templates:
                        shape_type = template['shape_features'].get('shape_type', 'Unknown')
                        shape_counts[shape_type] = shape_counts.get(shape_type, 0) + 1
                
                if shape_counts:
                    st.write("**Shape Distribution:**")
                    for shape_type, count in shape_counts.items():
                        st.write(f"• {shape_type}: {count}")
        else:
            st.info("📁 Templates not loaded yet - will load when you analyze an image")

    # Instructions
    with st.expander("ℹ️ How to use this enhanced system"):
        st.markdown("""
        **🎯 Enhanced Features:**
        1. **Better Shape Detection**: Differentiates circles, rectangles, curved vs straight shapes
        2. **Interactive Selection**: Click on any match to get detailed comparison
        3. **Shape Analysis**: Shows circularity, angularity, curvature, and more
        4. **Original Reference**: Automatically finds and shows original reference images
        5. **Detailed Measurements**: Comprehensive comparison with differences
        
        **🔍 Shape Detection Capabilities:**
        - **Circularity**: How close to a perfect circle (0-1)
        - **Angularity**: How many corners/angles
        - **Curvature Index**: How curved vs straight
        - **Straight Edge Ratio**: Percentage of straight edges
        - **Symmetry Score**: How symmetrical the shape is
        
        **📊 For Best Results:**
        - Ensure good contrast between profile and background
        - Use clear, well-defined images
        - Set correct pixels-to-mm ratio for accurate measurements
        - Look for the "Original" image marker for reference quality
        """)

if __name__ == "__main__":
    main()
