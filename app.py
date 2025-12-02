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
        self.original_images = {}

    def load_templates(self):
        """Pre-load all template images - only when needed"""
        if self.templates:
            return
            
        st.write("📂 Loading templates...")
        start_time = time.time()

        for class_name in os.listdir(self.template_root):
            class_path = os.path.join(self.template_root, class_name)
            if os.path.isdir(class_path):
                class_images = []
                
                # Look for original image
                for img_file in os.listdir(class_path):
                    if 'original' in img_file.lower() and img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        original_path = os.path.join(class_path, img_file)
                        original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
                        if original_img is not None:
                            self.original_images[class_name] = original_img
                        break
                
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            try:
                                standardized, measurements, shape_features = self.scale_normalize_with_analysis(img)
                                class_images.append({
                                    'original': img,
                                    'standardized': standardized,
                                    'filename': os.path.basename(img_path),
                                    'class': class_name,
                                    'measurements': measurements,
                                    'shape_features': shape_features or {},
                                    'is_original': 'original' in img_file.lower()
                                })
                            except Exception as e:
                                st.warning(f"Could not process {img_file}: {str(e)}")
                                continue
                
                if class_images:
                    self.templates[class_name] = class_images

        st.success(f"✅ Loaded {sum(len(v) for v in self.templates.values())} templates from {len(self.templates)} classes in {time.time()-start_time:.2f} seconds")

    def detect_profile_contour(self, image):
        """Enhanced contour detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try adaptive thresholding
        try:
            thresh = cv2.adaptiveThreshold(blurred, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
        except:
            # Fallback to simple thresholding
            _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        main_contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour
        epsilon = 0.001 * cv2.arcLength(main_contour, True)
        approximated = cv2.approxPolyDP(main_contour, epsilon, True)
        
        return approximated

    def analyze_shape_features(self, contour):
        """Enhanced shape analysis with error handling"""
        if contour is None or len(contour) < 3:
            return self.get_default_shape_features()
        
        try:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Circularity
            circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            # Bounding rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box_area = cv2.contourArea(box)
            rectangularity = area / box_area if box_area > 0 else 0
            
            # Corner detection
            epsilon = 0.02 * perimeter if perimeter > 0 else 0.1
            approx = cv2.approxPolyDP(contour, epsilon, True)
            corner_count = len(approx)
            
            # Curvature analysis
            curvature_index = self.calculate_curvature_index(contour)
            
            # Straight edge detection
            straight_edge_ratio = self.detect_straight_edges(contour)
            
            # Angularity
            angularity = corner_count / max(len(contour) / 10, 1)
            
            # Symmetry
            symmetry_score = self.calculate_symmetry(contour)
            
            # Compactness
            compactness = (perimeter ** 2) / (4 * math.pi * area) if area > 0 else 0
            
            # Shape classification
            shape_type = self.classify_shape(circularity, rectangularity, angularity, curvature_index)
            
            return {
                'circularity': min(circularity, 1.0),
                'rectangularity': min(rectangularity, 1.0),
                'angularity': min(angularity, 1.0),
                'curvature_index': min(curvature_index, 1.0),
                'straight_edge_ratio': min(straight_edge_ratio, 1.0),
                'corner_count': corner_count,
                'symmetry_score': min(symmetry_score, 1.0),
                'compactness': compactness,
                'shape_type': shape_type
            }
        except Exception as e:
            st.warning(f"Shape analysis error: {str(e)}")
            return self.get_default_shape_features()
    
    def get_default_shape_features(self):
        """Return default shape features when analysis fails"""
        return {
            'circularity': 0,
            'rectangularity': 0,
            'angularity': 0,
            'curvature_index': 0,
            'straight_edge_ratio': 0,
            'corner_count': 0,
            'symmetry_score': 0,
            'compactness': 0,
            'shape_type': 'Unknown'
        }
    
    def calculate_curvature_index(self, contour):
        """Calculate curvature index"""
        if len(contour) < 10:
            return 0
        
        try:
            contour = contour.reshape(-1, 2)
            curvatures = []
            
            for i in range(1, len(contour)-1):
                p1 = contour[i-1]
                p2 = contour[i]
                p3 = contour[i+1]
                
                v1 = p2 - p1
                v2 = p3 - p2
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    v1_norm = v1 / np.linalg.norm(v1)
                    v2_norm = v2 / np.linalg.norm(v2)
                    
                    dot_product = np.dot(v1_norm, v2_norm)
                    dot_product = np.clip(dot_product, -1.0, 1.0)
                    angle = np.arccos(dot_product)
                    curvatures.append(angle)
            
            if curvatures:
                avg_curvature = np.mean(curvatures)
                return min(avg_curvature / (math.pi/2), 1.0)
        except:
            pass
        
        return 0
    
    def detect_straight_edges(self, contour):
        """Detect straight edges"""
        if len(contour) < 10:
            return 0
        
        try:
            contour = contour.reshape(-1, 2)
            straight_segments = 0
            total_segments = max(len(contour) - 2, 1)
            
            for i in range(len(contour)-2):
                p1 = contour[i]
                p2 = contour[i+1]
                p3 = contour[i+2]
                
                # Check collinearity
                area = abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2)
                if area < 2:
                    straight_segments += 1
            
            return straight_segments / total_segments
        except:
            return 0
    
    def calculate_symmetry(self, contour):
        """Calculate symmetry score"""
        if len(contour) < 10:
            return 0
        
        try:
            contour = contour.reshape(-1, 2)
            
            # Find centroid
            M = cv2.moments(contour)
            if M['m00'] == 0:
                return 0
            
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Calculate distances
            centroid = np.array([cx, cy])
            distances = [np.linalg.norm(p - centroid) for p in contour]
            
            # Check symmetry
            symmetry_errors = []
            n = len(contour)
            for i in range(n):
                opposite_idx = (i + n//2) % n
                if distances[i] > 0 or distances[opposite_idx] > 0:
                    error = abs(distances[i] - distances[opposite_idx]) / max(distances[i], distances[opposite_idx])
                    symmetry_errors.append(error)
            
            if symmetry_errors:
                avg_error = np.mean(symmetry_errors)
                return 1 - min(avg_error, 1.0)
        except:
            pass
        
        return 0
    
    def classify_shape(self, circularity, rectangularity, angularity, curvature_index):
        """Classify shape type"""
        if circularity > 0.9:
            return "Circle/Ellipse"
        elif rectangularity > 0.85:
            return "Rectangle/Square"
        elif angularity > 0.7:
            return "Angular/Polygonal"
        elif curvature_index > 0.6:
            return "Curved/Organic"
        else:
            return "Complex/Mixed"

    def calculate_measurements(self, contour, scale_factor=1.0):
        """Calculate measurements with error handling"""
        if contour is None:
            return self.get_default_measurements()
        
        try:
            x, y, w, h = cv2.boundingRect(contour)
            
            height_px = h
            width_px = w
            perimeter_px = cv2.arcLength(contour, True)
            area_px = cv2.contourArea(contour)
            
            aspect_ratio = w / h if h > 0 else 0
            compactness = (perimeter_px ** 2) / (4 * math.pi * area_px) if area_px > 0 else 0
            
            # Find face features
            contour_points = contour.reshape(-1, 2)
            
            # Nose point
            upper_half = [p for p in contour_points if p[1] < y + h/2]
            nose_point = min(upper_half, key=lambda p: p[0]) if upper_half else min(contour_points, key=lambda p: p[0])
            
            # Chin point
            chin_point = max(contour_points, key=lambda p: p[1])
            face_height_px = abs(chin_point[1] - nose_point[1])
            
            # Ellipse fitting
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                ellipse_angle = ellipse[2]
            else:
                major_axis = max(w, h)
                minor_axis = min(w, h)
                ellipse_angle = 0
            
            # Scale conversion
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
            
            return {
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
        except Exception as e:
            st.warning(f"Measurement calculation error: {str(e)}")
            return self.get_default_measurements()
    
    def get_default_measurements(self):
        """Return default measurements"""
        return {
            'pixels': {
                'height': 0,
                'width': 0,
                'perimeter': 0,
                'area': 0,
                'face_height': 0,
                'aspect_ratio': 0,
                'compactness': 0,
                'major_axis': 0,
                'minor_axis': 0,
                'ellipse_angle': 0
            },
            'millimeters': None
        }

    def scale_normalize_with_analysis(self, image):
        """Scale and normalize image with analysis"""
        contour = self.detect_profile_contour(image)
        
        # Get measurements and shape features
        measurements = self.calculate_measurements(contour)
        shape_features = self.analyze_shape_features(contour)
        
        if contour is None:
            standardized = cv2.resize(image, (self.reference_size, self.reference_size))
            return standardized, measurements, shape_features

        try:
            x, y, w, h = cv2.boundingRect(contour)
            profile_region = image[y:y+h, x:x+w]
            
            # Scale preserving aspect ratio
            target_ratio = self.reference_size / max(w, h)
            new_width = int(w * target_ratio)
            new_height = int(h * target_ratio)
            
            # Resize
            resized = cv2.resize(profile_region, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Create canvas
            standardized = np.ones((self.reference_size, self.reference_size), dtype=np.uint8) * 255
            
            # Center the image
            start_x = (self.reference_size - new_width) // 2
            start_y = (self.reference_size - new_height) // 2
            standardized[start_y:start_y+new_height, start_x:start_x+new_width] = resized
            
            return standardized, measurements, shape_features
        except:
            standardized = cv2.resize(image, (self.reference_size, self.reference_size))
            return standardized, measurements, shape_features

    def preprocess_user_image(self, image):
        """Preprocess user image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        standardized, measurements, shape_features = self.scale_normalize_with_analysis(gray)
        return standardized, measurements, shape_features

    def enhanced_similarity(self, user_img, template_img, user_shape_features, template_shape_features):
        """Calculate enhanced similarity score"""
        try:
            # Base SSIM score
            ssim_score = ssim(user_img, template_img, full=True)[0]
            
            # Shape similarity
            shape_similarity = self.calculate_shape_similarity(
                user_shape_features or {},
                template_shape_features or {}
            )
            
            # Weighted combination
            final_score = 0.7 * ssim_score + 0.3 * shape_similarity
            return final_score
        except:
            # Fallback to SSIM only
            return ssim(user_img, template_img, full=True)[0]
    
    def calculate_shape_similarity(self, features1, features2):
        """Calculate shape similarity"""
        if not features1 or not features2:
            return 0.5
        
        try:
            comparisons = []
            
            # Compare available features
            for key in ['circularity', 'angularity', 'curvature_index', 'straight_edge_ratio']:
                if key in features1 and key in features2:
                    diff = 1 - abs(features1[key] - features2[key])
                    comparisons.append(diff)
            
            # Corner count similarity
            if 'corner_count' in features1 and 'corner_count' in features2:
                max_corners = max(features1['corner_count'], features2['corner_count'])
                if max_corners > 0:
                    corner_diff = 1 - abs(features1['corner_count'] - features2['corner_count']) / max_corners
                    comparisons.append(corner_diff)
            
            return np.mean(comparisons) if comparisons else 0.5
        except:
            return 0.5

    def find_similar_profiles(self, user_image, max_matches=5):
        """Find similar profiles"""
        self.load_templates()
        
        start_time = time.time()
        processed_user, user_measurements, user_shape_features = self.preprocess_user_image(user_image)

        matches = []
        for class_name, template_list in self.templates.items():
            for template in template_list:
                similarity = self.enhanced_similarity(
                    processed_user, 
                    template['standardized'],
                    user_shape_features,
                    template.get('shape_features', {})
                )
                
                matches.append({
                    'similarity': similarity,
                    'class': class_name,
                    'image': template['original'],
                    'processed': template['standardized'],
                    'filename': template['filename'],
                    'measurements': template['measurements'],
                    'shape_features': template.get('shape_features', {}),
                    'is_original': template.get('is_original', False)
                })

        matches.sort(key=lambda x: x['similarity'], reverse=True)

        # Get best unique class matches
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
    """Display shape analysis with error handling"""
    st.subheader(f"🔷 {title}")
    
    if not shape_features:
        st.info("No shape features available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📐 Shape Characteristics:**")
        
        # Shape type
        shape_type = shape_features.get('shape_type', 'Unknown')
        st.metric("Shape Type", shape_type)
        
        # Circularity
        circularity = shape_features.get('circularity', 0)
        st.progress(circularity)
        st.caption(f"Circularity: {circularity:.3f}")
        
        # Rectangularity
        rectangularity = shape_features.get('rectangularity', 0)
        st.progress(rectangularity)
        st.caption(f"Rectangularity: {rectangularity:.3f}")
    
    with col2:
        st.markdown("**📊 Shape Metrics:**")
        
        metrics = [
            ('angularity', 'Angularity'),
            ('curvature_index', 'Curvature Index'),
            ('straight_edge_ratio', 'Straight Edge Ratio'),
            ('corner_count', 'Corner Count'),
            ('symmetry_score', 'Symmetry Score'),
            ('compactness', 'Compactness')
        ]
        
        for key, label in metrics:
            value = shape_features.get(key, 0)
            if isinstance(value, (int, float)):
                st.write(f"• **{label}**: {value:.3f}")
            else:
                st.write(f"• **{label}**: {value}")

def display_measurements_with_comparison(user_measurements, match_measurements, title="Detailed Measurements"):
    """Display measurements comparison"""
    st.subheader(f"📏 {title}")
    
    if not user_measurements or not match_measurements:
        st.info("No measurements available for comparison")
        return
    
    try:
        user_pixels = user_measurements.get('pixels', {})
        match_pixels = match_measurements.get('pixels', {})
        
        comparison_data = []
        metrics = [
            ('Height', 'height', 'px'),
            ('Width', 'width', 'px'),
            ('Area', 'area', 'px²'),
            ('Perimeter', 'perimeter', 'px'),
            ('Face Height', 'face_height', 'px')
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
        else:
            st.info("No comparable measurements available")
    except:
        st.error("Error displaying measurements comparison")

def display_interactive_results(user_img, processed_user, matches, user_measurements, user_shape_features):
    """Display interactive results"""
    
    st.subheader("📊 Profile Matching Results")
    
    # Display input images
    col1, col2 = st.columns(2)
    with col1:
        st.image(user_img, caption="Original Input", use_column_width=True)
    with col2:
        st.image(processed_user, caption="Normalized Input", use_column_width=True)
    
    # Display user analysis
    if user_shape_features:
        display_shape_analysis(user_shape_features, "Your Profile Shape Analysis")
    
    # Interactive match selection
    st.subheader(f"🏆 Top {len(matches)} Matches")
    st.info("Click on a button below an image to select it for detailed comparison")
    
    # Initialize selection state
    if 'selected_match_idx' not in st.session_state:
        st.session_state.selected_match_idx = -1
    
    # Display matches in columns
    cols = st.columns(len(matches))
    
    for idx, (col, match) in enumerate(zip(cols, matches)):
        with col:
            try:
                match_img = Image.fromarray(match['processed'])
            except:
                match_img = Image.new('L', (100, 100), color=128)
            
            # Selection button
            if col.button(f"Select Match {idx+1}", key=f"select_{idx}"):
                st.session_state.selected_match_idx = idx
            
            # Display image
            st.image(match_img, use_column_width=True)
            
            # Highlight if selected
            is_selected = st.session_state.selected_match_idx == idx
            if is_selected:
                st.success(f"✅ Selected: {match['class']}")
            
            # Display match info
            st.metric(
                label=f"Match {idx+1}: {match['class']}",
                value=f"{match['similarity']:.3f}"
            )
            
            shape_type = match.get('shape_features', {}).get('shape_type', 'Unknown')
            st.caption(f"Shape: {shape_type}")
            st.caption(f"File: {match['filename']}")
    
    # Display detailed comparison for selected match
    if st.session_state.selected_match_idx >= 0 and st.session_state.selected_match_idx < len(matches):
        selected_match = matches[st.session_state.selected_match_idx]
        
        st.markdown("---")
        st.subheader(f"🎯 Detailed Analysis: {selected_match['class']}")
        
        col1, col2 = st.columns(2)
        with col1:
            try:
                st.image(Image.fromarray(selected_match['processed']), 
                        caption="Selected Match", 
                        use_column_width=True)
            except:
                st.warning("Could not display selected image")
            
            st.metric(
                label="Similarity Score",
                value=f"{selected_match['similarity']:.3f}",
                delta=f"Rank: {st.session_state.selected_match_idx + 1}"
            )
        
        with col2:
            # Display shape analysis
            display_shape_analysis(selected_match.get('shape_features', {}), 
                                 "Match Shape Analysis")
        
        # Display measurements comparison
        display_measurements_with_comparison(user_measurements, 
                                           selected_match['measurements'],
                                           "Measurement Comparison")
        
        # Show original reference if available
        if selected_match.get('is_original', False):
            st.success("🎉 This is an ORIGINAL reference image!")
        
        # Try to load and display the class's original image
        class_name = selected_match['class']
        if hasattr(st.session_state.matcher, 'original_images') and class_name in st.session_state.matcher.original_images:
            st.subheader("📁 Original Reference Image for this Class")
            original_img = st.session_state.matcher.original_images[class_name]
            
            col1, col2 = st.columns(2)
            with col1:
                try:
                    original_pil = Image.fromarray(original_img)
                    st.image(original_pil, caption=f"Original: {class_name}", use_column_width=True)
                except:
                    st.warning("Could not display original image")
            
            with col2:
                # Show similarity to original
                if not selected_match.get('is_original', False):
                    try:
                        selected_array = np.array(Image.fromarray(selected_match['processed']).convert('L'))
                        original_resized = cv2.resize(original_img, (selected_array.shape[1], selected_array.shape[0]))
                        similarity = ssim(selected_array, original_resized, full=True)[0]
                        st.metric("Similarity to Original", f"{similarity:.3f}")
                    except:
                        st.info("Could not calculate similarity to original")
    
    # Display matches summary table
    st.subheader("📋 All Matches Summary")
    
    results_data = []
    for i, match in enumerate(matches, 1):
        shape_features = match.get('shape_features', {})
        shape_type = shape_features.get('shape_type', 'Unknown')
        
        results_data.append({
            "Rank": i,
            "Class": match['class'],
            "Similarity": f"{match['similarity']:.3f}",
            "Shape Type": shape_type,
            "Circularity": f"{shape_features.get('circularity', 0):.3f}",
            "Angularity": f"{shape_features.get('angularity', 0):.3f}",
            "Selected": "✅" if st.session_state.selected_match_idx == i-1 else ""
        })
    
    try:
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True)
    except:
        st.warning("Could not display summary table")

def main():
    st.set_page_config(
        page_title="Enhanced Profile Matcher",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Enhanced Profile Image Matching System")
    
    # Initialize session state for selection
    if 'selected_match_idx' not in st.session_state:
        st.session_state.selected_match_idx = -1
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    max_matches = st.sidebar.slider("Maximum matches", 3, 10, 5)
    
    st.sidebar.header("📏 Measurement Settings")
    pixels_to_mm = st.sidebar.number_input(
        "Pixels to mm ratio", 
        min_value=0.0, 
        max_value=10.0, 
        value=0.0,
        help="Set scale for real measurements (0 = pixel measurements only)"
    )
    
    # Initialize matcher
    if 'matcher' not in st.session_state:
        TEMPLATE_PATH = "trained_data"
        if os.path.exists(TEMPLATE_PATH):
            st.session_state.matcher = EnhancedProfileMatcher(TEMPLATE_PATH)
        else:
            st.error(f"Template path '{TEMPLATE_PATH}' not found!")
            st.info(f"Please create a '{TEMPLATE_PATH}' folder with your template images")
            return
    
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
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            except:
                st.error("Could not open uploaded image")
                return
        
        with col2:
            st.info("Ready to analyze!")
            if st.button("🚀 Find Matches & Analyze", type="primary"):
                with st.spinner("Analyzing profile and finding matches..."):
                    try:
                        # Reset selection
                        st.session_state.selected_match_idx = -1
                        
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
                        
                        # Display results
                        display_interactive_results(
                            user_img_pil, 
                            processed_user_pil, 
                            matches, 
                            user_measurements,
                            user_shape_features
                        )
                        
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                        st.info("Please try with a different image")
    
    # Show template status
    with st.sidebar:
        st.header("📊 System Status")
        if hasattr(st.session_state, 'matcher') and st.session_state.matcher.templates:
            template_count = sum(len(v) for v in st.session_state.matcher.templates.values())
            class_count = len(st.session_state.matcher.templates)
            st.success(f"✅ Templates loaded: {template_count} images, {class_count} classes")
        else:
            st.info("📁 Templates will load when you analyze an image")

    # Instructions
    with st.expander("ℹ️ How to use this system"):
        st.markdown("""
        **🎯 Enhanced Features:**
        1. **Better Shape Detection**: Differentiates circles, rectangles, curved vs straight shapes
        2. **Interactive Selection**: Click buttons below images to select for detailed comparison
        3. **Shape Analysis**: Shows circularity, angularity, curvature metrics
        4. **Original Detection**: Finds and shows original reference images
        5. **Detailed Comparison**: Side-by-side measurements with differences
        
        **📁 Expected Folder Structure:**
        ```
        trained_data/
        ├── Class1/
        │   ├── original.jpg    # Auto-detected as reference
        │   ├── image1.jpg
        │   └── image2.jpg
        ├── Class2/
        │   ├── original.png
        │   └── variant.jpg
        └── ...
        ```
        
        **🔍 For Best Results:**
        - Use clear images with good contrast
        - Ensure profiles are clearly visible
        - Set pixels-to-mm ratio for real measurements
        - Look for "Original" marker for reference quality
        """)

if __name__ == "__main__":
    main()
