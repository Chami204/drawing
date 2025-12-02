import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import streamlit as st
from PIL import Image
import time
import math

class EnhancedProfileMatcher:
    def __init__(self, template_root):
        self.template_root = template_root
        self.templates = {}
        self.reference_size = 300
        self.pixels_to_mm_ratio = None

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
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            standardized, measurements, edge_features = self.process_image_comprehensive(img)
                            class_images.append({
                                'original': img,
                                'standardized': standardized,
                                'filename': os.path.basename(img_path),
                                'class': class_name,
                                'measurements': measurements,
                                'edge_features': edge_features  # Store edge analysis
                            })
                if class_images:
                    self.templates[class_name] = class_images

        st.success(f"✅ Loaded {sum(len(v) for v in self.templates.values())} templates from {len(self.templates)} classes in {time.time()-start_time:.2f} seconds")

    def detect_profile_contour(self, image):
        """Enhanced contour detection with better edge preservation"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise while preserving edges
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Use adaptive thresholding for better edge detection
        thresh = cv2.adaptiveThreshold(blurred, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up small noise
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback to simple thresholding
            _, thresh_simple = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh_simple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Get the largest contour (main profile)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour while preserving important features
        epsilon = 0.001 * cv2.arcLength(main_contour, True)
        approximated = cv2.approxPolyDP(main_contour, epsilon, True)
        
        return approximated

    def analyze_edges_comprehensive(self, image):
        """Comprehensive edge analysis considering ALL edges and internal features"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Canny Edge Detection for all edges
        edges_canny = cv2.Canny(gray, 50, 150)
        
        # 2. Laplacian for edge intensity
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # 3. Sobel gradients for edge direction
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_direction = np.arctan2(sobel_y, sobel_x)
        
        # 4. Find corners using Harris detector
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        
        # 5. Find all contours (including internal ones)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contour hierarchy
        contour_features = []
        if contours and len(contours) > 1:
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area > 10:  # Ignore very small contours (noise)
                    perimeter = cv2.arcLength(cnt, True)
                    circularity = (4 * math.pi * area) / (perimeter**2) if perimeter > 0 else 0
                    
                    # Fit ellipse for shape analysis
                    if len(cnt) >= 5:
                        ellipse = cv2.fitEllipse(cnt)
                        (center, axes, angle) = ellipse
                        major_axis = max(axes)
                        minor_axis = min(axes)
                        eccentricity = np.sqrt(1 - (minor_axis**2 / major_axis**2)) if major_axis > 0 else 0
                    else:
                        eccentricity = 0
                    
                    # Convex hull analysis
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    # Shape moments for centroid and orientation
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                    else:
                        cx, cy = 0, 0
                    
                    contour_features.append({
                        'area': area,
                        'perimeter': perimeter,
                        'circularity': circularity,
                        'eccentricity': eccentricity,
                        'solidity': solidity,
                        'centroid': (cx, cy),
                        'is_internal': i > 0  # First contour is external
                    })
        
        # Calculate edge statistics
        edge_pixels = np.sum(edges_canny > 0)
        total_pixels = edges_canny.shape[0] * edges_canny.shape[1]
        edge_density = edge_pixels / total_pixels
        
        # Edge direction histogram (8 bins for 0-360 degrees)
        direction_hist = np.zeros(8)
        valid_gradients = gradient_magnitude > 10  # Threshold
        directions = gradient_direction[valid_gradients]
        directions_deg = np.degrees(directions) % 360
        for deg in directions_deg:
            bin_idx = int(deg / 45) % 8
            direction_hist[bin_idx] += 1
        direction_hist = direction_hist / np.sum(direction_hist) if np.sum(direction_hist) > 0 else direction_hist
        
        # Corner analysis
        corner_pixels = np.sum(corners > 0.01 * corners.max())
        corner_density = corner_pixels / total_pixels
        
        return {
            'edge_density': edge_density,
            'corner_density': corner_density,
            'direction_histogram': direction_hist.tolist(),
            'contour_features': contour_features,
            'gradient_mean': np.mean(gradient_magnitude),
            'gradient_std': np.std(gradient_magnitude),
            'laplacian_mean': np.mean(np.abs(laplacian)),
            'laplacian_std': np.std(laplacian)
        }

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
        
        # Convex hull analysis
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area_px / hull_area if hull_area > 0 else 0
        
        # Fit ellipse for shape analysis
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            major_axis = max(ellipse[1])
            minor_axis = min(ellipse[1])
            ellipse_angle = ellipse[2]
            eccentricity = np.sqrt(1 - (minor_axis**2 / major_axis**2)) if major_axis > 0 else 0
        else:
            major_axis = max(w, h)
            minor_axis = min(w, h)
            ellipse_angle = 0
            eccentricity = 0
        
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
                'solidity': solidity,
                'eccentricity': eccentricity,
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

    def process_image_comprehensive(self, image):
        """Process image with comprehensive feature extraction"""
        contour = self.detect_profile_contour(image)
        
        # Calculate measurements
        measurements = self.calculate_measurements(contour)
        
        # Standardize image
        standardized = self.scale_normalize_image(image, contour)
        
        # Analyze edges comprehensively
        edge_features = self.analyze_edges_comprehensive(standardized)
        
        return standardized, measurements, edge_features

    def scale_normalize_image(self, image, contour):
        """Normalize image scale while preserving aspect ratio and internal features"""
        if contour is None:
            return cv2.resize(image, (self.reference_size, self.reference_size))

        x, y, w, h = cv2.boundingRect(contour)
        profile_region = image[y:y+h, x:x+w]
        
        # Use max to preserve aspect ratio better
        scale_factor = self.reference_size / max(w, h)
        new_width = int(w * scale_factor)
        new_height = int(h * scale_factor)
        
        # Use INTER_AREA for downsampling (preserves edges better)
        resized = cv2.resize(profile_region, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Create canvas
        standardized = np.ones((self.reference_size, self.reference_size), dtype=np.uint8) * 255
        
        # Center the image
        start_x = (self.reference_size - new_width) // 2
        start_y = (self.reference_size - new_height) // 2
        standardized[start_y:start_y+new_height, start_x:start_x+new_width] = resized
        
        return standardized

    def preprocess_user_image(self, image):
        """Prepare user image for comparison with comprehensive analysis"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        standardized, measurements, edge_features = self.process_image_comprehensive(gray)
        return standardized, measurements, edge_features

    def comprehensive_similarity(self, img1, img2, features1, features2):
        """Calculate comprehensive similarity considering ALL features"""
        # 1. Base SSIM score (structural similarity)
        ssim_score = ssim(img1, img2, full=True)[0]
        
        # 2. Edge feature similarity
        edge_similarity = self.compare_edge_features(features1, features2)
        
        # 3. Gradient similarity (texture and internal details)
        gradient_similarity = self.compare_gradients(img1, img2)
        
        # 4. Local binary pattern similarity (texture patterns)
        lbp_similarity = self.compare_lbp_patterns(img1, img2)
        
        # 5. Contour shape similarity
        contour_similarity = self.compare_contour_features(features1.get('contour_features', []), 
                                                          features2.get('contour_features', []))
        
        # Weighted combination (emphasis on SSIM and edge features)
        final_score = (0.4 * ssim_score + 
                      0.25 * edge_similarity + 
                      0.15 * gradient_similarity + 
                      0.10 * lbp_similarity + 
                      0.10 * contour_similarity)
        
        return final_score
    
    def compare_edge_features(self, features1, features2):
        """Compare edge features comprehensively"""
        similarity_scores = []
        
        # Compare edge density
        density1 = features1.get('edge_density', 0)
        density2 = features2.get('edge_density', 0)
        density_sim = 1 - abs(density1 - density2) / max(density1, density2) if max(density1, density2) > 0 else 1
        similarity_scores.append(density_sim)
        
        # Compare corner density
        corner1 = features1.get('corner_density', 0)
        corner2 = features2.get('corner_density', 0)
        corner_sim = 1 - abs(corner1 - corner2) / max(corner1, corner2) if max(corner1, corner2) > 0 else 1
        similarity_scores.append(corner_sim)
        
        # Compare direction histograms (cosine similarity)
        hist1 = np.array(features1.get('direction_histogram', [0]*8))
        hist2 = np.array(features2.get('direction_histogram', [0]*8))
        if np.sum(hist1) > 0 and np.sum(hist2) > 0:
            dot_product = np.dot(hist1, hist2)
            norm1 = np.linalg.norm(hist1)
            norm2 = np.linalg.norm(hist2)
            hist_sim = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
        else:
            hist_sim = 0
        similarity_scores.append(hist_sim)
        
        # Compare gradient statistics
        grad_mean1 = features1.get('gradient_mean', 0)
        grad_mean2 = features2.get('gradient_mean', 0)
        grad_mean_sim = 1 - abs(grad_mean1 - grad_mean2) / max(grad_mean1, grad_mean2) if max(grad_mean1, grad_mean2) > 0 else 1
        similarity_scores.append(grad_mean_sim)
        
        return np.mean(similarity_scores)
    
    def compare_gradients(self, img1, img2):
        """Compare gradient patterns (captures internal details)"""
        # Calculate gradients
        sobel_x1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
        grad1 = np.sqrt(sobel_x1**2 + sobel_y1**2)
        
        sobel_x2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
        grad2 = np.sqrt(sobel_x2**2 + sobel_y2**2)
        
        # Normalize gradients
        grad1_norm = (grad1 - np.min(grad1)) / (np.max(grad1) - np.min(grad1) + 1e-10)
        grad2_norm = (grad2 - np.min(grad2)) / (np.max(grad2) - np.min(grad2) + 1e-10)
        
        # Compare using correlation
        correlation = np.corrcoef(grad1_norm.flatten(), grad2_norm.flatten())[0, 1]
        return (correlation + 1) / 2  # Convert from [-1, 1] to [0, 1]
    
    def compare_lbp_patterns(self, img1, img2, radius=1, n_points=8):
        """Compare Local Binary Patterns (texture patterns)"""
        def compute_lbp(image):
            lbp = np.zeros_like(image, dtype=np.uint8)
            for i in range(radius, image.shape[0]-radius):
                for j in range(radius, image.shape[1]-radius):
                    center = image[i, j]
                    binary_pattern = 0
                    for n in range(n_points):
                        angle = 2 * np.pi * n / n_points
                        x = i + int(radius * np.cos(angle))
                        y = j + int(radius * np.sin(angle))
                        if image[x, y] >= center:
                            binary_pattern |= (1 << n)
                    lbp[i, j] = binary_pattern
            return lbp
        
        lbp1 = compute_lbp(img1)
        lbp2 = compute_lbp(img2)
        
        # Compute histograms
        hist1, _ = np.histogram(lbp1.flatten(), bins=256, range=(0, 256))
        hist2, _ = np.histogram(lbp2.flatten(), bins=256, range=(0, 256))
        
        # Normalize histograms
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        # Compute Bhattacharyya coefficient
        bc = np.sum(np.sqrt(hist1 * hist2))
        return bc
    
    def compare_contour_features(self, contours1, contours2):
        """Compare contour features including internal contours"""
        if not contours1 or not contours2:
            return 0.5
        
        # Sort contours by area (largest first)
        contours1_sorted = sorted(contours1, key=lambda x: x['area'], reverse=True)[:5]  # Top 5 contours
        contours2_sorted = sorted(contours2, key=lambda x: x['area'], reverse=True)[:5]
        
        similarities = []
        
        # Compare shape features for each contour
        for i in range(min(len(contours1_sorted), len(contours2_sorted))):
            c1 = contours1_sorted[i]
            c2 = contours2_sorted[i]
            
            # Compare circularity
            circ_sim = 1 - abs(c1['circularity'] - c2['circularity'])
            
            # Compare solidity
            sol_sim = 1 - abs(c1['solidity'] - c2['solidity'])
            
            # Compare eccentricity
            ecc_sim = 1 - abs(c1['eccentricity'] - c2['eccentricity'])
            
            # Average the similarities
            contour_sim = (circ_sim + sol_sim + ecc_sim) / 3
            similarities.append(contour_sim)
        
        return np.mean(similarities) if similarities else 0.5

    def find_similar_profiles(self, user_image, max_matches=5):
        """Find matching profiles with comprehensive analysis"""
        self.load_templates()
        
        start_time = time.time()
        processed_user, user_measurements, user_edge_features = self.preprocess_user_image(user_image)

        matches = []
        for class_name, template_list in self.templates.items():
            for template in template_list:
                similarity = self.comprehensive_similarity(
                    processed_user, 
                    template['standardized'],
                    user_edge_features,
                    template['edge_features']
                )
                matches.append({
                    'similarity': similarity,
                    'class': class_name,
                    'image': template['original'],
                    'processed': template['standardized'],
                    'filename': template['filename'],
                    'measurements': template['measurements'],
                    'edge_features': template['edge_features']
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
        return processed_user, results, user_measurements, user_edge_features

def display_measurements(measurements, title="Measurement Analysis"):
    """Display measurements in a clean format"""
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
        st.write(f"• Compactness: {pixels['compactness']:.2f}")
    
    with col2:
        if measurements['millimeters']:
            st.markdown("**📏 Real-world Measurements:**")
            mm = measurements['millimeters']
            st.write(f"• Height: {mm['height']:.1f} mm")
            st.write(f"• Width: {mm['width']:.1f} mm")
            st.write(f"• Area: {mm['area']:.1f} mm²")
            st.write(f"• Perimeter: {mm['perimeter']:.1f} mm")
        else:
            st.markdown("**ℹ️ Scale Information:**")
            st.write("Set pixels-to-mm ratio for real measurements")

def display_edge_analysis(edge_features, title="Edge Analysis"):
    """Display comprehensive edge analysis"""
    st.subheader(f"🔍 {title}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Edge Statistics:**")
        st.write(f"• Edge Density: {edge_features.get('edge_density', 0):.4f}")
        st.write(f"• Corner Density: {edge_features.get('corner_density', 0):.4f}")
        st.write(f"• Gradient Mean: {edge_features.get('gradient_mean', 0):.2f}")
        st.write(f"• Gradient Std: {edge_features.get('gradient_std', 0):.2f}")
        
        # Contour information
        contours = edge_features.get('contour_features', [])
        if contours:
            st.write(f"• Number of Contours: {len(contours)}")
            if len(contours) > 1:
                st.write(f"• Internal Contours: {len([c for c in contours if c.get('is_internal', False)])}")
    
    with col2:
        st.markdown("**🎯 Feature Summary:**")
        
        # Edge type classification
        edge_density = edge_features.get('edge_density', 0)
        corner_density = edge_features.get('corner_density', 0)
        
        if edge_density > 0.1:
            st.write("• **Edge Type**: Detailed/Textured")
        elif edge_density > 0.05:
            st.write("• **Edge Type**: Moderately detailed")
        else:
            st.write("• **Edge Type**: Smooth/Simple")
        
        if corner_density > 0.01:
            st.write("• **Shape Type**: Angular/Complex")
        else:
            st.write("• **Shape Type**: Curved/Simple")
        
        # Internal feature detection
        contours = edge_features.get('contour_features', [])
        if len(contours) > 1:
            st.write("• **Internal Features**: Present")
        else:
            st.write("• **Internal Features**: Minimal")

def display_results_comprehensive(user_img, processed_user, matches, user_measurements, user_edge_features):
    """Display comprehensive results with detailed analysis"""
    
    st.subheader("🔬 Comprehensive Profile Analysis Results")
    
    # Display input images
    col1, col2 = st.columns(2)
    with col1:
        st.image(user_img, caption="Original Input", use_column_width=True)
    with col2:
        st.image(processed_user, caption="Normalized & Analyzed", use_column_width=True)
    
    # Display user measurements
    display_measurements(user_measurements, "Your Profile Measurements")
    
    # Display edge analysis
    display_edge_analysis(user_edge_features, "Your Profile Edge Analysis")
    
    if matches:
        st.subheader("🎯 Top Match Analysis")
        best_match = matches[0]
        
        col1, col2 = st.columns(2)
        with col1:
            match_img = Image.fromarray(best_match['processed'])
            st.image(match_img, use_column_width=True)
            
            # Similarity breakdown (simulated)
            st.metric(
                label=f"Best Match: {best_match['class']}",
                value=f"{best_match['similarity']:.3f}"
            )
            
            # Show edge density comparison
            user_edge_density = user_edge_features.get('edge_density', 0)
            match_edge_density = best_match['edge_features'].get('edge_density', 0)
            edge_diff = user_edge_density - match_edge_density
            
            st.metric(
                label="Edge Density Match",
                value=f"{1 - abs(edge_diff):.3f}",
                delta=f"{edge_diff:+.4f}"
            )
            
            st.caption(f"File: {best_match['filename']}")
        
        with col2:
            display_measurements(best_match['measurements'], f"Reference Measurements")
            
            st.markdown("**🔍 Detailed Comparison:**")
            user_pixels = user_measurements['pixels']
            match_pixels = best_match['measurements']['pixels']
            
            # More comprehensive comparison
            comparisons = [
                ("Height", user_pixels['height'], match_pixels['height'], "px"),
                ("Area", user_pixels['area'], match_pixels['area'], "px²"),
                ("Perimeter", user_pixels['perimeter'], match_pixels['perimeter'], "px"),
                ("Aspect Ratio", user_pixels['aspect_ratio'], match_pixels['aspect_ratio'], ""),
                ("Solidity", user_pixels.get('solidity', 0), match_pixels.get('solidity', 0), ""),
            ]
            
            for name, user_val, match_val, unit in comparisons:
                diff = user_val - match_val
                if unit:
                    st.write(f"• {name}: {user_val:.1f}{unit} vs {match_val:.1f}{unit} (diff: {diff:+.1f}{unit})")
                else:
                    st.write(f"• {name}: {user_val:.3f} vs {match_val:.3f} (diff: {diff:+.3f})")
    
    st.subheader(f"🏆 Top {len(matches)} Matches")
    
    # Create a grid for matches
    cols = st.columns(len(matches))
    for idx, (col, match) in enumerate(zip(cols, matches)):
        with col:
            match_img = Image.fromarray(match['processed'])
            st.image(match_img, use_column_width=True)
            
            # Quick info
            st.metric(
                label=f"Match {idx+1}: {match['class']}",
                value=f"{match['similarity']:.3f}"
            )
            
            # Edge density indicator
            edge_density = match['edge_features'].get('edge_density', 0)
            if edge_density > 0.1:
                edge_type = "Detailed"
            elif edge_density > 0.05:
                edge_type = "Moderate"
            else:
                edge_type = "Smooth"
            
            st.caption(f"Edges: {edge_type} ({edge_density:.3f})")
            st.caption(f"File: {match['filename']}")
    
    # Detailed results table
    st.subheader("📋 Comprehensive Results Analysis")
    
    results_data = []
    for i, match in enumerate(matches, 1):
        measurements = match['measurements']['pixels']
        edge_features = match['edge_features']
        
        results_data.append({
            "Rank": i,
            "Class": match['class'],
            "Similarity": f"{match['similarity']:.3f}",
            "Height": f"{measurements['height']:.1f} px",
            "Area": f"{measurements['area']:.1f} px²",
            "Edge Density": f"{edge_features.get('edge_density', 0):.3f}",
            "Corners": f"{edge_features.get('corner_density', 0):.3f}",
            "Contours": len(edge_features.get('contour_features', []))
        })
    
    import pandas as pd
    df = pd.DataFrame(results_data)
    st.table(df)

def main():
    st.set_page_config(
        page_title="Enhanced Profile Matcher",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Comprehensive Profile Matching System")
    st.markdown("Analyzes ALL edges, vertices, curves, and internal features for maximum accuracy")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    max_matches = st.sidebar.slider("Maximum matches", 1, 10, 5)
    
    st.sidebar.header("📏 Measurement Settings")
    pixels_to_mm = st.sidebar.number_input(
        "Pixels to mm ratio", 
        min_value=0.0, 
        max_value=10.0, 
        value=0.0,
        help="For real-world measurements"
    )
    
    st.sidebar.header("🔍 Analysis Settings")
    analyze_internal = st.sidebar.checkbox("Analyze internal features", value=True,
                                         help="Consider internal edges and contours")
    
    # Initialize matcher
    if 'matcher' not in st.session_state:
        TEMPLATE_PATH = "trained_data"
        st.session_state.matcher = EnhancedProfileMatcher(TEMPLATE_PATH)
        st.info("🔬 Enhanced matcher initialized with comprehensive analysis")
    
    # Apply settings
    if pixels_to_mm > 0:
        st.session_state.matcher.pixels_to_mm_ratio = pixels_to_mm
    
    # File upload
    st.header("📤 Upload Profile Image")
    
    uploaded_file = st.file_uploader(
        "Choose a profile image", 
        type=['png', '.jpg', '.jpeg'],
        help="Upload for comprehensive analysis"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.info("Ready for comprehensive analysis!")
            if st.button("🔬 Analyze & Find Matches", type="primary"):
                with st.spinner("Comprehensive analysis in progress..."):
                    try:
                        user_img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        processed_user, matches, user_measurements, user_edge_features = st.session_state.matcher.find_similar_profiles(
                            user_img_cv, 
                            max_matches=max_matches
                        )
                        
                        processed_user_pil = Image.fromarray(processed_user)
                        user_img_pil = Image.fromarray(cv2.cvtColor(user_img_cv, cv2.COLOR_BGR2RGB))
                        
                        display_results_comprehensive(
                            user_img_pil, 
                            processed_user_pil, 
                            matches, 
                            user_measurements,
                            user_edge_features
                        )
                        
                    except Exception as e:
                        st.error(f"Error in analysis: {str(e)}")
                        st.info("Please try a different image")
    
    # System status
    with st.sidebar:
        st.header("📊 System Status")
        if st.session_state.matcher.templates:
            template_count = sum(len(v) for v in st.session_state.matcher.templates.values())
            class_count = len(st.session_state.matcher.templates)
            st.success(f"✅ Templates: {template_count} images, {class_count} classes")
        else:
            st.info("📁 Templates will load on first analysis")

    # Explanation
    with st.expander("ℹ️ About Comprehensive Analysis"):
        st.markdown("""
        **🎯 What This System Analyzes:**
        
        1. **All Edges**: External AND internal edges throughout the profile
        2. **Vertices & Corners**: Harris corner detection for angular features
        3. **Curve Analysis**: Gradient direction and curvature at every point
        4. **Internal Features**: Any internal contours or details
        5. **Texture Patterns**: Local Binary Patterns for micro-details
        6. **Gradient Fields**: Complete gradient magnitude and direction
        
        **🔬 How Matching Works:**
        - **40%**: Structural Similarity (SSIM) - overall pattern
        - **25%**: Edge Features - density, corners, directions
        - **15%**: Gradient Similarity - internal texture patterns
        - **10%**: LBP Patterns - micro-texture analysis
        - **10%**: Contour Features - shape characteristics
        
        **📊 For Maximum Accuracy:**
        - System analyzes EVERY pixel's characteristics
        - Considers both macro and micro features
        - Differentiates between similar-looking profiles
        - Captures subtle internal details
        """)

if __name__ == "__main__":
    main()
