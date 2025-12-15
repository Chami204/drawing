import cv2
import numpy as np
import os
import streamlit as st
from PIL import Image
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import faiss
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

class VisionProcessor:
    """Eye (Vision): OpenCV + Morphological Operations"""
    def __init__(self):
        self.reference_size = 224
        
    def preprocess_image(self, image):
        """Enhanced preprocessing with morphological operations"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Adaptive thresholding for better binarization
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 2. Morphological operations to clean the image
        kernel = np.ones((3, 3), np.uint8)
        
        # Opening to remove noise
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Closing to fill small holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # 3. Find largest contour
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return self._simple_resize(gray)
        
        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract region of interest
        roi = gray[y:y+h, x:x+w]
        
        # 4. Create mask from cleaned contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mask = mask[y:y+h, x:x+w]
        
        # Apply mask
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
        
        # 5. Normalize while preserving aspect ratio
        result = self._preserve_aspect_resize(masked_roi)
        
        return result
    
    def _preserve_aspect_resize(self, image):
        """Resize image while preserving aspect ratio"""
        h, w = image.shape
        
        # Calculate scaling factor
        scale = self.reference_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create square canvas with white background
        canvas = np.ones((self.reference_size, self.reference_size), dtype=np.uint8) * 255
        
        # Calculate position to center the image
        x_offset = (self.reference_size - new_w) // 2
        y_offset = (self.reference_size - new_h) // 2
        
        # Place image on canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def _simple_resize(self, image):
        """Fallback resize method"""
        return cv2.resize(image, (self.reference_size, self.reference_size))

class FeatureExtractor:
    """Brain (AI): PyTorch + ResNet18 for feature extraction"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.transform = self._get_transform()
        
    def _load_model(self):
        """Load pre-trained ResNet18 model"""
        model = models.resnet18(pretrained=True)
        
        # Remove the final classification layer
        model = nn.Sequential(*list(model.children())[:-1])
        
        # Set to evaluation mode
        model.eval()
        model.to(self.device)
        
        return model
    
    def _get_transform(self):
        """Get image transformations for ResNet"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features(self, image):
        """Extract features from image using ResNet18"""
        # Ensure 3 channels (RGB)
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        image_tensor = self.transform(image_rgb)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(image_tensor)
        
        # Flatten features
        features = features.squeeze().cpu().numpy()
        
        # Normalize features
        features = normalize(features.reshape(1, -1)).flatten()
        
        return features

class MemoryDatabase:
    """Memory (Database): FAISS for similarity search"""
    def __init__(self, dimension=512):
        self.dimension = dimension
        self.index = None
        self.template_info = []
        
    def build_index(self, features_list):
        """Build FAISS index from features"""
        # Convert list of features to numpy array
        features_array = np.array(features_list).astype('float32')
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add features to index
        self.index.add(features_array)
        
        return self.index
    
    def search(self, query_features, k=5):
        """Search for similar items in database"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Prepare query features
        query_array = np.array([query_features]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_array, k)
        
        return distances[0], indices[0]
    
    def add_template_info(self, info):
        """Add template information for retrieval"""
        self.template_info.append(info)

class EnhancedProfileMatcher:
    def __init__(self, template_root):
        self.template_root = template_root
        
        # Initialize components
        self.vision = VisionProcessor()
        self.brain = FeatureExtractor()
        self.memory = MemoryDatabase()
        
        # Storage for templates
        self.templates_loaded = False
        self.templates_info = []
        
    def load_templates(self):
        """Load and process all template images"""
        if self.templates_loaded:
            return
        
        st.write("🧠 Loading and processing templates...")
        start_time = time.time()
        
        features_list = []
        
        for class_name in os.listdir(self.template_root):
            class_path = os.path.join(self.template_root, class_name)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        
                        # Load image
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                        
                        # Preprocess with Vision
                        processed_img = self.vision.preprocess_image(img)
                        
                        # Extract features with Brain
                        features = self.brain.extract_features(processed_img)
                        
                        # Store in memory
                        self.memory.add_template_info({
                            'class': class_name,
                            'filename': img_file,
                            'processed_image': processed_img,
                            'original_path': img_path,
                            'features': features
                        })
                        
                        features_list.append(features)
        
        # Build FAISS index
        if features_list:
            self.memory.build_index(features_list)
            self.templates_loaded = True
            
            st.success(f"""
            ✅ System initialized:
            - Vision: {len(features_list)} images preprocessed
            - Brain: ResNet18 feature extraction ready
            - Memory: FAISS index built with {len(features_list)} vectors
            Time: {time.time()-start_time:.2f} seconds
            """)
    
    def find_similar_profiles(self, user_image, max_matches=5):
        """Find similar profiles using the complete system"""
        self.load_templates()
        
        start_time = time.time()
        
        # Process user image through Vision
        user_processed = self.vision.preprocess_image(user_image)
        
        # Extract features through Brain
        user_features = self.brain.extract_features(user_processed)
        
        # Search in Memory
        distances, indices = self.memory.search(user_features, k=max_matches)
        
        # Prepare results
        matches = []
        for idx, distance in zip(indices, distances):
            if idx < len(self.memory.template_info):
                template_info = self.memory.template_info[idx]
                
                # Convert distance to similarity score (0-1)
                similarity = max(0, 1 - distance / 10)  # Adjust scaling as needed
                
                matches.append({
                    'similarity': float(similarity),
                    'distance': float(distance),
                    'class': template_info['class'],
                    'filename': template_info['filename'],
                    'processed': template_info['processed_image'],
                    'original_path': template_info['original_path']
                })
        
        # Sort by similarity (descending)
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        st.write(f"⏱️ Matching completed in {time.time()-start_time:.2f} seconds")
        
        return user_processed, matches
    
    def visualize_processing_pipeline(self, image):
        """Visualize each step of the processing pipeline"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Original
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(gray, cmap='gray')
        axes[0, 0].set_title('1. Original Image')
        axes[0, 0].axis('off')
        
        # Step 2: Adaptive Threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        axes[0, 1].imshow(binary, cmap='gray')
        axes[0, 1].set_title('2. Adaptive Threshold')
        axes[0, 1].axis('off')
        
        # Step 3: Morphological Cleaning
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        axes[0, 2].imshow(cleaned, cmap='gray')
        axes[0, 2].set_title('3. Morphological Operations')
        axes[0, 2].axis('off')
        
        # Step 4: Contour Detection
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 2)
        axes[1, 0].imshow(contour_img)
        axes[1, 0].set_title('4. Contour Detection')
        axes[1, 0].axis('off')
        
        # Step 5: Extracted Profile
        if contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = gray[y:y+h, x:x+w]
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mask = mask[y:y+h, x:x+w]
            masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
            axes[1, 1].imshow(masked_roi, cmap='gray')
        else:
            axes[1, 1].imshow(gray, cmap='gray')
        axes[1, 1].set_title('5. Extracted Profile')
        axes[1, 1].axis('off')
        
        # Step 6: Final Normalized
        final = self.vision.preprocess_image(image)
        axes[1, 2].imshow(final, cmap='gray')
        axes[1, 2].set_title('6. Final Normalized')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        return fig

def display_system_dashboard(matcher):
    """Display system information dashboard"""
    st.sidebar.markdown("## 🏗️ System Architecture")
    
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        st.markdown("**👁️ Vision**")
        st.caption("OpenCV + Morphological Ops")
        st.caption("Image Cleaning & Normalization")
    
    with col2:
        st.markdown("**🧠 Brain**")
        st.caption("PyTorch + ResNet18")
        st.caption("Feature Extraction")
    
    with col3:
        st.markdown("**💾 Memory**")
        st.caption("FAISS Database")
        st.caption("Similarity Search")
    
    if hasattr(matcher, 'templates_loaded') and matcher.templates_loaded:
        st.sidebar.success("✅ System Ready")
        st.sidebar.metric("Templates Loaded", len(matcher.memory.template_info))
    else:
        st.sidebar.warning("⚠️ System Initializing...")

def display_results_with_selection(user_img, processed_user, matches):
    """Display results with interactive selection"""
    
    st.subheader("🔍 Matching Results")
    
    # Display input images
    col1, col2 = st.columns(2)
    with col1:
        st.image(user_img, caption="Original Input", use_column_width=True)
    with col2:
        st.image(processed_user, caption="Enhanced Processing", use_column_width=True)
    
    # 1. Show all similar matches for selection
    st.subheader("🎯 Select a Match")
    st.info("Click on any image below to select it as your best match")
    
    # Initialize selection in session state
    if 'selected_match_idx' not in st.session_state:
        st.session_state.selected_match_idx = 0
    
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
                f"Select #{idx+1}", 
                key=f"select_{idx}",
                on_click=set_selected_idx,
                args=(idx,)
            )
            
            # Show match info
            similarity_value = match['similarity']
            col.metric(f"Similarity", f"{similarity_value:.3f}")
            col.caption(f"Class: {match['class']}")
            col.caption(f"Distance: {match['distance']:.4f}")
            
            if is_selected:
                col.success("✅ Selected")
    
    # Show the selected match as best match
    if selected_match is None and len(matches) > 0:
        selected_match = matches[st.session_state.selected_match_idx]
    
    if selected_match:
        st.markdown("---")
        
        # Show the selected match as "Best Match"
        st.subheader(f"🏆 Best Match: {selected_match['class']}")
        
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            selected_img = Image.fromarray(selected_match['processed'])
            st.image(selected_img, caption=f"Selected Match (Class: {selected_match['class']})", use_column_width=True)
        
        with col2:
            st.markdown("**📊 Match Details**")
            st.write(f"**Class:** {selected_match['class']}")
            st.write(f"**Filename:** {selected_match['filename']}")
            st.write(f"**Similarity Score:** {selected_match['similarity']:.4f}")
            st.write(f"**FAISS Distance:** {selected_match['distance']:.4f}")
            
            # Show quality indicators
            if selected_match['similarity'] > 0.9:
                st.success("**Quality:** Excellent Match")
            elif selected_match['similarity'] > 0.7:
                st.info("**Quality:** Good Match")
            elif selected_match['similarity'] > 0.5:
                st.warning("**Quality:** Fair Match")
            else:
                st.error("**Quality:** Poor Match")
        
        with col3:
            st.markdown("**📈 Metrics**")
            st.metric("Similarity", f"{selected_match['similarity']:.3%}")
            st.metric("Distance", f"{selected_match['distance']:.4f}")
            st.caption(f"Rank: #{st.session_state.selected_match_idx + 1}")
        
        # Show all matches summary
        st.subheader("📋 All Matches Summary")
        
        import pandas as pd
        summary_data = []
        for i, match in enumerate(matches, 1):
            is_selected = i-1 == st.session_state.selected_match_idx
            summary_data.append({
                "Rank": i,
                "Class": match['class'],
                "Similarity": f"{match['similarity']:.3f}",
                "Distance": f"{match['distance']:.4f}",
                "Selected": "✅" if is_selected else ""
            })
        
        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)

def main():
    st.set_page_config(
        page_title="ALU SCAN Pro",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 ALU SCAN Pro - Advanced Profile Matching")
    st.markdown("""
    **Advanced Profile Matching System**
    - 👁️ **Vision**: OpenCV + Morphological Operations
    - 🧠 **Brain**: PyTorch ResNet18 Feature Extraction  
    - 💾 **Memory**: FAISS Similarity Search Database
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    max_matches = st.sidebar.slider("Matches to show", 1, 10, 5)
    show_processing = st.sidebar.checkbox("Show Processing Pipeline", True)
    
    # Initialize matcher in session state
    if 'matcher' not in st.session_state:
        TEMPLATE_PATH = "trained_data"
        if os.path.exists(TEMPLATE_PATH):
            st.session_state.matcher = EnhancedProfileMatcher(TEMPLATE_PATH)
            st.info("🚀 Initializing ALU SCAN Pro System...")
        else:
            st.error(f"❌ Template folder '{TEMPLATE_PATH}' not found!")
            st.info("Please create a 'trained_data' folder with class subfolders containing profile images.")
            return
    
    # Display system dashboard
    display_system_dashboard(st.session_state.matcher)
    
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
        "📤 Upload Profile Image", 
        type=['png', '.jpg', '.jpeg'],
        help="Upload an image of an aluminum profile for matching"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            # Check if we need to analyze or re-analyze
            analyze_needed = True
            if st.session_state.analysis_done:
                current_file_bytes = uploaded_file.getvalue()[:100]
                if 'last_file_bytes' in st.session_state:
                    if current_file_bytes == st.session_state.last_file_bytes:
                        analyze_needed = False
            
            if analyze_needed:
                if st.button("🔍 Find Matches", type="primary", use_container_width=True):
                    with st.spinner("🧠 Processing image through Vision → Brain → Memory..."):
                        try:
                            # Store file bytes
                            st.session_state.last_file_bytes = uploaded_file.getvalue()[:100]
                            
                            # Reset selection
                            st.session_state.selected_match_idx = 0
                            
                            # Convert to OpenCV format
                            user_img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                            
                            # Show processing pipeline visualization
                            if show_processing:
                                with st.expander("🔬 Processing Pipeline Visualization", expanded=True):
                                    fig = st.session_state.matcher.visualize_processing_pipeline(user_img_cv)
                                    st.pyplot(fig)
                            
                            # Perform matching
                            start_time = time.time()
                            processed_user, matches = st.session_state.matcher.find_similar_profiles(
                                user_img_cv, 
                                max_matches=max_matches
                            )
                            total_time = time.time() - start_time
                            
                            # Convert for display
                            processed_user_pil = Image.fromarray(processed_user)
                            user_img_pil = Image.fromarray(cv2.cvtColor(user_img_cv, cv2.COLOR_BGR2RGB))
                            
                            # Store results
                            st.session_state.processed_user = processed_user_pil
                            st.session_state.matches = matches
                            st.session_state.user_img_pil = user_img_pil
                            st.session_state.analysis_done = True
                            st.session_state.analysis_time = total_time
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            st.exception(e)
            else:
                if st.button("🔄 Re-analyze", type="secondary"):
                    st.session_state.analysis_done = False
                    st.rerun()
    
    # Display results if analysis is done
    if st.session_state.analysis_done and st.session_state.matches is not None:
        # Show analysis time
        if 'analysis_time' in st.session_state:
            st.sidebar.metric("Processing Time", f"{st.session_state.analysis_time:.2f}s")
        
        # Display results
        display_results_with_selection(
            st.session_state.user_img_pil, 
            st.session_state.processed_user, 
            st.session_state.matches
        )
    
    # Instructions and info
    with st.sidebar:
        st.markdown("---")
        st.markdown("**🎯 How to Use:**")
        st.markdown("1. 📤 Upload profile image")
        st.markdown("2. 🔍 Click 'Find Matches'")
        st.markdown("3. 🖱️ Click any match to select")
        st.markdown("4. 📊 Review selected match details")
        
        st.markdown("---")
        st.markdown("**🔧 System Info:**")
        if torch.cuda.is_available():
            st.success("GPU Acceleration: ✅ Available")
        else:
            st.info("GPU Acceleration: ⚠️ CPU Only")
        
        # System status
        if hasattr(st.session_state, 'matcher'):
            if st.session_state.matcher.templates_loaded:
                template_count = len(st.session_state.matcher.memory.template_info)
                st.metric("Database Size", f"{template_count} profiles")
    
    # How it works expander
    with st.expander("🔬 How ALU SCAN Pro Works", expanded=False):
        st.markdown("""
        ### 🏗️ Three-Part Architecture
        
        **👁️ Vision (OpenCV + Morphological Operations)**
        ```
        1. Adaptive Thresholding - Smart binarization
        2. Morphological Operations - Noise removal & cleaning
        3. Contour Detection - Profile extraction
        4. Aspect Ratio Preservation - Maintains proportions
        ```
        
        **🧠 Brain (PyTorch + ResNet18)**
        ```
        1. Feature Extraction - 512-dimensional vectors
        2. Geometric Understanding - Shape & structure analysis
        3. Deep Learning - Pre-trained on ImageNet
        4. Normalization - Consistent feature scaling
        ```
        
        **💾 Memory (FAISS Database)**
        ```
        1. Vector Storage - Efficient feature indexing
        2. Similarity Search - L2 distance calculations
        3. Fast Retrieval - Optimized for large datasets
        4. Scalable - Handles thousands of profiles
        ```
        
        ### 📊 Workflow
        1. **Upload** → User provides profile image
        2. **Vision Processing** → Image cleaning & normalization
        3. **Feature Extraction** → Convert to 512D vector
        4. **Database Search** → Find similar vectors in FAISS
        5. **Results Display** → Show ranked matches
        """)

if __name__ == "__main__":
    # Install instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📦 Required Packages:**")
    st.sidebar.code("""
    pip install streamlit opencv-python
    pip install torch torchvision
    pip install faiss-cpu  # or faiss-gpu
    pip install scikit-learn matplotlib
    """)
    
    main()
