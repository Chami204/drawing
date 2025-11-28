import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import streamlit as st
import tempfile
from PIL import Image
import requests
import io
import time

class ProfileMatcher:
    def __init__(self, template_root):
        self.template_root = template_root
        self.templates = {}
        self.reference_size = 300
        self.load_templates()

    def load_templates(self):
        """Pre-load all template images"""
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
                            standardized = self.scale_normalize(img)
                            class_images.append({
                                'original': img,
                                'standardized': standardized,
                                'filename': os.path.basename(img_path),
                                'class': class_name
                            })
                if class_images:
                    self.templates[class_name] = class_images

        st.success(f"✅ Loaded {sum(len(v) for v in self.templates.values())} templates from {len(self.templates)} classes in {time.time()-start_time:.2f} seconds")

    def scale_normalize(self, image):
        """Normalize image scale while preserving aspect ratio"""
        _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return cv2.resize(image, (self.reference_size, self.reference_size))

        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)

        profile_region = image[y:y+h, x:x+w]
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

        return cv2.resize(padded, (self.reference_size, self.reference_size))

    def preprocess_user_image(self, image):
        """Prepare user image for comparison"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return self.scale_normalize(gray)

    def find_similar_profiles(self, user_image, max_matches=5):
        """Find matching profiles with scale normalization"""
        start_time = time.time()
        processed_user = self.preprocess_user_image(user_image)

        matches = []
        for class_name, template_list in self.templates.items():
            for template in template_list:
                similarity = ssim(processed_user, template['standardized'], full=True)[0]
                matches.append({
                    'similarity': similarity,
                    'class': class_name,
                    'image': template['original'],
                    'processed': template['standardized'],
                    'filename': template['filename']
                })

        matches.sort(key=lambda x: x['similarity'], reverse=True)

        # Get best matches (up to max_matches)
        results = []
        seen_classes = set()
        for match in matches:
            if match['class'] not in seen_classes:
                results.append(match)
                seen_classes.add(match['class'])
                if len(results) >= max_matches:
                    break

        st.write(f"⏱️ Matching completed in {time.time()-start_time:.2f} seconds")
        return processed_user, results

def display_results_streamlit(user_img, processed_user, matches):
    """Display results in Streamlit"""
    
    st.subheader("📊 Profile Matching Results")
    
    # Display input images
    col1, col2 = st.columns(2)
    with col1:
        st.image(user_img, caption="Original Input", use_column_width=True)
    with col2:
        st.image(processed_user, caption="Normalized Input", use_column_width=True)
    
    # Display matches
    st.subheader(f"🎯 Top {len(matches)} Matches")
    
    # Create columns for matches
    cols = st.columns(len(matches))
    for idx, (col, match) in enumerate(zip(cols, matches)):
        with col:
            # Convert numpy array to PIL Image for display
            match_img = Image.fromarray(match['processed'])
            st.image(match_img, use_column_width=True)
            st.metric(
                label=f"Match {idx+1}: {match['class']}",
                value=f"{match['similarity']:.3f}"
            )
            st.caption(f"File: {match['filename']}")
    
    # Display detailed results table
    st.subheader("📋 Detailed Results")
    
    # Create results table
    results_data = []
    for i, match in enumerate(matches, 1):
        results_data.append({
            "Rank": i,
            "Class": match['class'],
            "Similarity Score": f"{match['similarity']:.3f}",
            "Filename": match['filename']
        })
    
    st.table(results_data)

def download_templates_from_github(github_repo_url, local_path):
    """Download template images from GitHub repository"""
    import zipfile
    import urllib.request
    
    if not os.path.exists(local_path):
        os.makedirs(local_path)
        
        # Convert GitHub repo URL to zip download URL
        if github_repo_url.endswith('/'):
            github_repo_url = github_repo_url[:-1]
        zip_url = github_repo_url + "/archive/main.zip"
        
        try:
            st.info("📥 Downloading templates from GitHub...")
            
            # Download zip file
            zip_path = os.path.join(local_path, "templates.zip")
            urllib.request.urlretrieve(zip_url, zip_path)
            
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(local_path)
            
            # Remove zip file
            os.remove(zip_path)
            
            st.success("✅ Templates downloaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error downloading templates: {e}")
            return False
    
    return True

def main():
    st.set_page_config(
        page_title="Profile Image Matcher",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Profile Image Matching System")
    st.markdown("Upload a profile image to find similar matches from our database.")
    
    # Configuration
    GITHUB_REPO_URL = "https://github.com/Chami204/drawing"  # Replace with your GitHub repo URL
    TEMPLATE_PATH = "trained_data"
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    max_matches = st.sidebar.slider("Maximum matches to display", 1, 10, 5)
    
    # Option to use GitHub or local templates
    use_github = st.sidebar.checkbox("Download templates from GitHub", value=True)
    
    # Initialize matcher
    if 'matcher' not in st.session_state:
        if use_github:
            if download_templates_from_github(GITHUB_REPO_URL, TEMPLATE_PATH):
                # Find the extracted folder (GitHub adds '-main' to folder name)
                extracted_folders = [f for f in os.listdir(TEMPLATE_PATH) 
                                   if os.path.isdir(os.path.join(TEMPLATE_PATH, f))]
                if extracted_folders:
                    actual_template_path = os.path.join(TEMPLATE_PATH, extracted_folders[0], "trained data")
                else:
                    actual_template_path = os.path.join(TEMPLATE_PATH, "trained data")
            else:
                st.error("Failed to download templates. Please check the GitHub URL.")
                return
        else:
            actual_template_path = TEMPLATE_PATH
        
        if os.path.exists(actual_template_path):
            st.session_state.matcher = ProfileMatcher(actual_template_path)
        else:
            st.error(f"Template path not found: {actual_template_path}")
            st.info("Please make sure your template data is available.")
            return
    
    # File upload section
    st.header("📤 Upload Profile Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a profile image in PNG, JPG, or JPEG format"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        st.subheader("Uploaded Image")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.info("Image ready for processing!")
            if st.button("🚀 Find Matches", type="primary"):
                with st.spinner("🔍 Finding similar profiles..."):
                    # Convert PIL image to OpenCV format
                    user_img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Find matches
                    processed_user, matches = st.session_state.matcher.find_similar_profiles(
                        user_img_cv, 
                        max_matches=max_matches
                    )
                    
                    # Convert processed image back to PIL for display
                    processed_user_pil = Image.fromarray(processed_user)
                    user_img_pil = Image.fromarray(cv2.cvtColor(user_img_cv, cv2.COLOR_BGR2RGB))
                    
                    # Display results
                    display_results_streamlit(user_img_pil, processed_user_pil, matches)

    # Instructions
    with st.expander("ℹ️ How to use this tool"):
        st.markdown("""
        1. **Upload an image**: Use the file uploader to select a profile image
        2. **Find matches**: Click the 'Find Matches' button to search for similar profiles
        3. **View results**: See the top matches with similarity scores
        
        **Features**:
        - Automatic scale normalization
        - Aspect ratio preservation
        - Structural similarity comparison
        - Multiple match results
        """)

if __name__ == "__main__":

    main()
