import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import os
from PIL import Image
import io

# Check for TensorFlow version compatibility
try:
    from tensorflow.keras.applications.efficientnet import preprocess_input
except ImportError:
    # Fallback for older TensorFlow versions
    st.warning("EfficientNet not found in your TensorFlow installation. Using a generic preprocessing function.")
    def preprocess_input(x):
        return (x / 127.5) - 1.0

# Set page configuration
st.set_page_config(
    page_title="Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a prettier UI - properly formatted
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 3.2rem;
        background: linear-gradient(90deg, #FF6B6B, #FF8C8C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #FF6B6B;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Upload section styling */
    .upload-container {
        background: linear-gradient(145deg, #ffffff, #f5f7f9);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        margin-bottom: 30px;
        border: 1px solid #f0f0f0;
        text-align: center;
    }
    
    .upload-icon {
        font-size: 3rem;
        color: #FF6B6B;
        margin-bottom: 15px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #FF8C8C);
        color: white;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
        box-shadow: 0 4px 10px rgba(255, 107, 107, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #FF5E5E, #FF7A7A);
        box-shadow: 0 6px 15px rgba(255, 107, 107, 0.4);
        transform: translateY(-2px);
    }
    
    /* Results container styling */
    .prediction-result {
        background: linear-gradient(145deg, #ffffff, #f5f7f9);
        border-radius: 20px;
        padding: 25px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    
    .prediction-flower-name {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #FF6B6B, #FF8C8C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        font-weight: 700;
    }
    
    .prediction-confidence {
        font-size: 1.3rem;
        color: #555;
        margin-bottom: 15px;
    }
    
    /* Flower info styling */
    .flower-info {
        background-color: #f8f9fa;
        border-radius: 16px;
        padding: 25px;
        border-left: 5px solid #FF6B6B;
        box-shadow: 0 5px 15px rgba(0,0,0,0.03);
        margin-top: 20px;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #FF6B6B, #FF8C8C);
        border-radius: 10px;
    }
    
    .stProgress {
        height: 10px;
    }
    
    /* Image caption styling */
    .caption {
        text-align: center;
        font-size: 0.9rem;
        color: #777;
        margin-top: 8px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        background-color: #f5f7f9;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FF6B6B !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    .sidebar-title {
        font-size: 1.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #555;
    }
    
    /* Hide default footer */
    footer {
        visibility: hidden;
    }
    
    /* Custom cards for model info */
    .info-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.03);
        margin-bottom: 20px;
        border: 1px solid #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

# Define flower categories - make sure these match exactly with your training labels order
categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Flower emoji mapping
flower_emojis = {
    'daisy': '🌼',
    'dandelion': '🌱',
    'rose': '🌹',
    'sunflower': '🌻',
    'tulip': '🌷'
}

# Flower beautiful images for the sidebar
flower_descriptions = {
    'daisy': "Simple yet charming with white petals and yellow centers. Symbolizes innocence and purity.",
    'dandelion': "Bright yellow blooms that transform into delicate seed heads. Known for resilience and medicinal properties.",
    'rose': "The queen of flowers with layered petals and enchanting fragrance. Symbolizes love and passion.",
    'sunflower': "Tall, cheerful blooms that follow the sun. Represents adoration and loyalty.",
    'tulip': "Elegant cup-shaped flowers in vibrant colors. Symbolizes perfect love and spring."
}

# Function to load the pre-trained model
@st.cache_resource
def load_trained_model(model_path='flower_model.h5'):
    """
    Load the saved fine-tuned model.
    If the model doesn't exist at the specified path, display an error message.
    """
    try:
        # Check if file exists before trying to load
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to make predictions with enhanced preprocessing
def predict_flower(image, model):
    IMG_SIZE = 224
    
    try:
        # Ensure we're working with a valid image
        if image is None or image.size == 0:
            raise ValueError("Invalid image input")
            
        # Convert to RGB (in case the image is in a different format)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to the model's expected input dimensions
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        
        # Create a batch dimension
        img_batch = np.expand_dims(img_resized, axis=0)
        
        # Apply the same preprocessing as during training
        processed_img = preprocess_input(img_batch)
        
        # Make prediction
        predictions = model.predict(processed_img)
        
        # Get the predicted class index and confidence
        pred_index = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_index])
        
        # Validate prediction index is in range
        if pred_index >= len(categories):
            st.warning(f"Model predicted class index {pred_index} which is out of range. Check model compatibility.")
            pred_index = 0  # Default to first category
        
        return {
            'flower_name': categories[pred_index],
            'confidence': confidence,
            'all_probabilities': {categories[i]: float(predictions[0][i]) for i in range(len(categories))},
            'raw_predictions': predictions[0].tolist()
        }
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Function to create a beautiful chart
def create_probability_chart(probabilities):
    sorted_probs = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))
    
    # Use custom colors for chart
    colors = ['#FF6B6B', '#FF8E72', '#FFAA86', '#FFBE9F', '#FFD4B2']
    
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_alpha(0.0)  # Make background transparent
        ax.patch.set_alpha(0.0)   # Make plot area transparent
        
        bars = ax.bar(
            [f"{name.capitalize()} {flower_emojis.get(name, '')}" for name in sorted_probs.keys()],
            [prob * 100 for prob in sorted_probs.values()],
            color=colors
        )
        
        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 1,
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold',
                color='#555'
            )
        
        ax.set_ylim(0, 105)
        ax.set_ylabel('Probability (%)', color='#555', fontsize=12)
        ax.set_xlabel('Flower Type', color='#555', fontsize=12)
        ax.set_title('Classification Confidence', color='#FF6B6B', fontsize=14, fontweight='bold')
        plt.xticks(rotation=30, ha='right', color='#555')
        
        # Customize grid and spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#dddddd')
        ax.spines['bottom'].set_color('#dddddd')
        ax.tick_params(colors='#888888')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

# Function to display prediction results
def display_prediction_results(result, image):
    if not result:
        st.error("Prediction failed. Please try with a different image.")
        return
    
    # Create columns for result display
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Use markdown for safer HTML rendering
        st.markdown(f"""
        <div class="prediction-result">
            <div class="prediction-flower-name">{result['flower_name'].capitalize()} {flower_emojis.get(result['flower_name'], '')}</div>
            <div class="prediction-confidence">Confidence: {result['confidence']*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create and display beautiful bar chart
        fig = create_probability_chart(result['all_probabilities'])
        if fig:
            st.pyplot(fig)
    
    with col2:
        # Make sure the flower name is in our descriptions
        flower_name = result['flower_name']
        if flower_name not in flower_descriptions:
            flower_descriptions[flower_name] = "A beautiful flowering plant with distinct characteristics."
        
        # Display information about the predicted flower using safer HTML
        st.markdown(f"""
        <div class="flower-info">
            <h4>{flower_emojis.get(flower_name, '🌸')} About {flower_name.capitalize()} Flowers</h4>
            <p>{flower_descriptions[flower_name]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add characteristics section separately
        st.markdown(f"""
        <div class="flower-info">
            <h4>🔍 Characteristics</h4>
            <p>{get_flower_characteristics(flower_name)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add growing conditions separately
        st.markdown(f"""
        <div class="flower-info">
            <h4>🌱 Growing Conditions</h4>
            <p>{get_growing_conditions(flower_name)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add confidence levels with custom progress bars
        st.markdown("<h4 style='margin-top: 20px;'>Confidence Levels</h4>", unsafe_allow_html=True)
        
        sorted_probs = dict(sorted(result['all_probabilities'].items(), key=lambda item: item[1], reverse=True))
        for category, prob in sorted_probs.items():
            st.markdown(f"**{category.capitalize()} {flower_emojis.get(category, '')}**")
            st.progress(prob)
            st.markdown(f"<small style='color: #888;'>{prob*100:.1f}%</small>", unsafe_allow_html=True)

# Helper functions for flower information
def get_flower_characteristics(flower_name):
    characteristics = {
        'daisy': "Perennial flowering plants with white petals surrounding a bright yellow center. Typical bloom size is 1-2 inches in diameter.",
        'dandelion': "Deep taproots and composite yellow flowers that turn into spherical seed heads. Known for their distinctive toothed leaves.",
        'rose': "Woody perennial flowering plant with thorny stems, fragrant blooms, and varied flower forms from simple to highly complex petal arrangements.",
        'sunflower': "Tall annual plants with large flower heads that can grow up to 12 feet high. The flower head consists of brown central discs surrounded by bright yellow petals.",
        'tulip': "Bulbous spring-blooming perennials with showy cup-shaped flowers in almost every color of the rainbow. Most have a single flower per stem."
    }
    return characteristics.get(flower_name, "Features not available for this flower type.")

def get_growing_conditions(flower_name):
    conditions = {
        'daisy': "Thrives in full sun to partial shade with well-drained soil. Water moderately and deadhead spent blooms to encourage more flowers.",
        'dandelion': "Adaptable to most soil conditions and thrives in sunny areas. Extremely resilient and can grow almost anywhere with minimal care.",
        'rose': "Prefers full sun, well-drained soil rich in organic matter, and good air circulation. Regular pruning and fertilizing are recommended.",
        'sunflower': "Requires full sun, well-drained soil, and regular watering. Plant in an area protected from strong winds due to their height.",
        'tulip': "Best in full sun to partial shade with well-drained soil. Plant bulbs in fall for spring blooms. Prefer cool temperatures during blooming."
    }
    return conditions.get(flower_name, "Growing information not available for this flower type.")

# Function to load a test image if model or file upload isn't available
def load_test_image():
    # Create a simple test image - a colored rectangle with text
    img = np.ones((300, 400, 3), dtype=np.uint8) * 255  # White background
    # Add a red rectangle
    cv2.rectangle(img, (50, 50), (350, 250), (0, 0, 255), -1)
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Test Image', (100, 150), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    return img

# Add demo mode
def demo_mode():
    st.info("Running in demo mode. Upload your own flower images or use the test features.")
    
    if st.button("Show Test Image"):
        test_img = load_test_image()
        st.image(test_img, caption="Test Image (not a real flower)", use_column_width=True)
        st.info("This is just a test image. In a real scenario, upload a flower photo.")
        
    st.markdown("### Sample Results Preview")
    st.write("Without a model loaded, here's how results would appear:")
    
    # Mocked results for demonstration
    mock_result = {
        'flower_name': 'rose',
        'confidence': 0.92,
        'all_probabilities': {
            'rose': 0.92,
            'tulip': 0.05,
            'daisy': 0.02,
            'sunflower': 0.01,
            'dandelion': 0.00
        },
        'raw_predictions': [0.02, 0.00, 0.92, 0.01, 0.05]
    }
    
    # Create columns for preview
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="prediction-result">
            <div class="prediction-flower-name">Rose 🌹</div>
            <div class="prediction-confidence">Confidence: 92.00%</div>
            <p><i>(Demo result - not from actual prediction)</i></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="flower-info">
            <h4> 🌹 About Rose Flowers</h4>
            <p>The queen of flowers with layered petals and enchanting fragrance. Symbolizes love and passion.</p>
            <p><i>(Demo information only)</i></p>
        </div>
        """, unsafe_allow_html=True)

# Main application
def main():
    # Sidebar with beautiful styling
    with st.sidebar:
        # Use safer HTML rendering
        st.markdown("<div class='sidebar-title'>🌿 Flower Classifier</div>", unsafe_allow_html=True)
        
        # Add a brief app description
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <p>Identify beautiful flowers with AI!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add flower thumbnails to sidebar
        st.markdown("<p style='text-align: center; color: #FF6B6B; font-weight: 600;'>Recognizable Flowers</p>", unsafe_allow_html=True)
        
        for i, (flower, emoji) in enumerate(flower_emojis.items()):
            st.markdown(f"""
            <div style="padding: 8px; margin-bottom: 10px; background-color: #f8f9fa; border-radius: 10px;">
                <span style="font-size: 1.5rem;">{emoji}</span> <span style="font-weight: 500;">{flower.capitalize()}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        model_path = st.text_input("Model Path", "flower_model.h5")
        
        # Add a debug mode option in the sidebar
        debug_mode = st.checkbox("Debug Mode", value=False)
        
        # Add a demo mode option
        demo_mode_enabled = st.checkbox("Demo Mode (no model required)", value=False)
        
        st.markdown("""
        <div style="margin-top: 30px; background-color: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center;">
            <p style="font-size: 0.8rem; color: #888;">Created with ❤️ using TensorFlow and EfficientNet</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Application title and description - using div instead of h1
    st.markdown("<div class='main-header'>Flower Classifier AI</div>", unsafe_allow_html=True)
    
    # Descriptive text
    st.markdown("""
    <p style='font-size: 1.2rem; text-align: center; margin-bottom: 2rem; color: #555;'>
        Upload an image of a flower and our AI will identify its type with beautiful details!
    </p>
    """, unsafe_allow_html=True)
    
    # If demo mode is enabled, show demo content
    if demo_mode_enabled:
        demo_mode()
        return
    
    # Load the fine-tuned model
    with st.spinner("Loading AI model... Please wait."):
        model = load_trained_model(model_path)
    
    if model is None:
        st.warning("""
        ### Model not found or failed to load!
        
        Please ensure the model file exists at the specified path. You can:
        1. Check the model path in the sidebar
        2. Enable "Demo Mode" to see how the app works without a model
        3. Upload or create a model file with the correct format
        """)
        
        # Offer demo mode option
        if st.button("Enable Demo Mode"):
            st.session_state.demo_mode = True
            st.experimental_rerun()
        return
    
    # Display model information if in debug mode
    if debug_mode:
        with st.expander("Model Technical Information"):
            st.write(f"Input shape: {model.input_shape}")
            st.write(f"Output shape: {model.output_shape}")
    
    # Create beautiful tabs
    tab1, tab2 = st.tabs(["✨ Identify Flower", "🔍 About the Model"])
    
    # Tab 1: Upload Image and Classify
    with tab1:
        # Create a more attractive upload section
        st.markdown("""
        <div class="upload-container">
            <div class="upload-icon">📸</div>
            <div style="font-size: 1.8rem; font-weight: 600; margin-bottom: 10px;">Upload a Flower Image</div>
            <p style="color: #777; margin-bottom: 20px;">Choose a clear photo of a flower for the best results</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Center the uploader
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                # Read the image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Display the uploaded image with prettier caption
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                    use_column_width=True
                )
                st.markdown("<p class='caption'>Your uploaded flower image</p>", unsafe_allow_html=True)
                
                # Center the button and make it prettier
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    classify_button = st.button("✨ Identify Flower", use_container_width=True)
                
                # Make prediction with beautiful loading animation
                if classify_button:
                    with st.spinner("✨ Analyzing your flower image..."):
                        result = predict_flower(image, model)
                    
                    if result:
                        # Divider for results
                        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
                        st.markdown("<div class='sub-header'>✨ Classification Results</div>", unsafe_allow_html=True)
                        display_prediction_results(result, image)
                        
                        # Debug information
                        if debug_mode:
                            with st.expander("Debug Information"):
                                st.write("### Raw Model Output")
                                st.write(f"Raw prediction values: {result['raw_predictions']}")
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    # Tab 2: Model Information with beautiful styling
    with tab2:
        st.markdown("<div class='sub-header'>About Our Flower Classification Model</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Use safer HTML with proper styling
            st.markdown("""
            <div class="info-card">
                <h3>🧠 Model Architecture</h3>
                <p>Our flower classification system uses transfer learning with EfficientNetB0 as the base model, enhanced with additional layers for flower recognition.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**Base:** EfficientNetB0 pre-trained on ImageNet")
            st.write("**Feature extraction:** Global Average Pooling")
            st.write("**Classification head:** Multiple dense layers with ReLU activation")
            st.write("**Output:** 5-class softmax layer for flower classification")
        
        with col2:
            # Use safer HTML with proper styling
            st.markdown("""
            <div class="info-card">
                <h3>📊 Performance & Training</h3>
                <p>The model was trained on a dataset of flower images across 5 categories. For best results:</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("• Use clear, well-lit photos of flowers")
            st.write("• Position the flower as the main subject in the image")
            st.write("• Avoid heavily filtered or edited images")
            st.write("• Try to use images similar to natural flower photographs")
        
        # Technical details in expanders
        with st.expander("Technical Requirements"):
            st.markdown("""
            ### Image Preprocessing
            For optimal results, images are automatically:
            - Resized to 224×224 pixels
            - Converted to RGB format
            - Normalized using EfficientNet's preprocessing function
            - Batch processed for inference
            
            The application handles all this processing automatically.
            """)
        
        # Display model summary if available
        if model and debug_mode:
            with st.expander("View Model Technical Summary"):
                # Capture model summary
                import io
                from contextlib import redirect_stdout
                
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    model.summary()
                summary = buffer.getvalue()
                
                st.code(summary, language="text")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {e}")
        st.info("Try enabling Demo Mode in the sidebar to see app features without requiring model file.") 