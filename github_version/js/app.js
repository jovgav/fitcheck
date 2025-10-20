// TensorFlow.js implementation for FitCheck fashion analyzer
class FitCheckClassifier {
    constructor() {
        this.model = null;
        this.isLoaded = false;
    }

    async loadModel() {
        try {
            console.log('Loading TensorFlow.js model...');
            this.model = await tf.loadLayersModel('./tfjs_model/model.json');
            this.isLoaded = true;
            console.log('Model loaded successfully!');
            return true;
        } catch (error) {
            console.error('Error loading model:', error);
            return false;
        }
    }

    preprocessImage(imageElement) {
        // Convert image to tensor and preprocess
        const tensor = tf.browser.fromPixels(imageElement)
            .resizeNearestNeighbor([224, 224])
            .expandDims(0)
            .div(255.0)
            .sub(tf.tensor([0.485, 0.456, 0.406]))
            .div(tf.tensor([0.229, 0.224, 0.225]));
        
        return tensor;
    }

    async predict(imageElement) {
        if (!this.isLoaded) {
            throw new Error('Model not loaded');
        }

        // Preprocess image
        const preprocessedImage = this.preprocessImage(imageElement);
        
        // Make prediction
        const prediction = this.model.predict(preprocessedImage);
        const probabilities = tf.softmax(prediction);
        
        // Get results
        const results = await probabilities.data();
        
        // Clean up tensors
        preprocessedImage.dispose();
        prediction.dispose();
        probabilities.dispose();
        
        return {
            prediction: results[1] > results[0] ? 'cool' : 'uncool',
            confidence: Math.max(results[0], results[1]) * 100
        };
    }
}

// Global classifier instance
const classifier = new FitCheckClassifier();

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', async () => {
    const statusDiv = document.getElementById('modelStatus');
    
    try {
        statusDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading AI model...';
        statusDiv.style.display = 'block';
        
        const loaded = await classifier.loadModel();
        
        if (loaded) {
            statusDiv.className = 'model-status model-loaded';
            statusDiv.innerHTML = '<i class="fas fa-check-circle"></i> AI model loaded and ready!';
        } else {
            statusDiv.className = 'model-status model-not-loaded';
            statusDiv.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Failed to load model. Using demo mode.';
        }
    } catch (error) {
        console.error('Error initializing app:', error);
        statusDiv.className = 'model-status model-not-loaded';
        statusDiv.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Error loading model. Using demo mode.';
    }
});

// File upload handling
const fileInput = document.getElementById('fileInput');

fileInput.addEventListener('change', handleFileSelect);

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleUploadClick() {
    const resultContainer = document.getElementById('resultContainer');
    if (resultContainer.style.display === 'block') {
        resetUpload();
    }
    document.getElementById('fileInput').click();
}

async function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file.');
        return;
    }

    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('resultContainer').style.display = 'none';

    try {
        // Create image element
        const imageElement = await createImageElement(file);
        
        // Analyze with AI model
        let result;
        if (classifier.isLoaded) {
            result = await classifier.predict(imageElement);
        } else {
            // Fallback to demo mode
            result = {
                prediction: Math.random() > 0.5 ? 'cool' : 'uncool',
                confidence: Math.floor(Math.random() * 40) + 30
            };
        }
        
        document.getElementById('loading').style.display = 'none';
        showResult(result, file);
        
    } catch (error) {
        document.getElementById('loading').style.display = 'none';
        console.error('Error analyzing image:', error);
        alert('An error occurred while analyzing the image.');
    }
}

function createImageElement(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = URL.createObjectURL(file);
    });
}

function showResult(data, file) {
    const container = document.getElementById('resultContainer');
    const text = document.getElementById('resultText');
    const imagePreview = document.getElementById('imagePreview');
    const header = document.querySelector('.header');

    // Hide the main question
    header.style.display = 'none';

    // Set result class
    container.className = 'result-container';
    
    // Set content
    const coolness = Math.round(data.confidence);
    text.textContent = `fitcheck says: ${coolness}% cool`;

    // Show image preview
    const reader = new FileReader();
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
        imagePreview.style.display = 'block';
    };
    reader.readAsDataURL(file);

    // Show result
    container.style.display = 'block';
    container.scrollIntoView({ behavior: 'smooth' });
}

function resetUpload() {
    const header = document.querySelector('.header');
    const resultContainer = document.getElementById('resultContainer');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    
    header.style.display = 'block';
    resultContainer.style.display = 'none';
    fileInput.value = '';
    imagePreview.style.display = 'none';
}