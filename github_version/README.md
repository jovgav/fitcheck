# FitCheck

A machine learning application that analyzes fashion images to determine if outfits are "cool" or "uncool" based on personal style preferences.

## 🚀 Live Demo

**Live Demo:** https://yourusername.github.io/fitcheck

## 🎯 Features

- **AI-Powered Analysis**: CNN model trained on personal style preferences
- **Real-time Inference**: Instant results in browser using TensorFlow.js
- **Beautiful UI**: Clean, minimalist design
- **Cross-Platform**: Works on any device with a modern browser

## 📁 Project Structure

```
├── static_app/           # Static web app (GitHub Pages ready)
│   ├── index.html       # Main application
│   └── js/app.js        # TensorFlow.js implementation
├── templates/           # Flask templates (for local development)
├── sample_images/       # Sample images for demo
├── app.py              # Flask web application
├── model.py            # PyTorch CNN model
├── train.py            # Training script
└── requirements.txt     # Python dependencies
```

## 🛠️ Quick Start

### Option 1: Static App (GitHub Pages)
```bash
# Open static_app/index.html in your browser
open static_app/index.html
```

### Option 2: Flask App (Local Development)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

## 🌐 Deployment

### GitHub Pages (Automatic)
1. This repository is configured for GitHub Pages
2. Your app is automatically deployed at: `https://yourusername.github.io/fitcheck`
3. The static app runs entirely in the browser - no server needed!

### Other Platforms
- **Netlify**: Drag `static_app/` folder to netlify.com/drop
- **Vercel**: `vercel` command in `static_app/` directory
- **Surge**: `surge` command in `static_app/` directory

## 🧠 How It Works

1. **Upload Image**: Users upload outfit photos
2. **AI Analysis**: TensorFlow.js model analyzes the image
3. **Instant Results**: Get percentage confidence scores
4. **Browser-Based**: Everything runs locally in the user's browser

## 📊 Model Architecture

- **CNN with Batch Normalization**
- **4 Convolutional Layers** (32→64→128→256 filters)
- **Adaptive Average Pooling**
- **Dropout for Regularization**
- **Binary Classification** (Cool/Uncool)

## 🎨 Customization

The app is designed to be easily customizable:
- Modify `static_app/index.html` for UI changes
- Update `static_app/js/app.js` for functionality
- Train new models with your own labeled data

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

---

**Made with ❤️ for fashion and machine learning**