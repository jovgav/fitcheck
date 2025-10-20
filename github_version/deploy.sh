#!/bin/bash

# Deployment script for FitCheck
echo "🚀 Deploying FitCheck Fashion Analyzer..."

# Check if we're in the right directory
if [ ! -f "static_app/index.html" ]; then
    echo "❌ Error: static_app/index.html not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Create deployment package
echo "📦 Creating deployment package..."
mkdir -p deploy
cp -r static_app/* deploy/

# Check if model conversion is needed
if [ ! -f "cool_uncool_model.onnx" ]; then
    echo "⚠️  No ONNX model found. Converting PyTorch model..."
    python convert_model.py
    
    if [ -f "cool_uncool_model.onnx" ]; then
        echo "🔄 Converting to TensorFlow.js format..."
        pip install tensorflowjs
        tensorflowjs_converter --input_format=onnx cool_uncool_model.onnx deploy/tfjs_model
        echo "✅ Model converted successfully!"
    else
        echo "⚠️  Model conversion failed. App will run in demo mode."
    fi
fi

echo ""
echo "🎉 Deployment package ready!"
echo ""
echo "📁 Files to upload:"
echo "   - deploy/index.html"
echo "   - deploy/js/app.js"
echo "   - deploy/tfjs_model/ (if model conversion succeeded)"
echo ""
echo "🌐 Deployment options:"
echo "   1. GitHub Pages: Create repository 'fitcheck' and upload files"
echo "   2. Netlify: Drag deploy/ folder to netlify.com/drop"
echo "   3. Vercel: Run 'vercel' in deploy/ directory"
echo "   4. Surge: Run 'surge' in deploy/ directory"
echo ""
echo "✨ Your FitCheck app will be live and accessible to anyone!"