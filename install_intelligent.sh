#!/bin/bash

# Installation script for MapleCLI with intelligent features

echo "🚀 MapleCLI Intelligent Features Installer"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found. Please install pip first."
    exit 1
fi

echo "✓ pip3 is available"
echo ""

# Ask user what to install
echo "Choose installation option:"
echo "1) Basic installation (no intelligent features)"
echo "2) Full installation (with intelligent features - recommended)"
echo "3) Full installation with GPU support (requires CUDA)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "📦 Installing basic MapleCLI..."
        pip3 install -e .
        ;;
    2)
        echo ""
        echo "📦 Installing MapleCLI with intelligent features..."
        echo "This will download ~500MB of dependencies (torch, sentence-transformers, etc.)"
        echo ""
        read -p "Continue? [y/N]: " confirm
        if [[ $confirm == [yY] ]]; then
            pip3 install -e ".[intelligent]"
        else
            echo "Installation cancelled."
            exit 0
        fi
        ;;
    3)
        echo ""
        echo "📦 Installing MapleCLI with GPU support..."
        echo "This requires CUDA to be installed on your system."
        echo ""
        read -p "Continue? [y/N]: " confirm
        if [[ $confirm == [yY] ]]; then
            pip3 install -e ".[intelligent]"
            pip3 uninstall -y faiss-cpu
            pip3 install faiss-gpu
        else
            echo "Installation cancelled."
            exit 0
        fi
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "✅ Installation complete!"
echo ""
echo "🎯 Quick Start:"
echo "  1. Run: maplecli chat"
echo "  2. Enable YOLO mode: :yolo"
echo "  3. Switch to project: :cd /path/to/project"
echo "  4. Analyze: :analyze"
echo "  5. Ask questions!"
echo ""
echo "📚 Documentation:"
echo "  - Quick Start: QUICKSTART_INTELLIGENT.md"
echo "  - Full Guide: INTELLIGENT_FEATURES.md"
echo "  - Examples: YOLO_EXAMPLES.md"
echo ""
echo "Happy coding! 🚀"

