#!/usr/bin/env python3
"""
Install dependencies for DocuSense
Run this if requirements.txt installation fails
"""

import subprocess
import sys


def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False


def main():
    """Install all required packages"""
    packages = [
        "fastapi",
        "uvicorn",
        "python-multipart",
        "pdfminer.six",
        "python-docx",
        "rank-bm25",
        "faiss-cpu",
        "numpy",
        "scikit-learn",
        "spacy",
        "networkx",
        "nltk"
    ]

    print("🚀 Installing DocuSense dependencies...")
    print("=" * 50)

    # Upgrade pip first
    print("Upgrading pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install each package
    failed_packages = []
    for package in packages:
        print(f"\nInstalling {package}...")
        if not install_package(package):
            failed_packages.append(package)

    # Download spaCy model
    print("\nDownloading spaCy English model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy model downloaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download spaCy model: {e}")
        failed_packages.append("en_core_web_sm")

    # Summary
    print("\n" + "=" * 50)
    if failed_packages:
        print(f"❌ Installation completed with errors.")
        print(f"Failed packages: {', '.join(failed_packages)}")
        print("\nTry installing failed packages manually:")
        for pkg in failed_packages:
            if pkg != "en_core_web_sm":
                print(f"pip install {pkg}")
            else:
                print("python -m spacy download en_core_web_sm")
    else:
        print("🎉 All dependencies installed successfully!")
        print("\nYou can now run: python test_basic.py")


if __name__ == "__main__":
    main()