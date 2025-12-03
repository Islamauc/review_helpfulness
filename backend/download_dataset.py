"""
Download dataset from Google Drive
"""

import os
import sys
from pathlib import Path

def download_from_google_drive(file_id, destination):
    """
    Download file from Google Drive using gdown.
    
    Args:
        file_id: Google Drive file ID
        destination: Path to save the file
    """
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown
    
    # Google Drive URL format
    url = f"https://drive.google.com/uc?id={file_id}"
    
    print(f"Downloading dataset from Google Drive...")
    print(f"File ID: {file_id}")
    print(f"Destination: {destination}")
    
    # Download the file
    gdown.download(url, destination, quiet=False)
    
    if os.path.exists(destination):
        file_size = os.path.getsize(destination) / (1024 * 1024 * 1024)  # GB
        print(f"\n✅ Download complete!")
        print(f"   File: {destination}")
        print(f"   Size: {file_size:.2f} GB")
        return True
    else:
        print(f"\n❌ Download failed!")
        return False


if __name__ == "__main__":
    # Google Drive file ID from the URL
    # URL: https://drive.google.com/file/d/1fr6aBADlb_xktMkiDahnv3QFiiaVn1MY/view?usp=drive_link
    FILE_ID = "1fr6aBADlb_xktMkiDahnv3QFiiaVn1MY"
    
    # Destination path (project root)
    project_root = Path(__file__).parent.parent
    destination = project_root / "Software_Cleaned_norm.csv"
    
    print("=" * 60)
    print("Dataset Download Script (Google Drive)")
    print("=" * 60)
    print()
    
    # Check if file already exists
    if destination.exists():
        response = input(f"File already exists: {destination}\nOverwrite? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            sys.exit(0)
    
    # Download
    print("📥 Downloading from Google Drive...")
    print("   This may take a while (file is ~2.3 GB)")
    print()
    
    success = download_from_google_drive(FILE_ID, str(destination))
    
    if success:
        print("\n✅ Dataset ready for training!")
        print(f"   Run: python3 backend/train_model.py")
    else:
        print("\n❌ Download failed. Please try:")
        print("   1. Make sure the Google Drive file is accessible")
        print("   2. Install gdown: pip install gdown")
        print("   3. Or download manually and place in project root")

