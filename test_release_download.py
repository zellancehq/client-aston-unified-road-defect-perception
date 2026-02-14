"""
Test script to verify GitHub release assets are publicly accessible
Run this after creating the release to verify the URLs work
"""

import requests

# Test URLs
YOLO_URL = "https://github.com/zellancehq/client-aston-unified-road-defect-perception/releases/download/v1.0/YOLOv12_Road_Defects_Model.pt"
RESNET_URL = "https://github.com/zellancehq/client-aston-unified-road-defect-perception/releases/download/v1.0/ResNet50_Road_Defects_Model.pth"

def test_url(url, name):
    print(f"\nTesting {name}...")
    print(f"URL: {url}")
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        if response.status_code == 200:
            size_mb = int(response.headers.get('content-length', 0)) / (1024 * 1024)
            print(f"✅ SUCCESS - File accessible ({size_mb:.2f} MB)")
            return True
        else:
            print(f"❌ FAILED - Status code: {response.status_code}")
            if response.status_code == 404:
                print("   Release or file not found. Make sure:")
                print("   1. Release is published (not draft)")
                print("   2. Tag is exactly 'v1.0'")
                print("   3. Files are uploaded with exact names")
            return False
    except Exception as e:
        print(f"❌ ERROR - {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("GitHub Release Asset Download Test")
    print("=" * 60)
    
    yolo_ok = test_url(YOLO_URL, "YOLOv12 Model")
    resnet_ok = test_url(RESNET_URL, "ResNet50 Model")
    
    print("\n" + "=" * 60)
    if yolo_ok and resnet_ok:
        print("✅ All model files are publicly accessible!")
        print("Your application will be able to download them.")
    else:
        print("⚠️  Some files are not accessible.")
        print("Please check the release configuration.")
    print("=" * 60)
