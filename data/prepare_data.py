import os
import cv2
from pathlib import Path

def check_and_prepare_data(source_dir="data/source", target_dir="data/target", resize_to=(256,256)):
    Path(source_dir).mkdir(parents=True, exist_ok=True)
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    source_imgs = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg','.png','.jpeg','.bmp'))])
    target_imgs = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg','.png','.jpeg','.bmp'))])

    print(f"✅ Found {len(source_imgs)} source images and {len(target_imgs)} target images")

    if not source_imgs:
        print("⚠️ No source images found in", source_dir)
        return

    # بررسی هماهنگی نام فایل‌ها
    missing = []
    for src in source_imgs:
        base = os.path.splitext(src)[0]
        found = any(base in t for t in target_imgs)
        if not found:
            missing.append(src)
    if missing:
        print("⚠️ Warning: These source files don't have corresponding targets:")
        for m in missing:
            print("  ", m)

    # بررسی قابلیت خواندن و اندازه‌ی تصاویر
    for folder in [source_dir, target_dir]:
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                continue
            path = os.path.join(folder, fn)
            img = cv2.imread(path)
            if img is None:
                print("❌ Failed to read image:", path)
                continue
            h, w = img.shape[:2]
            if resize_to:
                img_resized = cv2.resize(img, resize_to)
                cv2.imwrite(path, img_resized)
    print("✅ All images resized and verified.")

if __name__ == "__main__":
    check_and_prepare_data()
