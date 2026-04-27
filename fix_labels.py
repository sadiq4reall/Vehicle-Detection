"""
Fix YOLO labels for Large Vehicle (empty .txt) and Three Wheeler (wrong class ID).
Class mapping per data.yaml:
  0 = Four_Wheeler
  1 = Large_Vehicle
  2 = Three_Wheeler
  3 = Two_Wheeler
"""
import os
import xml.etree.ElementTree as ET
from PIL import Image

base = r'c:\Users\OMEN\Documents\Engr Abubakar Isa Project\sample'

# ============================================================
# FIX 1: Large Vehicle — regenerate .txt from .xml with class=1
# ============================================================
print("=" * 60)
print("FIX 1: Regenerating Large Vehicle labels from XML (class=1)")
print("=" * 60)

fixed_lv = 0
for split in ['train', 'test', 'valid']:
    label_dir = os.path.join(base, 'Large Vehicle', split, 'labels')
    img_dir = os.path.join(base, 'Large Vehicle', split, 'images')
    if not os.path.isdir(label_dir):
        continue

    for fname in os.listdir(label_dir):
        if not fname.endswith('.xml'):
            continue

        xml_path = os.path.join(label_dir, fname)
        txt_path = os.path.join(label_dir, fname.replace('.xml', '.txt'))

        # Get image dimensions for normalization
        img_name = fname.replace('.xml', '.jpg')
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            # Try other extensions
            for ext in ['.jpeg', '.png', '.JPG']:
                alt = os.path.join(img_dir, fname.replace('.xml', ext))
                if os.path.exists(alt):
                    img_path = alt
                    break

        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
        except:
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        lines = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # Convert to YOLO format (normalized x_center, y_center, width, height)
            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            # Clamp to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0.001, min(1, width))
            height = max(0.001, min(1, height))

            lines.append(f"1 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if lines:
            with open(txt_path, 'w') as f:
                f.write('\n'.join(lines) + '\n')
            fixed_lv += 1

print(f"✅ Regenerated {fixed_lv} Large Vehicle label files with class=1\n")

# ============================================================
# FIX 2: Three Wheeler — remap class 0 → 2
# ============================================================
print("=" * 60)
print("FIX 2: Remapping Three Wheeler class 0 → 2")
print("=" * 60)

fixed_tw = 0
remapped_lines = 0
for split in ['train', 'test', 'valid']:
    label_dir = os.path.join(base, 'Three Wheeler', split, 'labels')
    if not os.path.isdir(label_dir):
        continue

    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue

        fpath = os.path.join(label_dir, fname)
        with open(fpath, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        new_lines = []
        touched = False
        for ln in lines:
            parts = ln.split()
            if parts and parts[0] == '0':
                parts[0] = '2'
                touched = True
                remapped_lines += 1
            new_lines.append(' '.join(parts))

        if touched:
            with open(fpath, 'w') as f:
                f.write('\n'.join(new_lines) + '\n')
            fixed_tw += 1

print(f"✅ Remapped {remapped_lines} annotations in {fixed_tw} Three Wheeler files (0→2)\n")

# ============================================================
# VERIFY: Check class IDs across all categories
# ============================================================
print("=" * 60)
print("VERIFICATION: Class IDs per category")
print("=" * 60)

for vehicle in ['Four Wheeler', 'Large Vehicle', 'Three Wheeler', 'Two Wheeler']:
    ids = set()
    count = 0
    for split in ['train', 'test', 'valid']:
        d = os.path.join(base, vehicle, split, 'labels')
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.endswith('.txt'):
                continue
            for ln in open(os.path.join(d, f)):
                ln = ln.strip()
                if ln:
                    ids.add(int(ln.split()[0]))
                    count += 1
    print(f"  {vehicle:20s} → class IDs: {ids}  ({count} annotations)")

# Clean label caches so YOLO re-scans
print("\n🧹 Cleaning label caches...")
for vehicle in ['Four Wheeler', 'Large Vehicle', 'Three Wheeler', 'Two Wheeler']:
    for split in ['train', 'test', 'valid']:
        cache = os.path.join(base, vehicle, split, 'labels.cache')
        if os.path.exists(cache):
            os.remove(cache)
            print(f"  Removed: {cache}")

print("\n🎉 All fixes applied! Re-run sample.py or train_yolo.py to train with all 4 classes.")
