"""
fix_all_labels.py — One-shot dataset label repair script.

Fixes 3 issues:
1. Converts Large Vehicle XML annotations → YOLO .txt format (class_id=1)
2. Remaps Three Wheeler class_id from 0 → 2
3. Validates Four Wheeler (class_id=0) and Two Wheeler (class_id=3) are correct
4. Clears all .cache files

Target class mapping (matches data.yaml):
  0 = Four_Wheeler
  1 = Large_Vehicle
  2 = Three_Wheeler
  3 = Two_Wheeler
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os
import xml.etree.ElementTree as ET

BASE = "c:/Users/OMEN/Documents/Engr Abubakar Isa Project/sample"
SPLITS = ["train", "valid", "test"]

# ───────────────────────── 1. Convert Large Vehicle XML → YOLO txt ─────────────
print("\n═══ Step 1: Converting Large Vehicle XML → YOLO .txt ═══")
lv_converted = 0
lv_objects = 0

for split in SPLITS:
    label_dir = os.path.join(BASE, "Large Vehicle", split, "labels")
    image_dir = os.path.join(BASE, "Large Vehicle", split, "images")
    if not os.path.isdir(label_dir):
        print(f"  ⚠️  {split}/labels not found, skipping")
        continue

    xml_files = [f for f in os.listdir(label_dir) if f.endswith(".xml")]
    print(f"  Processing {split}: {len(xml_files)} XML files")

    for xml_file in xml_files:
        xml_path = os.path.join(label_dir, xml_file)
        txt_path = os.path.join(label_dir, os.path.splitext(xml_file)[0] + ".txt")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image dimensions from XML
        size_el = root.find("size")
        if size_el is None:
            print(f"    ⚠️  No <size> in {xml_file}, skipping")
            continue

        img_w = int(size_el.find("width").text)
        img_h = int(size_el.find("height").text)

        if img_w == 0 or img_h == 0:
            print(f"    ⚠️  Zero dimensions in {xml_file}, skipping")
            continue

        yolo_lines = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            xmin = float(bb.find("xmin").text)
            ymin = float(bb.find("ymin").text)
            xmax = float(bb.find("xmax").text)
            ymax = float(bb.find("ymax").text)

            # Clamp to image bounds
            xmin = max(0, min(xmin, img_w))
            ymin = max(0, min(ymin, img_h))
            xmax = max(0, min(xmax, img_w))
            ymax = max(0, min(ymax, img_h))

            # Convert to YOLO format: class x_center y_center width height (all normalized)
            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            if width <= 0 or height <= 0:
                continue

            # class_id = 1 for Large_Vehicle
            yolo_lines.append(f"1 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            lv_objects += 1

        # Write YOLO txt file
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
        lv_converted += 1

print(f"  ✅ Converted {lv_converted} files, {lv_objects} bounding boxes\n")

# ───────────────────────── 2. Remap Three Wheeler: 0 → 2 ─────────────────────
print("═══ Step 2: Remapping Three Wheeler class_id 0 → 2 ═══")
tw_files_changed = 0
tw_annotations_changed = 0

for split in SPLITS:
    label_dir = os.path.join(BASE, "Three Wheeler", split, "labels")
    if not os.path.isdir(label_dir):
        continue

    for f in os.listdir(label_dir):
        if not f.endswith(".txt"):
            continue
        p = os.path.join(label_dir, f)
        with open(p, "r") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]

        new_lines = []
        touched = False
        for ln in lines:
            parts = ln.split()
            if parts and parts[0] == "0":
                parts[0] = "2"
                tw_annotations_changed += 1
                touched = True
            new_lines.append(" ".join(parts))

        if touched:
            with open(p, "w") as fh:
                fh.write("\n".join(new_lines) + ("\n" if new_lines else ""))
            tw_files_changed += 1

print(f"  ✅ Remapped {tw_annotations_changed} annotations in {tw_files_changed} files\n")

# ───────────────────────── 3. Validate all labels ─────────────────────────────
print("═══ Step 3: Validating all labels ═══")
VEHICLE_DIRS = {
    "Four Wheeler": 0,
    "Large Vehicle": 1,
    "Three Wheeler": 2,
    "Two Wheeler": 3,
}
valid_ids = {0, 1, 2, 3}
total_annotations = 0
class_totals = {}
issues = []

for vehicle, expected_id in VEHICLE_DIRS.items():
    for split in SPLITS:
        label_dir = os.path.join(BASE, vehicle, split, "labels")
        if not os.path.isdir(label_dir):
            continue
        for f in os.listdir(label_dir):
            if not f.endswith(".txt"):
                continue
            p = os.path.join(label_dir, f)
            with open(p) as fh:
                for i, ln in enumerate(fh, 1):
                    ln = ln.strip()
                    if not ln:
                        continue
                    parts = ln.split()
                    if len(parts) != 5:
                        issues.append(f"{p}:{i} bad format ({len(parts)} fields)")
                        continue
                    cid = int(parts[0])
                    total_annotations += 1
                    class_totals[cid] = class_totals.get(cid, 0) + 1
                    if cid not in valid_ids:
                        issues.append(f"{p}:{i} invalid class_id={cid}")

print(f"  Total annotations: {total_annotations}")
print(f"  Per-class distribution: {class_totals}")
CLASS_NAMES = {0: "Four_Wheeler", 1: "Large_Vehicle", 2: "Three_Wheeler", 3: "Two_Wheeler"}
for cid, count in sorted(class_totals.items()):
    name = CLASS_NAMES.get(cid, f"UNKNOWN({cid})")
    print(f"    Class {cid} ({name}): {count}")
if issues:
    print(f"  ⚠️  {len(issues)} issues found:")
    for issue in issues[:10]:
        print(f"    {issue}")
else:
    print("  ✅ All labels valid!\n")

# ───────────────────────── 4. Clear .cache files ──────────────────────────────
print("═══ Step 4: Clearing .cache files ═══")
cache_cleared = 0
for vehicle in VEHICLE_DIRS:
    for split in SPLITS:
        cache = os.path.join(BASE, vehicle, split, "labels.cache")
        if os.path.exists(cache):
            os.remove(cache)
            cache_cleared += 1
            print(f"  🧹 Removed {cache}")

# Also check for .cache in valid folders (YOLO sometimes names it 'val')
for vehicle in VEHICLE_DIRS:
    for name in ["val", "valid"]:
        cache = os.path.join(BASE, vehicle, name, "labels.cache")
        if os.path.exists(cache):
            os.remove(cache)
            cache_cleared += 1
            print(f"  🧹 Removed {cache}")

print(f"  ✅ Cleared {cache_cleared} cache files\n")

print("═══ DONE ═══")
print(f"Dataset is ready for training with 4 classes:")
print(f"  0: Four_Wheeler, 1: Large_Vehicle, 2: Three_Wheeler, 3: Two_Wheeler")
