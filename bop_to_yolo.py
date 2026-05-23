import os
import json
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def convert_mask_to_yolo_segment(mask):
    # Znajdź kontury obiektu
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = mask.shape
    segments = []
    
    for contour in contours:
        if len(contour) >= 3:  # Poligon musi mieć minimum 3 punkty
            # Spłaszczamy macierz punktów i normalizujemy współrzędne do zakresu 0-1
            normalized_contour = []
            for pt in contour:
                x_norm = pt[0][0] / w
                y_norm = pt[0][1] / h
                normalized_contour.extend([x_norm, y_norm])
            segments.append(normalized_contour)
            
    return segments

def main():
    bop_root = "output/bop_data/probe/train_pbr"
    yolo_root = "yolo_dataset"
    
    if not os.path.exists(bop_root):
        raise FileNotFoundError(f"Nie znaleziono wygenerowanych danych BlenderProc w: {bop_root}")
        
    # 1. Tworzenie czystej struktury folderów dla YOLO
    for split in ['train', 'val']:
        os.makedirs(os.path.join(yolo_root, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(yolo_root, split, 'labels'), exist_ok=True)

    # 2. Zbieramy listę wszystkich klatek ze wszystkich scen
    all_frames = []
    scene_folders = sorted([f for f in os.listdir(bop_root) if f.isdigit()])
    
    print("Skanowanie wygenerowanych scen...")
    for scene in scene_folders:
        scene_path = os.path.join(bop_root, scene)
        scene_gt_path = os.path.join(scene_path, 'scene_gt.json')
        
        if not os.path.exists(scene_gt_path):
            continue
            
        with open(scene_gt_path, 'r') as f:
            scene_gt = json.load(f)
            
        for im_id_str in scene_gt.keys():
            im_id = int(im_id_str)
            all_frames.append({
                'scene': scene,
                'im_id': im_id,
                'im_id_str': im_id_str,
                'objects': scene_gt[im_id_str]
            })

    print(f"Znaleziono łącznie {len(all_frames)} wyrenderowanych obrazów.")
    if len(all_frames) == 0:
        print("Błąd: Brak klatek do przetworzenia!")
        return

    # 3. Podział danych na train (80%) i val (20%)
    train_frames, val_frames = train_test_split(all_frames, test_size=0.2, random_state=42)
    
    def process_split(frames, split_name):
        print(f"\nPrzetwarzanie i kopiowanie zbioru: {split_name}...")
        
        for frame in tqdm(frames):
            scene = frame['scene']
            im_id = frame['im_id']
            im_id_str = frame['im_id_str']
            
            # Nowa, unikalna nazwa pliku w strukturze YOLO (np. scene_000000_im_000000)
            # Dzięki temu nazwy na pewno się nie pokryją między różnymi scenami
            yolo_base_name = f"sc_{scene}_im_{im_id:06d}"
            
            # Ścieżka źródłowa obrazu RGB
            src_img_path = os.path.join(bop_root, scene, 'rgb', f"{im_id:06d}.jpg") # lub .png jeśli tak zapisałeś
            if not os.path.exists(src_img_path):
                src_img_path = os.path.join(bop_root, scene, 'rgb', f"{im_id:06d}.png")
                
            dst_img_path = os.path.join(yolo_root, split_name, 'images', yolo_base_name + os.path.splitext(src_img_path)[1])
            
            # Zapis etykiet tekstowych YOLO
            label_lines = []
            
            for obj_idx, obj in enumerate(frame['objects']):
                # Filtrujemy tylko Twój właściwy obiekt (id = 1)
                if obj.get('obj_id') != 1:
                    continue
                    
                # Ścieżka do maski widoczności
                mask_path = os.path.join(bop_root, scene, 'mask_visib', f"{im_id:06d}_{obj_idx:06d}.png")
                if not os.path.exists(mask_path):
                    continue
                    
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None or np.sum(mask) == 0:
                    continue
                    
                # Konwersja maski na współrzędne wielokąta YOLO
                yolo_segments = convert_mask_to_yolo_segment(mask)
                
                for segment in yolo_segments:
                    # Format linii YOLO: <id_klasy> <x1> <y1> <x2> <y2> ...
                    # Nasza klasa 'probe' dostaje indeks 0
                    segment_str = " ".join([f"{coord:.6f}" for coord in segment])
                    label_lines.append(f"0 {segment_str}")
            
            # Jeśli obrazek zawiera interesujący nas obiekt, kopiujemy go i zapisujemy etykietę
            if label_lines:
                shutil.copy(src_img_path, dst_img_path)
                
                dst_label_path = os.path.join(yolo_root, split_name, 'labels', yolo_base_name + ".txt")
                with open(dst_label_path, 'w') as f:
                    f.write("\n".join(label_lines))

    process_split(train_frames, 'train')
    process_split(val_frames, 'val')

    # 4. Generowanie pliku dataset.yaml
    yaml_content = f"""
path: {os.path.abspath(yolo_root)}
train: train/images
val: val/images

names:
  0: probe
"""
    with open(os.path.join(yolo_root, "dataset.yaml"), "w") as f:
        f.write(yaml_content.strip())
        
    print(f"\nSukces! Dane gotowe w folderze '{yolo_root}'.")

if __name__ == "__main__":
    main()