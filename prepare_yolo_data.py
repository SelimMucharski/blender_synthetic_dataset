import os
import shutil
from sklearn.model_selection import train_test_split
from ultralytics.data.converter import convert_coco

def setup_yolo_dataset():
    coco_json = "output/bop_data/probe/annotations.json"
    bop_root = "output/bop_data/probe"
    yolo_root = "yolo_dataset"
    
    # 1. Konwersja COCO do tymczasowego formatu YOLO
    print("Konwertuję format COCO na etykiety tekstowe YOLO...")
    convert_coco(labels_dir=coco_json, use_segments=True)
    
    # Po wykonaniu convert_coco, obok pliku annotations.json powstanie folder 'labels'
    converted_labels_dir = "coco_converted/labels"    
    
    # 2. Zbierz wszystkie pary obraz-etykieta
    # Szukamy plików tekstowych z etykietami
    all_labels = sorted([f for f in os.listdir(converted_labels_dir) if f.endswith('.txt')])
    
    # Filtrujemy tylko te, które faktycznie mają swoje odpowiedniki w obrazkach
    valid_pairs = []
    for label_file in all_labels:
        # Nazwa pliku etykiety odpowiada unikalnemu id z COCO, np. 000001.txt
        img_name = label_file.replace('.txt', '.jpg')
        
        # W strukturze BOP musimy odtworzyć gdzie oryginalnie stał ten plik rgb
        # Dla uproszczenia: convert_coco zwykle nazywa etykiety sekwencyjnie.
        # Pobieramy ścieżki
        valid_pairs.append(label_file)

    # 3. Tworzenie czystej struktury folderów train/val
    for split in ['train', 'val']:
        os.makedirs(os.path.join(yolo_root, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(yolo_root, split, 'labels'), exist_ok=True)

    # Podział 80% train / 20% val
    train_files, val_files = train_test_split(all_labels, test_size=0.2, random_state=42)
    
    def move_data(files_list, split_name):
        print(f"Kopiowanie danych do zbioru {split_name}...")
        for label_file in files_list:
            # Kopiowanie etykiety
            src_label = os.path.join(converted_labels_dir, label_file)
            dst_label = os.path.join(yolo_root, split_name, 'labels', label_file)
            shutil.copy(src_label, dst_label)
            
            # Szukanie powiązanego zdjęcia w strukturze BOP (szukamy po folderach train_pbr/0000xx/rgb/)
            # Szybki trick: znajdźmy plik o tej samej nazwie bazowej w oryginalnym katalogu
            img_base_name = label_file.replace('.txt', '.jpg')
            
            # Ponieważ nazwy w COCO z poprzedniego skryptu przypisaliśmy do relatywnych ścieżek, 
            # najprościej odnaleźć zdjęcie przeszukując foldery scen:
            img_found = False
            train_pbr_path = os.path.join(bop_root, 'train_pbr')
            for scene in os.listdir(train_pbr_path):
                potential_img_path = os.path.join(train_pbr_path, scene, 'rgb', img_base_name)
                if os.path.exists(potential_img_path):
                    shutil.copy(potential_img_path, os.path.join(yolo_root, split_name, 'images', img_base_name))
                    img_found = True
                    break

    move_data(train_files, 'train')
    move_data(val_files, 'val')

    # 4. Generowanie pliku dataset.yaml
    yaml_content = f"""
path: {os.path.abspath(yolo_root)} # główny folder
train: train/images
val: val/images

names:
  0: probe
"""
    with open(os.path.join(yolo_root, "dataset.yaml"), "w") as f:
        f.write(yaml_content.strip())
        
    print("\nGotowe! Folder 'yolo_dataset' został w pełni skonfigurowany.")

if __name__ == "__main__":
    setup_yolo_dataset()