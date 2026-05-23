from ultralytics import YOLO

def main():
    # 1. Ładujemy model YOLO26 zorientowany na SEGMENTACJĘ (stąd końcówka -seg).
    # Wariant 'n' (nano) pobierze się automatycznie przy pierwszym uruchomieniu.
    # Jest super szybki, lekki i idealny do sprawdzenia, jak sieć radzi sobie z Twoim obiektem.
    model = YOLO("yolo26n-seg.pt")
    model.train(
        data="yolo_dataset/dataset.yaml", 
        epochs=50, 
        imgsz=640,
        # Poprawne parametry zmiany kolorów/jasności (HSV)
        hsv_h=0.015,  # Losowy odcień (0.0 - 1.0)
        hsv_s=0.7,    # Losowe nasycenie (0.0 - 1.0)
        hsv_v=0.4,    # Losowa jasność (0.0 - 1.0)
        # Dodatkowe mocne augmentacje niszczące idealną geometrię i tekstury:
        degrees=10.0, # Losowa rotacja obrazu o +/- 10 stopni
        scale=0.5,    # Losowe skalowanie obiektu (symuluje bycie bliżej/dalej kamery)
        perspective=0.001, # Lekkie zniekształcenie perspektywy
        fliplr=0.5    # Losowe odbicie lustrzane w poziomie (szansa 50%)
    )
    print("Trening zakończony! Twoje wagi są gotowe.")

if __name__ == "__main__":
    main()