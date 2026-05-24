import cv2
from ultralytics import YOLO

def main():
    # 1. Wczytaj swoje najlepsze wytrenowane wagi
    # Jeśli trenowałeś model wielokrotnie, upewnij się, że ścieżka 'train' jest poprawna (np. train2, train3 itd.)
    model_path = "runs/segment/train-6/weights/best.pt"
    
    print(f"Ładowanie modelu z: {model_path}...")
    model = YOLO(model_path)
    
    # 2. Inicjalizacja kamery internetowej
    # Argument '0' oznacza domyślną kamerę w systemie. Jeśli masz kilka kamer, spróbuj '1', '2' itd.
    cap = cv2.VideoCapture(0)
    
    # Opcjonalne ustawienie rozdzielczości kamery (dostosuj jeśli Twoja kamera obsługuje inne)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Błąd: Nie można otworzyć kamery internetowej.")
        return

    print("\nKamera uruchomiona pomyślnie!")
    print("Naciśnij klawisz 'q' w oknie podglądu, aby zakończyć działanie programu.\n")

    while True:
        # Przechwytywanie kolejnej klatki z kamery
        ret, frame = cap.read()
        if not ret:
            print("Błąd: Nie można odebrać klatki z kamery.")
            break

        # 3. Uruchomienie predykcji YOLO na bieżącej klatce
        # verbose=False wycisza potok logów w konsoli, aby obraz działał płynniej
        # conf=0.5 oznacza próg pewności - model pokaże obiekt tylko, jeśli jest pewien na min. 50%
        results = model.predict(frame, conf=0.5, verbose=False)

        # 4. Automatyczne renderowanie masek i ramek na klatce
        # Metoda .plot() tworzy kopię obrazu z nałożonymi detekcjami
        annotated_frame = results[0].plot()

        # 5. Wyświetlenie przetworzonego obrazu w oknie
        cv2.imshow("YOLO26 - Test segmentacji obiektu 'probe'", annotated_frame)

        # Sprawdzanie, czy użytkownik nacisnął klawisz 'q', aby wyjść z pętli
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Czyszczenie zasobów po zamknięciu programu
    cap.release()
    cv2.destroyAllWindows()
    print("Program zakończony.")

if __name__ == "__main__":
    main()