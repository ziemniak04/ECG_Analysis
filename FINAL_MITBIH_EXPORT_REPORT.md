# Raport końcowy — MIT-BIH heartbeat classification, faza 10

Ponownie zweryfikowano istniejący kompletny eksport bez nadpisywania plików danych. Pozostaje on w `data/processed/ecg_dataset_v1/`.

## Wynik

- 48 przetworzonych nagrań, 47 grup osób w obrębie MIT-BIH.
- 109452 kompletne, zmapowane segmenty, logiczne X `[109452, 2, 252]`, float32, y int64.
- N=90594, S=2781, V=7235, F=802, Q=8040 — dokładnie zgodne z fazą 9.
- 48 shardów NPZ oraz dataset_metadata.json, patient_groups.json, class_distribution.csv i lead_configuration.csv.
- Wszystkie shardy można wczytać bez pickle; osie i metadane per uderzenie są zgodne.
- Per-recording Z-score po filtracji i przed segmentacją pozostaje bez zmian, bez wspólnego dopasowania train/test.

## Weryfikacja grup i źródeł

Oficjalna [dokumentacja PhysioNet](https://physionet.org/content/mitdb/1.0.0/) potwierdza 48 nagrań od 47 osób oraz wspólnego pacjenta rekordów 201/202. Mapa `metadata/patient_groups.json` zachowuje tę parę razem. Pozostałe singletony wynikają z dokumentowanej liczby osób i jawnego powtórzenia; mapa oznacza tę podstawę odrębnie od bezpośrednio opisanej relacji 201/202. Tokeny grup są lokalnymi identyfikatorami datasetu, nie rzeczywistymi tożsamościami; relacje poza MIT-BIH pozostają nieznane.

Ponowne `--verify-only` sprawdziło wszystkie shardy, liczebności, pozycje i symbole adnotacji z WFDB, kanały z nagłówków oraz niezmienność SHA-256 **676 plików** źródłowych. Hashe aktualnego kodu odpowiadają manifestowi eksportu. Oryginalna baza i gotowy eksport nie zostały zmienione.

## Testy

Uruchomiono `.venv/bin/python -m pytest`: **342 passed, 132 warnings in 64.10s**. Są to 321 testów wcześniejszych faz i 21 testów eksportu. Ostrzeżenia pochodzą z pandas używanego wewnątrz WFDB. Nie wystąpiły błędy. Testy obejmują deterministyczny powtórny eksport kontrolny, ochronę istniejących plików, dtype/shape, etykiety, wszystkie shardy, konfiguracje kanałów i wspólną grupę 201/202.

Działający przykład odczytu został wykonany: dla rekordu 100 uzyskano X `(2271, 2, 252)` float32 i y `(2271,)` int64. Instrukcje odczytu są w `DATASET_FOR_ANNA.md`.

## Pliki i zakres zmian

Implementacja `src/prepare_dataset.py`, testy `tests/test_prepare_dataset.py`, mapa grup i eksport były już obecne i zostały wykorzystane bez duplikowania. W tym przebiegu uzupełniono `DATASET_FOR_ANNA.md` oraz `README_preprocessing.md` i utworzono ten raport. Nie zmieniono kodu ani ustawień preprocessingu i nie regenerowano istniejących dużych danych.

Dokumentacja jawnie objaśnia nazwy pól: original_annotation_symbols są przechowywane jako original_symbols, konfiguracja preprocessingu i segmentacji w preprocessing_config, a wersja mapy pod mapping_version. Informacje nie zostały utracone ani przeliczone.

## Instrukcja odczytu dla Anny

Z katalogu projektu:

```python
from src.prepare_dataset import load_record_shard

data = load_record_shard("data/processed/ecg_dataset_v1/recordings/100.npz")
X, y = data["X"], data["y"]
record_ids = data["record_ids"]
patient_groups = data["patient_groups"]
```

## Pozostałe ograniczenia i granica zadań

Zachowano oryginalną kolejność i nazwy kanałów: indeks 0/1 nie jest tym samym odprowadzeniem we wszystkich nagraniach. Normalizacja jest offline i wymaga kompatybilnej procedury dla nowych nagrań; nie jest globalnym skalerem treningowym. Istnieją szum, efekty brzegowe i silny imbalance. Dane nie są diagnozą rytmu ani pacjenta.

Eksport dotyczy wyłącznie pojedynczych uderzeń MIT-BIH i pięciu grup AAMI-style. Przyszłe PTB-XL i klasyfikacja rytmu wymagają oddzielnego kontraktu danych; ich przetwarzania nie wdrożono. Nie dodano splitów, balansowania, modeli, treningu, ewaluacji ani Streamlit. Zakres fazy 10 jest ukończony; dalsze zadania modelowe pozostają po stronie Anny.
