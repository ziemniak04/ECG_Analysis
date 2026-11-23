# Handoff integracji ECG-APP dla Anny

Stan: ukończono modele i finalne oceny PTB-XL19 oraz MIT-BIH26. Na osobnym `origin/main` (przejrzany a2fe7c5) istnieją już Streamlit i inference helpers dla innej architektury `src/ml/ECGNet`. W tej gałęzi AI NIE zaimplementowano kompatybilnego połączenia tych komponentów z opisanymi poniżej checkpointami. Detektor i jego zgodność z tym kontraktem wymagają odrębnej walidacji. Patrz `docs/FEATURE_BRANCH_PUBLICATION_PLAN.md`; nie zastępować modeli samą podmianą plików wag. Starsze raporty faz dokumentują stan w danym momencie; np. komunikaty o braku PyTorch w fazie16 nie opisują obecnego stanu. Bieżące wyniki zbiorcze są w `docs/experiments/` i `EXPERIMENTS_OVERVIEW.md`.

## Dwa odrębne zadania

| Zadanie | Implementacja | Wejście float32 | Logity | Klasy |
|---|---|---|---|---|
| PTB-XL rytmy | `src.models.ptbxl_cnn.PTBXLRhythmCNN`,74116 parametrów | `[B,12,1000]`,10s,100Hz | `[B,4]` | SR0,AFIB1,STACH2,SBRAD3 |
| MIT-BIH beaty | `src.models.mitbih_cnn_rr.MITBIHBeatCNNRR`,17989 parametrów | ECG `[B,2,252]` + RR `[B,6]`,360Hz | `[B,5]` | N0,S1,V2,F3,Q4 |

Nie mieszać etykiet beatów AAMI-style z diagnozami rytmu. N nie oznacza zdrowego pacjenta, a SR nie wyklucza innych nieprawidłowości ECG. Q obejmuje paced/unclassifiable; symbol `f` (paced-normal fusion) różni się od `F` (ventricular-normal fusion). Wyniki Softmax nie są skalibrowanym prawdopodobieństwem klinicznym.

## Zamrożone modele i checkpointy — przekazać oddzielnie

Pliki wag nie są publikowane w Git niezależnie od ich rozmiaru. Magdalena przekazuje je Annie osobnym uzgodnionym kanałem (np. prywatny zasób uczelniany), z zachowaniem poniższych ścieżek lub jawnym ustawieniem lokalnej ścieżki w przyszłej aplikacji. Nie dodawać ich przez `git add -f`.

| Model | Lokalny checkpoint | SHA256 |
|---|---|---|
| PTB-XL B unweighted,epoka6 | `results/model_training/phase18c_optimization/run_001/B_unweighted/best_model.pt` | `40dab87717a6c563e34f9c77e546b5f6f35f44eab73b4357bd4ed0181fc1351e` |
| MIT-BIH Phase24B,epoka6 | `results/model_training/phase24b_mitbih_cnn_rr/run_001/best_model.pt` | `9440f3494af59d3272d7c8de115c03b592d2738279957d97b5e7a6ac9c235536` |

MIT-BIH Phase25B NIE zastępuje wybranego24B. `results/model_training/phase25b_reduced_f/run_001/best_model.pt` jest tylko checkpointem porównawczym, nie modelem aplikacji.

Sprawdzenie po transferze: `sha256sum <ścieżka_checkpointu>`. Dla MIT-BIH best checkpoint zawiera `model_configuration`, `preprocessing_configuration.RR`, `preprocessing_configuration.RR_transform`, `class_names` i `model_state_dict`. Przekazać też tekstowe konfiguracje/reproducibility z odpowiednich runów oraz, w razie korzystania z obecnego adaptera danych, cały niezmieniony manifest/config/scaler. Nie przenosić samych luźnych NPZ bez manifestu i metadanych.

## Finalne wyniki — już wykonane, nie uruchamiać ponownie

| Model/test | Przykłady/grupy | Accuracy | Macro F1 | Macro AUPRC |
|---|---|---|---|---|
| PTB-XL oficjalny fold10 |1958 ECG/1711 pacjentów|0.927477|0.792973|0.850302|
| MIT-BIH zamrożony test |23392 beaty/9 grup|0.836825|0.618694|0.690384|

PTB-XL precision/recall: SR97.22%/94.54%, AFIB84.21%/85.91%, STACH80.00%/90.00%, SBRAD42.71%/65.08%. SBRAD FP55. MIT-BIH precision/recall: N95.50%/85.23%, S17.86%/32.52%, V61.36%/88.02%, F57.10%/56.47%, Q55.22%/85.55%; F TP205/FP154/FN158. Są to różne zadania i kohorty, bez wspólnej tabeli rankingu modeli.

MIT-BIH F i Q w teście są prawie w całości skoncentrowane w pojedynczych grupach. Wynik globalny maskuje bardzo słabe działanie u niektórych pacjentów. W24A historyczny QC niezamierzenie odczytał9 testowych shardów, bez użycia do treningu/scalingu/inferencji/selekcji; nie deklarujemy nigdy wcześniej nieodczytanego testu. Incydent ujawniono w26. Testy obu zadań są wykorzystane: nie służą już do doboru modelu/progów, prezentacyjnego tuningu ani kolejnej oceny.

## Preprocessing i kontrakt uploadu

### PTB-XL

12 odprowadzeń w kolejności `I,II,III,AVR,AVL,AVF,V1,V2,V3,V4,V5,V6` (zachować oryginalne nazwy). Pełne10s/100Hz/1000próbek. Butterworth HP0.5Hz i LP45Hz,order4,SOS zero-phase; bez notch50/60Hz przy100Hz. Z-score osobno dla rekordu i kanału,ddof0,epsilon1e-8. Jednostki po normalizacji bezwymiarowe. Funkcje istniejące: `load_rhythm_record`/`preprocess_rhythm_record` w `src/ptbxl_rhythm.py` dotyczą oficjalnych metadanych i rekordów PTB-XL, NIE dowolnego uploadu. Adapter arbitrary-upload, kontrola jednostek/leadów/fs i ewentualne resampling/windowing wymagają osobnego zaprojektowania i walidacji; nie zgadywać kanałów ani częstotliwości.

Kohorta czteroklasowa zawiera dokładnie jeden kod spośród wszystkich12 rozpoznanych kodów rytmu. Brak kodu nie oznacza SR. Oryginalne etykiety mogą współistnieć poza tą wybraną kohortą.

### MIT-BIH ECG i RR

Dwa kanały,360Hz, istniejące filtrowanie0.5–45Hz order4,zero-phase,bez notch, Z-score całego rekordu osobno per kanał. Nie stosować offset/scale ECG starego pacjenta do nowego uploadu: z checkpointu odczytać metodę, a per-record normalizację wykonać zgodnie z istniejącym kontraktem na bieżącym pełnym rekordzie. Nigdy nie normalizować ponownie gotowych eksportowanych okien. Kanał0/1 nie oznacza identycznych anatomicznych leadów we wszystkich rekordach; nie zamieniać ich milcząco.

Okno `[annotation-90,annotation+162)` ma252próbki i adnotację naindeksie90. `segment_heartbeats` używa podanych pozycji, nie wyszukuje nowych Rpeaków.

RR w kolejności `RR_FEATURE_NAMES`:
1. previous_rr_seconds;
2. preceding_baseline_rr_seconds: mediana do8 wcześniejszych interwałów, WYŁĄCZA bieżący previousRR;
3. previous_over_baseline;
4–6. odpowiadające maski dostępności0/1.

Użyć `src.mitbih_rr.rr_features` na rosnących pozycjach wszystkich poprawnych beatów w JEDNYM rekordzie, również boundary-skipped beatów. Nie łączyć rekordów/pacjentów, nie używać nextRR ani etykiet klasy jako wartości wejściowych. Pierwszy beat ma trzy braki, drugi tylko previousRR. `transform_rr(raw,valid,state)` imputuje TRAIN median, stosuje zapisany TRAIN mean/std do pierwszych3kolumn i dodaje nieskalowane maski. `state` pochodzi z checkpointu24B; nie fitować go na uploadzie ani testach.

Pełny anotowany MIT-BIH: ekspertowe timestampy dostępne. Surowy ciągły upload: wymagany osobny zwalidowany detektor QRS/beatów, którego obecnie NIE ma. Przeoczone/dodatkowe piki i jitter zmieniają RR oraz segmentację; skuteczność na ekspertowych adnotacjach nie dowodzi działania end-to-end. Izolowane okno nie dostarcza historiiRR ani per-record normalizacji. Całkowicie brakująceRR można technicznie zamaskować/imputować, ale nie jest to zwalidowany tryb wdrożeniowy. Cały preprocessing pozostaje offline mimo przyczynowych cechRR.

## Istniejący interfejs wag — przykład, nie gotowe API uploadów

Poniższy kod pokazuje faktyczne klasy i klucze wag. Nie jest nowym endpointem inferencji; wymaga już poprawnie przygotowanych tensorów. Nie uruchamiać go na finalnych zbiorach testowych.

```python
from pathlib import Path
import hashlib
import torch
from src.models.ptbxl_cnn import PTBXLRhythmCNN
from src.models.mitbih_cnn_rr import MITBIHBeatCNNRR

def load_local_model(path, expected_sha256, model_class):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Checkpoint SHA256 mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = model_class()
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, saved

# PTB-XL: logits = model(X), X float32 [B,12,1000].
# MIT-BIH: logits = model(X, RR), X float32 [B,2,252], RR float32 [B,6].
# with torch.inference_mode(): predictions = logits.argmax(dim=1)
# Nie wywołano tu żadnego modelu ani nie załadowano checkpointów.
```

Lazy loaders istnieją dla przygotowanych eksportów: `PTBXLRhythmDataset` oraz `MITBIHBeatDataset`; `MITBIHRRDataset` ma świadomie tylko train/validation. Nie wyłączać test-locków w UI. Eksporty/manifesty i duże dane nie są częścią Git, więc sam clone nie udostępnia przykładów ani wag.

## Lokalizacja danych i osobny transfer

- MIT-BIH: `data/processed/ecg_dataset_v1/`,109452 beaty,48rekordów/47grup; RR v1: `data/processed/mitbih_rr_context_v1/` (train/validation).
- PTB-XL: `data/processed/ptbxl_rhythm_v1/runs/90b449bf54b649509519747f417b9e5e/`,19605rekordów/17223pacjentów; kanoniczny manifest `PTBXL_DOWNLOAD_MANIFEST.json` wewnątrz tego runu.
- Surowe i przetworzone dane, oryginalne metadane kliniczne, pełne manifesty rekordowe i per-sample predykcje pozostają lokalne. Ewentualny transfer danych osobno według warunków datasetu i ustaleń projektu; nie przez publiczne repo. Do samego UI nie potrzeba kopii zbiorów final-test.

Publiczne techniczne ID rekordów/grup w konfiguracji splitu nie są tożsamościami osób. Mimo to nie publikować uploadów ani surowych rekordów/predykcji pacjentowych. Metadane PTB-XL CSV (także kliniczne opisy/demografia) nie są dołączane do nowego drzewa repo.

## Zadania Anny

Zintegrować istniejącą aplikację z main z tymi konkretnymi checkpointami, zachowując osobne kontrakty i dorobek `src/ml/`: jawny wybór zadania, upload/odczyt i kontrolę kontraktu, obsługę błędów, zarządzanie lokalnym checkpointem/cache, wywołanie modelu na właściwych tensorach, czytelne wykresy/etykiety i informację o ograniczeniach. Zaprojektować bezpieczną obsługę uploadów, bez utrwalania danych w Git i bez logowania surowych danych użytkowników. Oddzielić demonstrację istniejących preprocessed inputs od niezwalidowanego workflow raw-upload. Integracja detektora i adaptacja nowych formatów wymagają osobnego uzgodnienia/walidacji. Nie zmieniać wag, progów, preprocessingu lub labelmappingu w celu poprawy wyglądu wyników. Nie uruchamiać przy starcie UI skryptów treningowych ani evaluatorów final-test.

To projekt badawczy, nie klinicznie zwalidowane urządzenie diagnostyczne. Słabe minority recognition, dominacja klas/pacjentów, domain shift, lead variability i brak kalibracji ograniczają prezentowane wyniki. Nie przedstawiać decyzji jako diagnozy.
