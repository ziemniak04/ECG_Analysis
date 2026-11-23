# Phase 18 — pełny trening PTB-XL: raport końcowy

Zatwierdzony eksperyment zakończył się poprawnie po **8 epokach** przez **early stopping**.
Wybrano **epokę 4** na podstawie najwyższego Macro F1 całego foldu walidacyjnego 9.
**Nie odczytywano sygnałów z foldu 10 i nie obliczano metryk testowych.**

## Dane i integralność

Użyto istniejącego eksportu `data/processed/ptbxl_rhythm_v1/` i ukończonego manifestu
`runs/90b449bf54b649509519747f417b9e5e/PTBXL_DOWNLOAD_MANIFEST.json`.
Indeks potwierdził 19605 rekordów i 17223 pacjentów, spójność z oryginalną metadaną
PTB-XL 1.0.3, oryginalnymi etykietami, identyfikatorami i oficjalnymi foldami.

| Zakres | Foldy | ECG | Pacjenci | SR | AFIB | STACH | SBRAD |
|---|---|---:|---:|---:|---:|---:|---:|
| Trening | 1–8 | 15689 | 13749 | 13354 | 1189 | 643 | 503 |
| Walidacja | 9 | 1958 | 1763 | 1667 | 146 | 82 | 63 |

Podziały pacjentów są rozłączne, również względem testu (sprawdzono metadane,
bez odczytu sygnałów testowych). W każdej epoce wykonano **981 batchy treningowych**
i **123 walidacyjne**, uwzględniając ostatnie niepełne batche: dokładnie 15689 i 1958 ECG.
Łącznie wykonano **7848 aktualizacji Adam**. Nie zastosowano limitów batchy ani podzbiorów.

Przed treningiem zweryfikowano wszystkie **17647 NPZ treningowych i walidacyjnych**:
SHA256, `[12,1000]`, dtype, skończone wartości oraz zgodność etykiet, identyfikatorów,
foldów, kanałów i konfiguracji z eksportem. Loader ponawiał te kontrole podczas odczytu.
Wykonanie dodatkowo blokowało odczyt sygnałów foldu 10; liczba takich odczytów i prób: **0**.

Końcowy audyt potwierdził:

- te same identyfikatory i liczebności danych w każdej epoce;
- zgodność metryk wszystkich epok z ponownym przeliczeniem zapisanych predykcji;
- poprawny wybór najlepszej epoki i zgodność tensorów najlepszego modelu z checkpointem;
- identyczne logity po ponownym wczytaniu modelu na tym samym batchu walidacyjnym;
- brak zmian rozmiaru i mtime wszystkich **19668 plików obu eksportów**;
- niezmienność SHA256 **114 chronionych plików** kodu, konfiguracji, dokumentacji
  i wcześniejszych wyników, w tym artefaktów Phase 17.

Kontrola rozmiarów/mtime eksportów nie jest odrębnym pełnym haszowaniem całego MIT-BIH.
Pełne SHA256 sprawdzano dla wszystkich użytych NPZ PTB-XL; nie powtarzano preprocessingu.
Szczegóły i sumy kontrolne artefaktów: [final_integrity.json](results/model_training/phase18_full/audit_run_001/final_integrity.json).

## Dokładna konfiguracja

Użyto bez zmian [configs/ptbxl_phase18_full_v1.json](configs/ptbxl_phase18_full_v1.json).
Jej SHA256: `128ec326947d7d86d740f42e6f2ee3f209e0efa316d3a3931d267dd1771d595c`.
Plan wykonania jest identyczny z zatwierdzonym w Checkpoint A.

- Istniejący 1D-CNN: **74116 parametrów**, wejście `[B,12,1000]`, cztery logity.
- Adam, learning rate **0.001**, batch size **16**, CPU, **2 wątki**, seed **42**.
- Maksymalnie 15 epok, patience **4**, min_delta **0**, wybór przez pełnowalidacyjne Macro F1.
- CrossEntropyLoss bez Softmax przed funkcją straty, z wagami klas.
- Wagi `N_train / (4 * n_class)` obliczono wyłącznie z foldów 1–8:
  `[0.2937134802341461, 3.2987804412841797, 6.099922180175781, 7.797713756561279]` w kolejności SR, AFIB, STACH, SBRAD.
- Te same wagi treningowe zastosowano do raportowania straty walidacyjnej;
  metryki klasyfikacyjne są nieważone i dotyczą naturalnej liczebności klas.
- Brak ponownego filtrowania, normalizacji, zmian etykiet, oversamplingu lub undersamplingu.
- Zachowano dotychczasową normalizację per rekord/per odprowadzenie z gotowego eksportu.

Środowisko: Python **3.12.3**, PyTorch **2.15.0+cpu**, NumPy **2.5.4**.
Pełne wersje zależności, seed, deterministyczne ustawienia i SHA checkpointów zapisano w
[reproducibility.json](results/model_training/phase18_full/run_001/reproducibility.json).
Deterministyczność dotyczy tej konfiguracji i środowiska; różne wersje bibliotek lub sprzęt
nie gwarantują wyników identycznych bitowo.

## Historia treningu

Czas wywołania treningu wraz z końcowym zapisem artefaktów: **3249.76 s = 54 min 9.76 s**.
Suma czasów pełnych epok: **3235.54 s**. Kontrola wstępna 17647 NPZ zajęła dodatkowo
**325.22 s** i nie jest wliczona w podany czas treningu. Testy i audyt końcowy również są osobne.
Trening nie był przerywany ani wznawiany.

| Epoka | Czas [s] | Train loss | Validation loss | Val accuracy | Val Macro F1 | Val Macro AUPRC |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 605.27 | 0.674911 | 0.409015 | 0.755363 | 0.599652 | 0.750696 |
| 2 | 420.45 | 0.411338 | 0.323752 | 0.852911 | 0.694919 | 0.830297 |
| 3 | 367.70 | 0.357153 | 0.353043 | 0.853933 | 0.697803 | 0.830205 |
| 4 | 372.81 | 0.342754 | 0.311731 | 0.901941 | 0.757428 | 0.857313 |
| 5 | 364.95 | 0.297546 | 0.286889 | 0.831971 | 0.674371 | 0.848839 |
| 6 | 366.03 | 0.285014 | 0.301513 | 0.806946 | 0.697427 | 0.851384 |
| 7 | 368.57 | 0.284495 | 0.253102 | 0.883044 | 0.736241 | 0.874605 |
| 8 | 369.77 | 0.248969 | 0.262966 | 0.884065 | 0.743475 | 0.855401 |

Czasy obejmują utworzenie loaderów, trening, walidację oraz zapis checkpointu i metryk.
Oddzielne czasy treningu i walidacji znajdują się w [epoch_durations.csv](results/model_training/phase18_full/run_001/epoch_durations.csv).
Po epokach 5–8 Macro F1 nie przekroczyło wyniku epoki 4, więc licznik osiągnął 4 i trening
zakończył się zgodnie z zatwierdzoną regułą. Niższa strata lub wyższe AUPRC w innej epoce
nie zmieniają ustalonego kryterium wyboru modelu.

Metryki treningowe liczone są z predykcji powstających podczas kolejnych aktualizacji modelu
(online, przed aktualizacją danego batcha, z aktywnym dropout/BatchNorm w trybie treningowym).
Walidacja używa stałego modelu po epoce, `eval()` i `inference_mode()`. Nie są to identyczne
warunki pomiaru. Nie wykonano dodatkowej oceny całego zbioru treningowego po epoce.

## Najlepszy model: epoka 4, cały fold 9

- **Macro F1: 0.757428426**.
- **Macro AUPRC: 0.857312651**.
- **Accuracy: 0.901940756 = 90.19%** (1766/1958).
- Ważona validation loss: **0.311731107**.

| Klasa | Liczba ECG | Recall / sensitivity | Specificity | Precision | F1 | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| SR | 1667 | 0.902220 | 0.938144 | 0.988173 | 0.943242 | 0.995692 |
| AFIB | 146 | 0.931507 | 0.975717 | 0.755556 | 0.834356 | 0.867809 |
| STACH | 82 | 0.914634 | 0.974414 | 0.609756 | 0.731707 | 0.871629 |
| SBRAD | 63 | 0.809524 | 0.956728 | 0.383459 | 0.520408 | 0.694120 |

Confusion matrix: wiersze = prawdziwa klasa, kolumny = predykcja.

| Prawdziwa / predykcja | SR | AFIB | STACH | SBRAD |
|---|---:|---:|---:|---:|
| SR | 1504 | 40 | 44 | 79 |
| AFIB | 3 | 136 | 4 | 3 |
| STACH | 3 | 4 | 75 | 0 |
| SBRAD | 12 | 0 | 0 | 51 |

Najbardziej widoczne ograniczenie: **SBRAD precision 38.35%**, mimo recall 80.95%.
Spośród 133 predykcji SBRAD tylko 51 jest poprawnych; 79 błędnych predykcji dotyczy SR,
a 3 AFIB. Walidacja zawiera jedynie 63 prawdziwe SBRAD, więc jej wyniki wymagają ostrożności.
Accuracy jest zdominowana przez liczną klasę SR; sama nie opisuje rozpoznawania arytmii.

Recall = TP/(TP+FN), specificity = TN/(TN+FP), precision = TP/(TP+FP),
F1 = 2TP/(2TP+FP+FN). Metryki per klasa wykorzystują one-vs-rest.
Macro F1 to średnia czterech F1. Macro AUPRC jest średnią nieważoną
**nieinterpolowanego average precision** czterech klas one-vs-rest, z grupowaniem remisów
prawdopodobieństw; nie jest polem liczonym metodą trapezów.
Wszystkie cztery klasy występują w pełnym treningu i walidacji.

Zmierzony czas samego forward pass walidacji najlepszego modelu wynosił
**2.965 ms/ECG**, amortyzowany dla batch size 16 na CPU.
Pomiar nie obejmuje odczytu NPZ, przygotowania batcha ani obliczania metryk i nie oznacza
opóźnienia pojedynczego żądania aplikacji.

## Porównanie z Phase 17 i znaczenie ważenia klas

Phase 17 używała jedynie 128 ECG treningowych i 64 walidacyjnych, 2 epok oraz nieważonej
straty. Najlepsze Macro F1 wynosiło **0.334641**, recall: SR 0, AFIB 0, STACH 1, SBRAD 1.
Obecnie Macro F1 wynosi **0.757428**, a recall odpowiednio **0.902220, 0.931507, 0.914634,
0.809524**. Model rozpoznaje teraz SR i AFIB; recall STACH i SBRAD jest niższe niż na
małym, wcześniejszym podzbiorze. Nie można zatem twierdzić, że recall wszystkich
mniejszościowych klas wzrosło.

**To nie jest kontrolowane porównanie wpływu wag klas.** Zmieniły się rozmiar i skład
zbiorów, rozkład klas, liczba epok i ważenie straty. Wynik pokazuje działanie pełnego
ważonego eksperymentu, ale nie dowodzi, że poprawę spowodowało samo ważenie.
Nie wykonano dodatkowego eksperymentu bez wag ani dobierania parametrów po wynikach.

## Pliki wynikowe i checkpointy

Wszystkie wyniki: [results/model_training/phase18_full/run_001/](results/model_training/phase18_full/run_001/).

- [best_model.pt](results/model_training/phase18_full/run_001/best_model.pt) — wybrany model z epoki 4.
- [last_resumable_checkpoint.pt](results/model_training/phase18_full/run_001/last_resumable_checkpoint.pt) — pełny stan końca epoki 8,
  Adam, RNG, historia, konfiguracja i stan early stopping; hardlink do `checkpoints/epoch_008.pt`.
- `checkpoints/epoch_001.pt` do `epoch_008.pt` — pełne stany ukończonych epok.
- `training_config.json`, `selection_manifest.json`, `architecture.json`, `reproducibility.json`.
- `epoch_001_metrics.json` do `epoch_008_metrics.json` — pełne metryki i predykcje wszystkich epok.
- `epoch_metrics.csv`, `epoch_metrics_full.csv`, `epoch_durations.csv`.
- `per_class_epoch_metrics.csv`, `best_validation_per_class.csv`, `best_validation_metrics.json`.
- `best_validation_predictions.csv`, `validation_confusion_matrix.csv`.
- `result.json`, `full_run_summary.json`, `training.log`.

![Straty treningowe i walidacyjne](results/model_training/phase18_full/run_001/loss_curves.png)

![Walidacyjne Macro F1](results/model_training/phase18_full/run_001/validation_macro_f1.png)

![Walidacyjne Macro AUPRC](results/model_training/phase18_full/run_001/validation_macro_auprc.png)

![Macierz pomyłek wybranego modelu](results/model_training/phase18_full/run_001/validation_confusion_matrix.png)

SHA256 najlepszego modelu: `e07b51c152d864845ad368e8c38a1741446e3b81f1da15bf48491b7d79a53310`.
SHA256 ostatniego checkpointu: `4e1c910c4929b1cb70f29f97bcab05a836eb963efd7658fdedad7e2082de75fb`.
Pełna lista sum artefaktów znajduje się w `audit_run_001/final_integrity.json`.
Wygenerowane wyniki i checkpointy są objęte istniejącymi regułami `.gitignore`.

## Bezpieczeństwo wykonania i wznowienia

Dodano wyłącznie nakładkę audytową [src/ptbxl_full_run.py](src/ptbxl_full_run.py),
CLI [examples/run_phase18_full.py](examples/run_phase18_full.py), walidator końcowy
[examples/validate_phase18_full.py](examples/validate_phase18_full.py) i
[tests/test_ptbxl_full_run.py](tests/test_ptbxl_full_run.py).
Nie zmieniono istniejącego trenera, modelu, loaderów, preprocessingu, eksportów ani raportów.
Nakładka deleguje obliczenia do zweryfikowanego trenera i dodaje czasy, kontrolę foldów,
status i dodatkowe tabele/wykres AUPRC.

Istniejący trener zapisuje checkpoint atomowo po każdej pełnej epoce. Wznowienie po
rzeczywistym przerwaniu wymaga **nowego** katalogu wyników i tej samej konfiguracji,
metadanych oraz hashy kodu. Przywracane są Adam i RNG, a niepełna epoka jest odtwarzana.
Opcje CLI: `--resume-from`, `--prior-audit-dir`, `--output-dir`, `--audit-dir`;
nowy katalog audytu wymaga uprzedniego `--preflight-only` z tymi samymi ścieżkami.
Nie należy nadpisywać istniejącego katalogu ani traktować ponownego uruchomienia od
seed 42 jako wznowienia.

**Ten eksperyment został zakończony regułą early stopping.** Jego końcowy checkpoint
celowo nie pozwala kontynuować epok z pominięciem tego warunku. Żaden kolejny trening
nie został uruchomiony. Zapisane stany pozostają dostępne do audytu i odtworzenia modelu.

## Testy i dalszy etap

Po treningu wykonano **73 testy: wszystkie przeszły**, z **2 ostrzeżeniami deprecacyjnymi
WFDB/pandas** dotyczącymi `DataFrame.set_index(..., inplace=True)`. Ostrzeżeń nie tłumiono;
nie były błędami treningu. To ukierunkowany zestaw Phase 16–18, nie ponowne uruchomienie
całego repozytorium.

```bash
MPLCONFIGDIR=/tmp/ecg_phase18_full_mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/test_ptbxl_full_run.py tests/test_ptbxl_readiness.py \
  tests/test_ptbxl_metrics.py tests/test_ptbxl_training.py tests/test_ptbxl_protocol.py \
  tests/test_ptbxl_torch.py -ra
```

Log: [relevant_tests_after_training.log](results/model_training/phase18_full/audit_run_001/relevant_tests_after_training.log).
Kontrole obejmują m.in. architekturę, forward/backward, metryki, ważoną stratę,
wybór foldów i pacjentów, deterministyczność, checkpoint/reload, przerwanie/wznowienie
oraz zakaz odczytu testowych sygnałów w nakładce Phase 18.
Audyt rzeczywistych artefaktów zakończył się osobno statusem **passed**.

Następny etap wymaga osobnego zatwierdzenia: zamrozić wybrany checkpoint, etykiety,
preprocessing i regułę argmax, a następnie zaplanować jednorazową ocenę na dotąd
niewykorzystanym foldzie 10 z metrykami per klasa i analizą ograniczeń. Nie dobierać
hiperparametrów na foldzie 10. Wyniki powyżej są walidacyjne, wybrane w procesie uczenia,
więc nie stanowią końcowej niezależnej oceny ani deklaracji skuteczności klinicznej.
Nie rozpoczęto kolejnej fazy, Streamlit ani treningu MIT-BIH.
