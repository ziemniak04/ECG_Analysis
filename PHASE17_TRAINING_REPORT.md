# Faza 17 — pipeline treningu i krótki baseline CPU PTB-XL

**Trening działa na rzeczywistych ECG. Wykonano jeden krótki eksperyment: 2 epoki i 16 aktualizacji Adam.** Wyniki są wstępne, z małych równo licznych podzbiorów klas. Nie opisują jakości na pełnych foldach, nie są wynikiem finalnego testu ani deklaracją skuteczności klinicznej. Wybrany model nie rozpoznał żadnego SR ani AFIB w tej małej walidacji; nie nadaje się obecnie do prezentowania jako gotowy klasyfikator użytkowy.

## Zachowane dane i architektura

Wykorzystano istniejące `PTBXLExportIndex`, `PTBXLRhythmDataset` i `PTBXLRhythmCNN`, bez modyfikowania ich implementacji. Model fazy 16: **74116 parametrów**, wejście float32 `[B,12,1000]`, wyjście `[B,4]` logitów. Klasy: SR=0, AFIB=1, STACH=2, SBRAD=3. PyTorch działa w wersji 2.15.0+cpu.

Eksport `data/processed/ptbxl_rhythm_v1/`, ukończony run `90b449bf54b649509519747f417b9e5e`. Dane są już przefiltrowane i znormalizowane per-record/per-lead; nie wykonywano preprocessingu ponownie, nie zmieniono eksportów MIT-BIH/PTB-XL i nie pobierano danych.

Foldy 1–8: 15689 rekordów, 13749 pacjentów; fold 9: 1958 rekordów, 1763 pacjentów; fold 10: 1958 rekordów, 1711 pacjentów. Weryfikacja metadanych potwierdziła brak przecinania pacjentów. Waveformy foldu 10 nie były używane przez nowy trening, ewaluację, checkpoint selection ani dobór hiperparametrów. Pipeline nie tworzy test Dataset/DataLoader; odczyt metadanych całej kohorty służy wyłącznie kontroli protokołu.

## Dokładny eksperyment

| Ustawienie | Wartość |
|---|---|
| Urządzenie / wątki / workers | CPU / 1 / 0 |
| Seed | 42, Python/NumPy/PyTorch, deterministic algorithms |
| Training subset | 128 rzeczywistych ECG, 128 pacjentów, po 32/klasę, foldy 1–8 |
| Validation subset | 64 rzeczywiste ECG, 64 innych pacjentów, po 16/klasę, fold 9 |
| Wybór | deterministyczny SHA256(seed, partition, ecg_id) w każdej klasie; bez kryteriów wizualnych/wynikowych |
| Batch / epoki | 16 / 2 |
| Partie na epokę | 8 train + 4 validation |
| Łącznie | 16 partii treningowych = 16 aktualizacji; 8 partii walidacyjnych |
| Optymalizator | Adam, lr=0.001, bez weight decay |
| Loss | CrossEntropyLoss, surowe logity, bez Softmax przed loss |
| Class weighting | **wyłączone** |
| Best model | validation Macro F1, wcześniejsza epoka przy remisie |
| Early stopping | patience=5, min_delta=0; nie uruchomił zatrzymania w dwóch epokach |

Wybrano podzbiory osobno z dopuszczalnych foldów, bez losowego splitu. Ich pełne ID, globalne indeksy, pacjenci, foldy i SHA256 źródłowego manifestu zapisano w selection_manifest.json. Wszystkie wybrane rekordy zostały użyte w każdej epoce; nie pominięto żadnej partii wskutek cap. Dodatkowo jedna partia walidacyjna (16 rekordów, wszystkie fold 9) posłużyła do dwóch identycznych forward passów sprawdzających przeładowanie najlepszego stanu. Nie był to dodatkowy krok treningu ani odrębny eksperyment jakościowy.

Równa liczba klas jest własnością tego jawnego podzbioru, nie zmianą całego datasetu. Oryginalne częstości są inne, więc metryk nie wolno przenosić na całą kohortę. Nie włączono oversamplingu/undersamplingu bazy ani augmentacji. Opcjonalne class weights pipeline liczy wyłącznie przez istniejącą funkcję fazy 16 z faktycznie wybranych indeksów treningowych; dla równo licznego baseline wyniosłyby po 1. Nie obliczano wag z foldów 9/10.

Przed wykonaniem drukowano konfigurację, liczebności, urządzenie i liczbę partii; preflight: `results/model_training/phase17_preflight/dry_run.log`. Trening: `results/model_training/phase17_baseline/training.log` i zachowany log wywołania baseline_execution.log w katalogu preflight. Nie wykonywano prób alternatywnych seedów lub hiperparametrów na rzeczywistych danych. Testy w katalogach tymczasowych korzystają z kontrolowanych fixture'ów, nie stanowią kolejnych klinicznych eksperymentów.

## Wyniki wstępne — wyłącznie te podzbiory

| Epoka | Train loss (online) | Validation loss | Train accuracy | Validation accuracy | Validation Macro F1 | Validation Macro AUPRC/AP |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.347445 | 1.356680 | 0.335938 | 0.500000 | 0.334641 | 0.601453 |
| 2 | 1.184685 | 1.302055 | 0.562500 | 0.343750 | 0.243111 | 0.619216 |

Wybrano **epokę 1**: Macro F1 **0.334641**, Macro AUPRC **0.601453**, accuracy **0.500000**. Choć loss i AP poprawiły się w epoce 2, checkpoint wybrano zgodnie z ustalonym kryterium Macro F1, a nie przez zmianę celu po obejrzeniu wyników. Dwie epoki to kontrola infrastruktury, nie dostateczne badanie uczenia lub generalizacji.

Najlepszy model — metryki per klasa na 64 rekordach walidacyjnych:

| Klasa | Support | Recall / sensitivity | Specificity | Precision | F1 | AUPRC/AP |
|---|---:|---:|---:|---:|---:|---:|
| SR | 16 | 0.000000 | 1.000000 | niezdefiniowana (brak predykcji) | 0.000000 | 0.245572 |
| AFIB | 16 | 0.000000 | 1.000000 | niezdefiniowana (brak predykcji) | 0.000000 | 0.401205 |
| STACH | 16 | 1.000000 | 0.729167 | 0.551724 | 0.711111 | 0.883859 |
| SBRAD | 16 | 1.000000 | 0.604167 | 0.457143 | 0.627451 | 0.875177 |

SR i AFIB nie otrzymały żadnej predykcji argmax: precision jest **niezdefiniowana**, recall i F1 wynoszą 0. Wszystkie klasy występują w true labels, po 16 przykładów, więc Macro F1 i AP są zdefiniowane. Nie należy utożsamiać wysokiej specificity dla nieprzewidywanych klas z dobrym rozpoznawaniem. Różnica między AP (ranking prawdopodobieństw) a recall argmax jest oczekiwana; nie dobierano progów w celu poprawienia wyniku.

Macierz pomyłek: wiersze true, kolumny predicted, kolejność SR/AFIB/STACH/SBRAD:

```text
  0   0   1  15
  0   0  12   4
  0   0  16   0
  0   0   0  16
```

Średni czas forward najlepszego pomiaru walidacyjnego: **3.984607 ms/rekord**, CPU, batch 16, 1 wątek. Liczone wyłącznie model(x), z wyłączeniem NPZ/I/O, transferu, Softmax/loss i obliczania metryk. To pojedynczy pomiar techniczny zależny od urządzenia i obciążenia, nie gwarantowany czas aplikacji. Trening zakończył się przed uruchomieniem pełnej sesji pytest.

## Definicje i wykresy

Definicje wszystkich metryk, null przy braku klasy/predykcji, ważenie loss i zastrzeżenia dostępne w PTBXL_TRAINING_GUIDE.md. Macro AUPRC oznacza tu średnie nieinterpolowane Average Precision z prawdopodobieństw one-vs-rest; nie trapezoidalne PR-AUC. Softmax służy tylko metrykom. Macierz ma stałe cztery klasy. Training metrics pochodzą z logits przed kolejnymi aktualizacjami, przy aktywnym Dropout i treningowym BN; validation jest epoch-end eval/inference_mode.

![Loss curves — preliminary](results/model_training/phase17_baseline/loss_curves.png)

Krzywe loss obejmują te same dwa podzbiory, ale train jest pomiarem online, val pomiarem po zakończonej epoce.

![Validation Macro F1 — preliminary](results/model_training/phase17_baseline/validation_macro_f1.png)

Macro F1 na fold-9 subset spada między epokami; zachowano checkpoint epoki 1.

![Validation confusion matrix — preliminary](results/model_training/phase17_baseline/validation_confusion_matrix.png)

W tym małym eksperymencie predykcje skupiają się na STACH/SBRAD. Figura nie opisuje końcowego testu ani pełnej populacji.

## Checkpointy i poprawność

Najlepszy: **`results/model_training/phase17_baseline/best_model.pt`**, SHA256 `6e50c44b372e45b87d98bc201ff5a8ef37ce3e01aab87874fd6a53fa85c6a4ee`. Checkpointy ukończonych epok: `checkpoints/epoch_001.pt`, `epoch_002.pt`. Zawierają model/Adam/RNG, historię, stan early stopping, najlepszy stan oraz dokładny plan/kod/configuration hash. Weryfikacja reload porównała logity identycznie na tej samej partii walidacyjnej. Najlepszy stan jest dokładnie stanem epoki 1, a nie ostatnim modelem przebranym za najlepszy.

Osobny skrypt verify_baseline.py odtworzył CM, accuracy, precision/recall/specificity/F1 i AP z zapisanych labels/probabilities. AP policzono niezależną pętlą po progach. Loss sprawdzono przez -log(p_true) (nieważony baseline). Zweryfikowano, że każdy przykład ma właściwą etykietę/fold/pacjenta, optymalizator ma kroki 8/16, parametry są skończone i najlepszy stan odpowiada maksymalnemu validation F1. Sprawdzono SHA256 wszystkich 192 użytych eksportów. Brak przecinania pacjentów, brak foldu 10 i brak zmian preprocessing/configuration. Wynik: `results/model_training/phase17_preflight/baseline_verification.json`.

## Testy i zachowanie wcześniejszej pracy

Pełna zakończona sesja: **532 passed, 0 failed, 0 skipped, 229 warnings**, 202.96 s. Obejmuje 510 wcześniejszych testów i 22 nowe testy metryk/treningu. Ostrzeżenia pozostają deprecacjami pandas/WFDB; nie zostały wyciszone. Pierwsza próba pełnej sesji została zakończona sygnałem SIGTERM (exit 143) bez pełnego podsumowania; zachowano log, a zakończony retry jest źródłem powyższych liczb. Nie uruchamiano ponownie rzeczywistego treningu.

Testy: znane metryki i remisy, null dla absent classes, selekcja deterministyczna, zakaz test fold, skończoność/aktualizacje, poprawność wag i weighted denominator, checkpoint reload, zgodność bitowa wznowionej i nieprzerwanej ścieżki w kontrolowanych fixture'ach, early stopping oraz przerwanie po ukończonej epoce. Weryfikowano eval-mode i brak zmian buforów BN w walidacji, brak odczytu nadmiarowej partii po cap i odmowę nadpisania istniejącego katalogu.

Polecenie końcowej sesji: `MPLCONFIGDIR=/tmp/ecg_phase17_mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m pytest -ra`. Log: full_pytest_retry.log w katalogu preflight. Statyczne `git diff --check` również przechodzi.

Porównano SHA256 wcześniejszego kodu, modeli fazy 16, raportów i manifestów względem snapshotu przed fazą 17: zmieniono wyłącznie `.gitignore` (dodano ignorowanie results/model_training/). Nie zmieniono datasetów ani dotychczasowego modelu/loadera. Poprzednie niezatwierdzone zmiany użytkowniczki na feature/ptbxl-1dcnn zachowano; nie wykonano commit, reset ani force-push. `git check-ignore` potwierdza wykluczenie checkpointów i nowych wyników.

## Pliki i następny krok

Nowe moduły: src/ptbxl_metrics.py, src/ptbxl_training.py; CLI examples/train_ptbxl_baseline.py; configs/ptbxl_phase17_baseline.json; tests/test_ptbxl_metrics.py, tests/test_ptbxl_training.py. Dokumentacja: niniejszy raport, PTBXL_TRAINING_GUIDE.md, PHASE18_FULL_TRAINING_PLAN.md. Wszystkie wymagane wyniki są w results/model_training/phase17_baseline/; audyty i logi testów osobno w phase17_preflight/.

Przed pełnym treningiem trzeba zatwierdzić osobny budżet/konfigurację fazy 18, zachować pełne oficjalne train/validation, ustalić early stopping i ewentualne wagi wyłącznie z treningu oraz zamrozić środowisko. Fold 10 pozostaje zamknięty do odrębnej końcowej oceny po zamrożeniu decyzji. Obecny checkpoint ma charakter eksperymentalny. Nie rozpoczęto pełnego treningu, MIT-BIH CNN, Streamlit ani fazy 18.
