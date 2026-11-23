# PTB-XL — pipeline treningu i krótki baseline, faza 17

Magdalena odpowiada obecnie za modelowanie/trening/ewaluację, Anna głównie za Streamlit i integrację. Używamy niezmienionego `PTBXLRhythmCNN` z fazy 16 (74116 parametrów), istniejącego `PTBXLRhythmDataset` i gotowych NPZ. W tej fazie wykonujemy wyłącznie krótki techniczny eksperyment. Wyniki są wstępne i nie opisują jakości klinicznej ani pełnej kohorty.

## Dane i pacjenci

Eksport: `data/processed/ptbxl_rhythm_v1/`. Kompletny manifest: `runs/90b449bf54b649509519747f417b9e5e/PTBXL_DOWNLOAD_MANIFEST.json`. 19605 ECG / 17223 pacjentów; X float32 `[12,1000]`, y torch.long 0=SR, 1=AFIB, 2=STACH, 3=SBRAD. Nie filtrujemy ani nie normalizujemy ponownie sygnałów. NPZ są ładowane na żądanie z kontrolą SHA256 i metadanych.

| Partycja | Oficjalne foldy | ECG | Pacjenci |
|---|---|---:|---:|
| Training | 1–8 | 15689 | 13749 |
| Validation | 9 | 1958 | 1763 |
| Final test — niewykorzystywany | 10 | 1958 | 1711 |

Zachowujemy wszystkie oryginalne strat_fold. Indeks weryfikuje, że pacjent ma jeden fold. Pipeline fazy 17 nie tworzy Dataset ani DataLoader dla testu. Dostęp do metadanych całej kohorty służy sprawdzeniu integralności i podziału; nie oznacza odczytu waveformów testowych ani ich oceny.

## Podzbiory krótkiego eksperymentu

Domyślna konfiguracja: `configs/ptbxl_phase17_baseline.json`.

- Train: 32 rekordy każdej klasy = 128 ECG. Validation: 16 każdej = 64 ECG.
- W każdej klasie sortujemy według SHA256 reprezentacji JSON `[seed, partition, ecg_id]`, z ecg_id jako rozstrzygnięciem remisu; wybieramy pierwsze n i potem porządkujemy po indeksie źródłowym.
- Seed=42, brak selekcji według morfologii, jakości lub wyników modelu. Brak duplikowania rekordów. Podzbiory train i validation wybierane osobno z dopuszczalnych foldów; sprawdzane są pacjenci i ID.
- Równa liczba klas zmienia częstości względem naturalnej kohorty z dominującym SR. Accuracy, F1 i AP tego małego zbioru nie są oszacowaniem wyników na całym foldzie 9 ani populacji.
- 2 epoki, batch=16, limit 8 partii treningowych i 4 walidacyjnych/epokę. Domyślnie obejmuje to cały wybrany podzbiór. Przy innych limitach metryki opisują tylko faktycznie przetworzone rekordy; epoch JSON zawiera ich ID i liczebności. Brak pokrycia klasy jest jawny.

## Trening

Implementacja: `src/ptbxl_training.py`; metryki: `src/ptbxl_metrics.py`; CLI: `examples/train_ptbxl_baseline.py`.

Adam, lr=0.001, bez weight decay; CrossEntropyLoss na **surowych logitach** `[B,4]`. Nie ma Softmax przed loss. Softmax liczony jest dopiero do metryk one-vs-rest. Wagi klas są opcjonalne, w domyślnym baseline wyłączone. Po włączeniu korzystamy z istniejącego `training_class_weights(index, selected_train_indices)`, n_train/(4*n_class), wyłącznie dla wybranych danych treningowych. Dla pełnego treningu źródłem byłoby całe 1–8, bez foldów 9/10. Na równo licznych podzbiorach wagi wynoszą po 1; nie jest to dodatkowy eksperyment porównawczy.

Epoch loss = suma numeratorów CE / suma denominatorów. Bez wag denominator to liczba przykładów; z wagami suma wag prawdziwych etykiet. Pozwala to uniknąć błędnego uśredniania batch means przy różnych wielkościach lub rozkładach klas. W loss walidacji można używać tych samych, już wyznaczonych wag treningowych; nie estymujemy ich na walidacji.

Training działa z `model.train()`, gradientami i optimizer.step. Validation z `model.eval()` i `torch.inference_mode()`, bez aktualizacji BatchNorm, Dropout i parametrów. Training metrics korzystają z kolejnych logits przed aktualizacją batcha, a nie z osobnego pomiaru końcowego modelu na train. Loss train i val nie są bezpośrednio równoważnymi pomiarami tego samego stanu sieci.

Wszystkie partie są sprawdzane pod kątem foldów, dtype, kształtu, klas 0–3, skończonych logitów/loss/gradientów i parametrów po update. Seed ustawiany dla Python, NumPy i PyTorch; shuffle treningu używa osobnego generatora seed+epoch. CPU jest domyślne (1 wątek, workers=0), deterministic algorithms włączone. CUDA jest dostępne wyłącznie po jawnej konfiguracji, `cuda.is_available()` i wykonaniu rzeczywistej operacji kontrolnej. Ten eksperyment uruchomiono na CPU. Determinizm nie jest gwarantowany między różnymi wersjami bibliotek, urządzeniami i backendami.

## Definicje metryk

Macierz pomyłek ma stałą kolejność SR/AFIB/STACH/SBRAD: **wiersze true, kolumny predicted**. Decyzja = argmax logitów/probabilities, bez dostrajania progów.

- Accuracy = suma przekątnej / N.
- Dla klasy one-vs-rest: recall/sensitivity = TP/(TP+FN); specificity = TN/(TN+FP); precision = TP/(TP+FP).
- F1 = 2TP/(2TP+FP+FN) dla klasy występującej w true labels; jeśli model jej nigdy nie przewidzi, F1=0, choć precision jest niezdefiniowane.
- Macro F1 = średnia czterech F1. Brak prawdziwych przykładów klasy oznacza jej recall i F1 jako JSON null, a pełne czteroklasowe macro jako null. Dostępna osobna średnia `over_available_classes` nie udaje pełnego macro.
- AUPRC jest tu **nieinterpolowanym Average Precision**, sumą precision na progach ważoną przyrostem recall. Wyniki o tej samej probability grupowane są na jednym progu. Nie używamy trapezoidalnego PR-AUC. AP liczymy one-vs-rest z Softmax; Macro AUPRC to średnia czterech AP.
- Gdy brak pozytywów albo negatywów dla klasy, celowo zwracamy AP=null (nieinformacyjny przypadek), wraz z listą przyczyn. Specificity jest null przy braku negatywów, precision przy braku pozytywnych predykcji. Nie zastępujemy undefined wartościami NaN lub wymyślonym zerem.
- Czas inferencji walidacyjnej: suma czasu tylko `model(x)` w eval/inference_mode / liczba rekordów; synchronizacja przed/po na CUDA. Nie obejmuje odczytu NPZ, transferu, loss, Softmax ani obliczania metryk. Zależy od batch size, CPU i obciążenia; to pomiar techniczny, nie deklaracja czasu klinicznej aplikacji. Dodatkowy check round-trip nie wchodzi do tego czasu.

Źródła definicji: [CrossEntropyLoss PyTorch 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html), [Average Precision w oficjalnej dokumentacji sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). Implementacja metryk używa NumPy, bez instalowania scikit-learn; testy obejmują znane wyniki, remisy i brak klas. Polityka null przy braku negatywów jest naszym jawnym wyborem, nie twierdzeniem o identycznym zachowaniu każdej biblioteki.

## Wybór checkpointu i early stopping

Najlepszy model = najwyższe validation Macro F1; przy remisie pozostaje wcześniejsza epoka. Nie używamy testu, training loss ani AUPRC do wyboru. Early stopping ma konfigurowalne patience/min_delta i własny licznik poprawy; w baseline patience=5 przy 2 epokach. Jeśli Macro F1 walidacji jest null wskutek braku klasy po batch cap, wykonanie kończy się błędem zamiast wybrać model na nieporównywalnym kryterium.

Każda ukończona epoka zapisuje atomowo własny `checkpoints/epoch_NNN.pt`, model/Adam/RNG, bieżący i najlepszy stan, licznik early stopping, historię, hash manifestu, wybór rekordów i konfigurację/kod. `best_model.pt` jest osobnym końcowym snapshotem najlepszego modelu; jego przeładowanie musi dać identyczne logity na tej samej partii walidacyjnej. To dodatkowa kontrola poprawności serializacji, nie dodatkowy pomiar jakości.

## Bezpieczne uruchomienie i wznowienie

Domyślne wykonanie odmawia użycia istniejącego katalogu wyników. Obecny `results/model_training/phase17_baseline/` jest już wynikiem tej fazy, więc nie uruchamiać polecenia z tą samą ścieżką. Przykład odtworzenia w nowym katalogu:

```bash
cd ~/Pulpit/ECG-APP
MPLCONFIGDIR=/tmp/ecg_phase17_mpl .venv/bin/python -m examples.train_ptbxl_baseline --dry-run
MPLCONFIGDIR=/tmp/ecg_phase17_mpl .venv/bin/python -m examples.train_ptbxl_baseline \
  --output-dir results/model_training/phase17_reproduced
```

Po przerwaniu zachowane są ukończone epoki. Wznowienie odtwarza pełny stan z ostatniego checkpointu **do nowego katalogu**, nigdy nie nadpisując wcześniejszych wyników:

```bash
MPLCONFIGDIR=/tmp/ecg_phase17_mpl .venv/bin/python -m examples.train_ptbxl_baseline \
  --resume-from results/model_training/phase17_baseline/checkpoints/epoch_001.pt \
  --output-dir results/model_training/phase17_resumed
```

Powyższe polecenie jest instrukcją, a nie wykonanym dodatkowym eksperymentem. Epoch 002 checkpoint przy konfiguracji 2 epok jest już ukończony i nie można go przedłużyć pod pozorem resume. Konfiguracja, seed, kod, ścieżka/hash manifestu i selekcja muszą być identyczne. Zmiana celu treningu wymaga nowego eksperymentu. Przerwana częściowa epoka jest odtwarzana od ostatniej ukończonej; jeśli nie ma checkpointu, restart w nowym katalogu zaczyna od seed. Nie zapewniamy wznowienia od środka batcha. `torch.load(..., weights_only=True)` nie dopuszcza dowolnych klas pickle; używać wyłącznie własnych zweryfikowanych checkpointów.

Infrastruktura obsługuje przyszły tryb full z dowolną liczbą epok, per_class=null, cap=null i early stopping. Wymaga osobnej konfiguracji `mode="full"` oraz jawnego `--allow-full-training`; **nie został uruchomiony ani zatwierdzony w fazie 17**. Zwykła domyślna konfiguracja wymusza mały podzbiór i maksymalnie dwie epoki.

## Artefakty

`results/model_training/phase17_baseline/`: training_config.json, selection_manifest.json, architecture.json, training.log, epoch_NNN_metrics.json, epoch_metrics.csv, best_validation_metrics.json, best_validation_predictions.csv, best_model.pt, checkpoints/, result.json, reproducibility.json i trzy wykresy. Metryki zawierają faktyczne ID, true/predicted labels i probabilities, więc można niezależnie odtworzyć macierz/AP. Katalog model_training i *.pt są wykluczone z Git. Nie commitować datasetów ani checkpointów. Aktualne wyniki i testy podaje PHASE17_TRAINING_REPORT.md; dalsze kroki to osobny PHASE18_FULL_TRAINING_PLAN.md.
