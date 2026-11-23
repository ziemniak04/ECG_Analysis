# Faza 22 — plan pełnego treningu MIT-BIH (jeszcze nieuruchomiony)

Ten dokument jest planem po checkpoint A fazy 21. **Pełny trening wymaga osobnej, wyraźnej zgody.** Faza 21 nie zapisuje kandydata wytrenowanego na MIT-BIH i nie otwiera końcowego testu. PTB-XL pozostaje zakończonym, niezależnym zadaniem.

## Zamrożone elementy

- Eksport `data/processed/ecg_dataset_v1/`, bez ponownej filtracji, normalizacji i segmentacji.
- `configs/mitbih_patient_split_v1.json`, SHA256 `8c5cc28d8dcae170fd222432d9f247fd19207ba01be60763241da4b7cae61831`.
- Trening: 29 grup / 62926 uderzeń. Walidacja: 9 grup / 23134 uderzenia. Test: 9 grup / 23392 uderzenia, zamknięty do osobno zatwierdzonej końcowej oceny.
- 201/202 pozostają razem w walidacji. Nie losować nowego podziału uderzeń.
- `MITBIHBeatCNN`, 17797 parametrów, `[B,2,252]` float32 → `[B,5]` logitów; y long, N/S/V/F/Q = 0/1/2/3/4. To grupy uderzeń AAMI-style, nie pięć nazwanych diagnoz rytmu.

## Proponowany pierwszy eksperyment

Dokładna konfiguracja: `PHASE21_MITBIH_TRAINING_CONFIG.json`. CPU, 2 wątki, batch 32, Adam lr 0.001, seed 42, najwyżej 15 epok, patience 4, min_delta 0, bez schedulerów i augmentacji. Wagi `sqrt_balanced`: pierwiastek z `n_train/(5*n_c_train)`, potem podział przez średnią wag. Wagi pochodzą wyłącznie z treningu. Nie zakładamy poprawy F1; to umiarkowany kompromis wobec dominacji N i bardzo dużej pełnej wagi F. Unweighted i balanced pozostają opcjami konfiguracyjnymi, a nie automatycznie uruchamianymi dodatkowymi eksperymentami.

Loader nie zmienia eksportu: ograniczone, leniwe cache 29 shardów treningowych i 10 walidacyjnych, num_workers=0. Cache może stopniowo objąć cały trening i walidację, lecz nie test. Uderzenia treningowe są losowo mieszane bez powtórzeń z generatora `seed+epoch`; walidacja zachowuje pełną deterministyczną kolejność. Zmniejszenie cache może znacznie pogorszyć losowe odczyty skompresowanych NPZ; zwiększenie workerów nie jest w tej wersji dozwolone, aby nie mnożyć pamięci.

Przyszły katalog: `results/model_training/mitbih_phase22_full/run_001/`. Utworzono wyłącznie katalog nadrzędny; run_001 nie został utworzony ani uruchomiony. Jeżeli przed zatwierdzonym startem ta nazwa będzie zajęta, należy wybrać nowy numer, nigdy nadpisywać.

## Uruchomienie dopiero po zgodzie

```bash
MPLCONFIGDIR=/tmp/ecg_mitbih_mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 .venv/bin/python -m examples.train_mitbih   --config PHASE21_MITBIH_TRAINING_CONFIG.json   --output-dir results/model_training/mitbih_phase22_full/run_001   --approve-full-training
```

To polecenie **nie zostało wykonane**. Flaga jest zabezpieczeniem interfejsu; nie zastępuje zgody użytkowniczki.

## Co zapisywać i jak wybierać model

Każda epoka obejmie wszystkie 62926 treningowych i wszystkie 23134 walidacyjnych uderzenia — bez batch cap. Walidacja w `eval()` i `inference_mode()`. Najlepszy checkpoint: najwyższy Macro F1 na kompletnej walidacji; przy remisie zachować wcześniejszą epokę. Early stopping po czterech epokach bez poprawy. Nie używać końcowego testu do wyboru strategii wag, liczby epok, progów ani architektury.

Trainer zapisuje plan z sumami źródeł, manifestu, protokołu i wersjami runtime; konfigurację, log postępu, pełne metryki epok, per-group metryki, kompletne stany modelu/Adam/RNG i najlepszy model. Każdy checkpoint ma plik SHA256. Pliki epok są nieprzepisywalne; zapis wykorzystuje plik tymczasowy i atomowe utworzenie docelowego pliku. Brak kompletnej pary checkpoint + poprawna suma kontrolna wyklucza wznowienie z tej epoki; użyć wcześniejszej poprawnej pary.

Nie przechowujemy surowych predykcji wszystkich uderzeń w historii checkpointu, tylko pełne agregaty i metryki grup; zapobiega to niepotrzebnemu rozrostowi. `best_model.pt` powstaje po zakończeniu lub obsłużonym przerwaniu; przy twardym zabiciu procesu najlepsze wagi pozostają w ostatnim kompletnym checkpoincie epoki.

## Bezpieczne wznowienie

```bash
MPLCONFIGDIR=/tmp/ecg_mitbih_mpl .venv/bin/python -m examples.train_mitbih   --config PHASE21_MITBIH_TRAINING_CONFIG.json   --output-dir results/model_training/mitbih_phase22_full/run_002   --resume-from results/model_training/mitbih_phase22_full/run_001/epoch_003.pt   --approve-full-training
```

`epoch_003.pt` to przykład, nie istniejący wynik. Należy wskazać faktycznie ostatni kompletny, zweryfikowany checkpoint. Kod, runtime, konfiguracja, manifest i podział muszą być identyczne. Wznowienie następuje od kolejnej epoki z odtworzonym Adam/RNG; aktualizacje z przerwanej części epoki są odrzucane. W nowym procesie cache ponownie się napełni. Nie wznawiać runu zakończonego early stopping lub pełnym limitem epok. Test na syntetycznych danych potwierdza identyczne końcowe tensory dla przebiegu ciągłego i wznowionego. Brak checkpointu przed końcem pierwszej epoki oznacza brak bezpiecznego punktu wznowienia; nie udawać odzyskania tych aktualizacji.

## Metryki i ograniczenia

Accuracy, pięcioklasowy Macro F1, one-vs-rest Macro AP (opisany jako Macro AUPRC), precision/recall/specificity/F1, macierz pomyłek i czas inferencji CPU. Czas inferencji obejmuje sam forward; czas epoki i loading są osobno. Metryki treningowe są online, przed poszczególnymi aktualizacjami, więc nie są oceną jednego zamrożonego modelu. Strata ważona jest agregowana przez dokładny licznik i sumę wag, nie przez średnią ze średnich batchy. Walidacyjna strata przy ważonym CE używa tych samych wag treningowych; nie porównywać jej bezpośrednio ze stratą innego wariantu wag.

Monitorować F/Q: support uderzeń i grup, recall, precision, FP/FN i macierze per grupa. Dla grup bez danej klasy metryki nieokreślone są null, nie zero. Walidacja ma tylko 38 F, a Q jest zdominowana przez jedno nagranie; nie dobierać podziału ponownie dla lepszych wyników. Accuracy sama jest niewystarczająca przy 83% N w treningu. Nie traktować sąsiadujących uderzeń tej samej osoby jako niezależnych pacjentów; nie przedstawiać beat-level przedziałów ufności jako dowodu generalizacji między pacjentami.

Po pełnym treningu przeanalizować wyłącznie walidację i zgłosić wyniki oraz ograniczenia. Ostateczna ocena zamrożonego modelu na teście wymaga osobnej fazy i zgody. Nie rozpoczynać jej automatycznie.
