# Faza 21, checkpoint A — gotowość treningu MIT-BIH

**Status: infrastruktura gotowa technicznie; pełny trening nie został rozpoczęty i wymaga osobnej zgody.** Benchmark był ograniczony, nie wytrenowano ani nie wybrano nowego modelu. Nie wykonano oceny końcowego testu MIT-BIH ani ponownej oceny PTB-XL.

## Zachowane źródła i zamrożony protokół

Przeczytano raport fazy 20, dokumentację interfejsu i architektury oraz istniejący trainer PTB-XL. Nowy `src/mitbih_training.py` korzysta z istniejących `MITBIHBeatDataset`, `MITBIHBeatCNN`, protokołu i funkcji wag. Z PTB-XL wykorzystuje wyłącznie niezmienione pomocnicze funkcje seed/RNG/atomowego zapisu oraz sprawdzoną funkcję binary average precision. Nie modyfikuje modelu ani trainera PTB-XL.

Eksport: `data/processed/ecg_dataset_v1/`, 109452 uderzenia, 48 nagrań, 47 technicznych grup. Protokół `configs/mitbih_patient_split_v1.json`, SHA256 `8c5cc28d8dcae170fd222432d9f247fd19207ba01be60763241da4b7cae61831`. Potwierdzono rozłączność grup, pokrycie całego eksportu i poniższe dokładne liczby z manifestu. Rekordy **201 i 202 mają jedną grupę i razem należą do walidacji**.

| Część | Grupy | Nagrania | Uderzenia | N | S | V | F | Q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Trening | 29 | 29 | 62926 | 52254 | 1715 | 4389 | 401 | 4167 |
| Walidacja | 9 | 10 | 23134 | 19094 | 491 | 1444 | 38 | 2067 |
| Końcowy test — zamknięty | 9 | 9 | 23392 | 19246 | 575 | 1402 | 363 | 1806 |

Testowe liczby pochodzą wyłącznie z istniejącego manifestu. Nowy kod fazy 21 nie otwiera testowych NPZ ani nie używa ich etykiet do wag, benchmarku, wyboru konfiguracji czy oceny modelu. Pełny zestaw historycznych testów obejmuje preprocessing/QC i kontrolowane fixture; nie uruchamia końcowej oceny nowego modelu ani ponownej rzeczywistej ewaluacji PTB-XL fold 10.

## Nierównowaga — wyłącznie dane treningowe

| Klasa | Uderzenia train | Udział train | A: unweighted | B: balanced | C: sqrt balanced, średnia 1 |
|---|---:|---:|---:|---:|---:|
| N | 52254 | 83.040% | 1 | 0.240847 | 0.200587 |
| S | 1715 | 2.725% | 1 | 7.338309 | 1.107213 |
| V | 4389 | 6.975% | 1 | 2.867441 | 0.692118 |
| F | 401 | 0.637% | 1 | 31.384539 | 2.289766 |
| Q | 4167 | 6.622% | 1 | 3.020206 | 0.710316 |

A. **Unweighted CE:** najprostszy punkt odniesienia, zachowuje rozkład uderzeń. Przy 83.04% N może premiować ignorowanie S/F; wysoka Accuracy nie dowodzi dobrej detekcji mniejszości. Jednocześnie nie wymusza bardzo dużych aktualizacji od pojedynczych rzadkich przykładów.

B. **Balanced CE:** `n_train/(5*n_c_train)`. Zwiększa znaczenie S i F, lecz F/N ma względną wagę około 130.3. Przy koncentracji F u jednego pacjenta może nasilić przeuczenie, wrażliwość na błędy etykiet i fałszywe alarmy kosztem N. Nie zwiększa liczby niezależnych pacjentów.

C. **Częściowe ważenie:** `sqrt(balanced) / mean(sqrt(balanced))`. Względne F/N około 11.4; zachowuje umiarkowaną korektę zamiast pełnego wyrównania wpływu klas. Wspólne skalowanie do średniej 1 nie zmienia średniej ważonej CE, której mianownikiem jest suma wag w batchu. Nie jest to kalibracja prawdopodobieństw.

**Proponowany pierwszy run: C, `sqrt_balanced`.** Jest to uzasadniony kompromis do sprawdzenia, nie zmierzona poprawa Macro F1. Wynik PTB-XL z unweighted nie przesądza o innym zadaniu MIT-BIH. A i B są opcjami, a nie zaplanowanymi automatycznie dodatkowymi treningami. Wszystkie wagi wyznacza ta sama funkcja korzystająca wyłącznie z 29 treningowych grup; nie ma globalnego oversamplingu ani wag z walidacji/testu.

### Koncentracja F i Q

W treningu F: 401 przykładów z 10 grup, ale 372 (92.77%) pochodzą z 208. Walidacja F: tylko 38 z czterech grup. Q w treningu: 4167 z pięciu grup, w tym 4161 z 102/107. Walidacja Q: 2067 z dwóch grup, w tym 2062 z 104. Dane testowe pozostają zamknięte; znane ograniczenia ich koncentracji opisano już w fazie 20 na podstawie metadanych.

Monitorować per-class precision, recall, F1, FP/FN i macierze oraz metryki według grup. Trainer zapisuje metryki wszystkich grup, także brak klas i null dla nieokreślonych wartości. Pojedyncze błędy F silnie zmienią walidacyjny recall. Duża liczba uderzeń Q nie oznacza dużej różnorodności pacjentów. Wagi nie rozwiązują zależności uderzeń wewnątrz nagrania ani różnic odprowadzeń. Nie zmieniamy protokołu po obejrzeniu wyników.

## Pomiar CPU — rzeczywiste przetworzone uderzenia

PyTorch CPU, 2 wątki, seed 42, num_workers=0. Każdy batch size: osobny świeży model o 17797 parametrach, 2 batchy rozgrzewki i 6 mierzonych treningowych, potem 2+6 walidacyjnych w eval/inference_mode. Łącznie **24 aktualizacje na trzech tymczasowych modelach**, czyli 896 wystąpień treningowych przykładów (z możliwymi powtórzeniami między wariantami). Nie są to epoki; modele porzucono, nie zapisano ich checkpointów ani metryk skuteczności. Na walidacji mierzyliśmy czas, nie wybieraliśmy architektury ani strategii wag według wyników.

| Batch | Train beats/s | Validation beats/s | Szac. train s | Szac. validation s | Suma s/epokę, ciepły cache |
|---:|---:|---:|---:|---:|---:|
| 16 | 566.8 | 1842.3 | 111.02 | 12.56 | 123.57 |
| 32 | 854.4 | 2112.5 | 73.67 | 10.95 | 84.63 |
| 64 | 817.3 | 2594.4 | 77.06 | 8.93 | 85.99 |

Najkrótszy zmierzony szacunek ma **batch 32**: trening około 73.67 s, walidacja 10.95 s, łącznie **84.63 s** (około 1.41 min). Różnica względem 64 wynosi tylko około 1.6% i nie stanowi statystycznego dowodu przewagi; wybieramy 32 jako rozsądny początek na tym CPU, nie jako optymalny batch dla jakości modelu.

Pierwsze napełnienie cache: trening 2.955 s / 29 shardów; walidacja 1.056 s / 10 shardów, razem 4.011 s. To zimny **cache loadera**, nie pomiar dysku po opróżnieniu systemowego page cache — OS cache nie był czyszczony. Pełne X odczytano wyłącznie dla 29 treningowych i 10 walidacyjnych nagrań; istniejący loader sprawdził ich sumy SHA256, typy, kształty, etykiety i skończoność. Lista jest w benchmark.json.

Dla batch 32 loading/collate przy ciepłym cache: 0.03986 s dla 6 batchy treningowych (około 17.7% czasu) i 0.03888 s dla 6 walidacyjnych (około 42.8%). Walidacja ma znacznie krótsze obliczenia. Cache mieści wszystkie shardy tych dwóch części, dlatego losowy shuffle nie wymusza kolejnych dekompresji przy każdym batchu. Dane nadal ładowane są leniwie; nie wczytujemy testu ani nie przebudowujemy eksportu.

Szacunek pierwszej epoki dla 32: **88.64 s** plus pełne metryki/logowanie/checkpointy. To ekstrapolacja z 6 batchy, nie pomiar pełnej epoki. Nie obejmuje kosztu agregacji AP na całej populacji, zapisów ani zmian obciążenia laptopa. Praktycznie planować około **1–3 min/epokę**, maksymalnie około **15–45 min dla 15 epok**, z możliwością wcześniejszego zatrzymania. To orientacyjny zapas, nie przedział ufności ani gwarantowany czas. Należy porównać z rzeczywistą pierwszą epoką dopiero po zgodzie; nie wykonywać teraz całej epoki tylko dla dokładniejszego oszacowania.

## RAM, dysk i checkpointy

- Cache train: 127.37 MiB; validation: 46.88 MiB; razem około 174.25 MiB tablic NumPy.
- Szczyt RSS procesu w pomiarze: 639.21 MiB; obejmuje biblioteki, cache i aktywacje. To pamięć procesu, nie całego systemu. Pełna historia/metryki nie były tu przechowywane; przed startem warto mieć co najmniej około 2 GiB wolnej pamięci jako zapas, nie traktować 639 MiB jako limitu.
- MemAvailable w końcu pomiaru: 4.38 GiB; wolny dysk: 349.10 GiB. Są to migawki z /proc i filesystemu, nie gwarancja przyszłej dostępności.
- Zmierzony serializowany zestaw tensorów model + najlepsze wagi + Adam + stan torch RNG: **316327 B (0.302 MiB)**. Nie zapisano go na dysk jako model.
- Rzeczywisty checkpoint będzie większy o plan, Python/NumPy RNG i narastającą historię grup/epok. Dla 15 epok orientacyjnie rezerwować **64 MiB** na checkpointy i metryki; to budżet projektowy z zapasem, nie zmierzony rozmiar przyszłego eksportu. Pełna historia jest powtarzana w kolejnych checkpointach. Brak problemu pojemności na obecnym dysku.

## Gotowa konfiguracja i zachowanie trainera

`PHASE21_MITBIH_TRAINING_CONFIG.json`: CPU, **batch 32, Adam lr 0.001, seed 42, 2 wątki, max 15 epok, patience 4, min_delta 0, sqrt_balanced**, cache 29/10, zero workerów. Bez Softmax przed CE. Bez schedulerów, oversamplingu i augmentacji. Architektura i preprocessing fazy 20 pozostają bez zmian.

Pełny trainer sprawdza jawne zatwierdzenie zanim wykona I/O danych, odrzuca istniejący katalog wynikowy oraz nie udostępnia loadera końcowego testu. Każda przyszła epoka pokryje wszystkie 62926 uderzeń treningowych i **komplet 23134 walidacyjnych**; sprawdzane są dokładne globalne indeksy i brak duplikatów. Model wybierany przez najwyższy walidacyjny Macro F1; remis zostawia wcześniejszą epokę. Nie ma cap na walidację w pełnym trainerze.

Plan wiąże config, model, klasy, manifest, protokół, sumy kodu i wersje torch/numpy. Wiążący plan po testach to `readiness_training_plan.json`. Jego `split_sha256` jest hashem kanonicznego JSON protokołu; nie należy go mylić z podanym wyżej SHA256 bajtów pliku zamrożonego protokołu. Wewnętrzny plan katalogu cpu_benchmark opisuje stan kodu w chwili pomiaru, przed końcowym dopracowaniem kontroli checkpointów/logowania; nie jest konfiguracją zatwierdzonego treningu.

Każdy plik epoki tworzony jest atomowo bez nadpisania i ma SHA256. Checkpoint obejmuje model, Adam, RNG Python/NumPy/torch, historię, najlepsze wagi i stan early stopping. Wznowienie wymaga identycznego planu i nowego katalogu; odbywa się na granicy pełnych epok. Częściowa epoka jest odrzucana i odtwarzana. Uszkodzona suma, nieskończone wagi/stany Adam, zmieniony config lub zakończony run powodują błąd. Przy twardym przerwaniu bez kompletnego pliku sumy kontrolnej należy użyć wcześniejszego poprawnego checkpointu, nie zgadywać.

Przyszła lokalizacja: `results/model_training/mitbih_phase22_full/run_001/`. Utworzono wyłącznie katalog nadrzędny; **run_001 nie istnieje, nie zawiera modeli i nie został uruchomiony**. Polecenia przyszłego startu/wznowienia zawiera `PHASE22_MITBIH_FULL_TRAINING_PLAN.md`, wyraźnie jako czynności po osobnej zgodzie.

## Definicje i testy metryk

Confusion matrix: wiersze rzeczywiste, kolumny argmax, N/S/V/F/Q. TP/FP/FN/TN są one-vs-rest. Accuracy = poprawne/wszystkie; precision = TP/(TP+FP), recall = TP/(TP+FN), specificity = TN/(TN+FP), F1 = 2TP/(2TP+FP+FN). Macro F1 to średnia pięciu F1. Macro AUPRC to średnia pięciu **nieinterpolowanych average precision**, ze wspólnymi progami dla remisów — nie pole trapezami. Softmax służy jedynie do AP, nie do wejścia CE i nie jest deklaracją klinicznej kalibracji.

Przy brakujących klasach per-group lub w fixture: nieokreślone wartości to JSON null; pełny macro jest null, jeśli któryś składnik nie jest określony. F1 klasy obecnej, lecz całkowicie pominiętej, wynosi zero. Inference time mierzy sam CPU forward, bez loading/loss/Softmax/metryk; osobno zapisywane są loading i czas całego przejścia. Metryki treningowe są online z dropout/BN, przed aktualizacjami, a walidacyjne po epoce z eval/inference_mode. Strata ważona jest sumowana jako dokładny licznik/suma wag, także dla ostatniego niepełnego batcha.

Nowe testy: **28**, obejmują config, ręcznie sprawdzoną macierz i metryki, AP/remisy, klasy nieobecne, błędne dane, train-only wagi, deterministyczny shuffle i pełne pokrycie, brak testu/patient leakage, ważoną agregację straty, gate przed I/O, zapis/odczyt i sumy checkpointów, identyczne wznowienie, early stopping i przerwanie. Trening epok w testach odbywa się tylko na małych syntetycznych shardach; nie jest eksperymentem na rzeczywistym pełnym zbiorze.

Pełne `.venv/bin/python -m pytest -ra`: **648 passed, 229 warnings, 198.67 s**. Ostrzeżenia pandas/WFDB dotyczą deprecacji inplace w set_index/drop; pozostawiono je widoczne, bez zmian zależności i bez wyciszania. Log: `results/model_setup/phase21_mitbih/run_001/full_pytest.log`.

## Integralność i utworzone pliki

Przed pracą zapisano SHA256 istniejących źródeł/raportów/konfiguracji oraz wszystkich wyników model_training/model_evaluation i fazy 20; zapisano także inwentarz, rozmiary i mtime obu eksportów. Końcowa kontrola `final_integrity_check.json` musi potwierdzać brak zmian, identyczny gotowy plan i brak pełnego runu. Nie zmieniano zastanych zmian Git, istniejącego .gitignore ani nie tworzono commitów. Istniejące reguły ignorują duże dane i checkpointy.

Nowe pliki:

- `src/mitbih_training.py`, `src/mitbih_metrics.py`, `src/mitbih_readiness.py`.
- `examples/benchmark_mitbih_cpu.py`, `examples/train_mitbih.py` (drugi nieuruchomiony na danych rzeczywistych).
- `tests/test_mitbih_training.py`.
- `PHASE21_MITBIH_TRAINING_CONFIG.json`, niniejszy raport, `PHASE22_MITBIH_FULL_TRAINING_PLAN.md`.
- Osobne artefakty `results/model_setup/phase21_mitbih/run_001/`: benchmark, zasoby, plan, inwentarz grup/klas, CSV wag, logi testów, kontrola zachowania wcześniejszej pracy i sumy kontrolne.

**Wniosek:** pełny trening jest gotowy technicznie do osobnego zatwierdzenia przy powyższych ograniczeniach. W fazie 21 checkpoint A zatrzymujemy się na konfiguracji, pomiarach i testach. Nie rozpoczynamy fazy 22, testu końcowego MIT-BIH ani żadnych zmian PTB-XL/Streamlit.
