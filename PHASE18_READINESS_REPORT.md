# Faza 18 — checkpoint A: konfiguracja, benchmark CPU i gotowość

**Wynik: infrastruktura gotowa technicznie do pełnego treningu CPU, oczekuje na odrębną wyraźną zgodę użytkowniczki. Nie rozpoczęto pełnej epoki ani treningu wieloepokowego.** Utworzono nową konfigurację, zmierzono ograniczone próby wydajności i sprawdzono zasoby/protokół/resume. Nie oceniano jakości modelu na benchmarku i nie używano waveformów foldu 10.

## Zamrożony protokół danych

Istniejący eksport `data/processed/ptbxl_rhythm_v1/`, ukończony manifest `runs/90b449bf54b649509519747f417b9e5e/PTBXL_DOWNLOAD_MANIFEST.json`. Klasy: SR=0, AFIB=1, STACH=2, SBRAD=3. Nie zmieniono filtrów, normalizacji, architektury ani istniejących Dataset/loader/trainer.

| Partycja | Oficjalne foldy | Rekordy | Pacjenci |
|---|---|---:|---:|
| Train | 1–8 | 15689 | 13749 |
| Validation | 9 | 1958 | 1763 |
| Test — wyłącznie metadane protokołu | 10 | 1958 | 1711 |

Indeks fazy 16 potwierdza jedną wartość fold na pacjenta, zgodność z oryginalnymi CSV i brak nakładania grup. Dla przyszłego treningu wybrano **całe train i całe validation**, bez quotas, batch caps, shuffle walidacji ani drop_last. Przy batch 16: 981 partii treningowych i 123 walidacyjnych; ostatnie partie będą niepełne, a wszystkie 1958 ECG walidacji trafią do epoch-level Macro F1. Plan ID/indeksów/hashy: readiness_plan.json.

Waveformy testowe nie są ładowane przez benchmark ani nowy pipeline fazy 18. Inspekcja metadanych testu służy wyłącznie weryfikacji separacji pacjentów. Brak final test metrics i brak strojenia na foldzie 10.

## Zmierzony benchmark CPU

CPU: **Intel(R) Core(TM) i5-8350U CPU @ 1.70GHz**, 8 logicznych CPU widocznych, affinity=8. PyTorch 2.15.0+cpu. Brak nowej instalacji zależności; pomiar CPU-only. Nie ustalono limitu cgroup CPU — pole w audycie jest null, nie twierdzimy, że zasoby są dedykowane.

Każda z czterech prób: nowa inicjalizacja istniejącego CNN (74116 parametrów), seed 42, Adam lr=.001, ważony CrossEntropyLoss z wagami całego train. Jedna partia rozgrzewki + sześć mierzonych partii treningowych; analogicznie osobny pomiar walidacji. Modele są tymczasowe i odrzucone, bez checkpointu wybranego z benchmarku. Łącznie **28 ograniczonych aktualizacji benchmarkowych** (w tym rozgrzewki), zero pełnych epok. Nie uruchamiano train_model ani full-training CLI. Dla batch16 mierzono po 96 ECG w każdej fazie/próbie; dla batch32 po 192; dodatkowe 16/32 były rozgrzewką. Rekordy mogą się powtarzać między konfiguracjami — to porównanie kosztu, nie cztery eksperymenty jakościowe.

Przepustowość obejmuje rzeczywisty DataLoader, odczyt NPZ, SHA256/metadane, transfer CPU, model, loss i (w treningu) backward/Adam oraz przygotowanie prawdopodobieństw do metryk. Końcową agregację metryk i setup mierzono oddzielnie i ekstrapolowano. W próbach użyto istniejącego epoch_pass; batch-loss w logach służy kontroli numerycznej. Nie użyto accuracy/F1/AP prób do doboru konfiguracji.

| Batch | Wątki | Train ECG/s | Validation ECG/s | Szacowana pełna epoka (min) | Peak RSS próby (MiB) |
|---|---:|---:|---:|---:|---:|
| 16 | 1 | 42.30 | 70.78 | 6.68 | 561.9 |
| 16 | 2 | 54.17 | 84.62 | 5.25 | 565.8 |
| 32 | 1 | 39.95 | 68.93 | 7.05 | 601.1 |
| 32 | 2 | 47.82 | 85.46 | 5.88 | 601.6 |

**Rekomendacja: batch=16, wątki=2.** Początkowo rozważano 32; zmierzony wariant 16/2 miał najkrótszą prognozę epoki. Batch32/2 był około 12% wolniejszy w prognozie całego cyklu, choć jego sama walidacja była podobnie szybka. Wybór oparto wyłącznie na czasie/zasobach. Z góry ustalona polityka: wśród wyników do 10% od najszybszego preferować batch32, potem mniej wątków; w tym pomiarze rekomendację nadal wyznacza 16/2. Sześć partii nie stanowi dużego niezależnego benchmarku statystycznego.

## Szacunek czasu — nie pomiar pełnej epoki

Dla rekomendacji:

- Train: około **289.8 s**.
- Validation: około **23.3 s**.
- Oszacowana agregacja metryk/setup: **1.7 s**.
- Razem: **314.7 s ≈ 5.25 min/epokę**.
- Zakres planistyczny: **5.04–6.75 min/epokę**.
- 15 epok bez early stopping: nominalnie **78.7 min**; górny wariant planistyczny **101.2 min**. Patience może skrócić rzeczywistą liczbę epok.

Metoda: ceil(N/batch) × średni zmierzony czas pełnej partii, z osobnym przybliżeniem wzrostu kosztu metryk i setup. Końcową niepełną partię potraktowano jak pełną. Zakres korzysta z min/max sześciu czasów i dodatkowego 25% górnego marginesu; **nie jest przedziałem ufności ani gwarancją czasu**. Struktury CPU/MKLDNN i cache dyskowy są rozgrzane, kolejne próby nie są niezależne. Długi trening może być wolniejszy przez throttling, procesy w tle, cold I/O lub inne limity. Początkowe wczytanie całego indeksu i zapis checkpointów/raportów nie są wprost ujęte w podanym czasie; koszt pełnej agregacji metryk jest tylko przybliżony. Nie wykonano pełnej epoki dla potwierdzenia estymacji, zgodnie z zakresem checkpoint A.

## Pamięć i dysk

Największy zaobserwowany RSS procesu w próbach: **601.6 MiB**; potwierdzony również przez Linux VmHWM. RSS obejmuje PyTorch, indeks metadanych i bieżące modele/aktywacje, nie samą tablicę X. Dane pozostają ładowane leniwie. Dostępny RAM w /proc po pomiarze: **5.43 GiB** (widoczny systemowo, nie gwarantowana kwota kontenera).

Trainer zachowuje probabilities/labels/history, a każdy kolejny checkpoint powtarza całą dotychczasową historię. Dlatego projekcja pamięci i dysku uwzględnia wzrost, zamiast mnożyć sam rozmiar wag przez 15. Szacowany Python history dla pełnych 15 epok: **77.9 MiB**. Konserwatywny budżet procesu = max zaobserwowany RSS + ta estymacja: **0.66 GiB**. Widoczny dostępny RAM przewyższa dwukrotność tego budżetu. To oszacowanie, nie zmierzony peak pełnego treningu; serializer, backend i system mogą potrzebować dodatkowej pamięci.

Wolny dysk po pomiarze: **375.22 GB / 349.45 GiB**. Dodatkowe dane ECG nie są pobierane.

| Pozycja | Rozmiar / czas | Charakter |
|---|---:|---|
| Istniejący checkpoint epoki 1 | 1244391 bajtów | zmierzony |
| Istniejący checkpoint epoki 2 | 1256423 bajtów | zmierzony |
| Zapis kopii checkpointu I/O | 0.0343 s | jeden mały pomiar z fsync/atomic install |
| Odczyt kopii I/O | 0.0802 s | jeden mały pomiar |
| Największy checkpoint 15. epoki full | 18.14 MB | szacunek |
| Wszystkie 15 checkpointów epok | 156.00 MB | szacunek narastającej historii |
| JSON metryk wszystkich epok | 52.36 MB | szacunek |
| Run z buforem tymczasowego checkpointu | 227.13 MB | szacunek, nie zmierzony pełny eksport |
| Zalecana rezerwa | **1 GiB** | min. 1 GiB lub 2× estymacja |

Rozmiary szacowano z realnego przyrostu między checkpointami fazy 17, proporcji liczby rekordów i rzeczywistego pełnego planu. Zsumowano trójkątny wzrost historii we wszystkich checkpointach i osobne JSON. Nie generowano fikcyjnych przebiegów ani pełnych wyników treningu do szacowania rozmiarów. Jest wystarczająco dużo dysku względem rezerwy. Probe `phase17_checkpoint_io_probe.pt` to wyłącznie kopia istniejącego checkpointu fazy 17 do pomiaru I/O, **nie model fazy 18 do wdrożenia**; jego przeładowanie jest identyczne, oryginał niezmieniony.

## Niezbalansowanie i pierwsza konfiguracja

| Klasa | Training ECG (foldy 1–8) | Waga n_train/(4*n_class) |
|---|---:|---:|
| SR | 13354 | 0.29371348 |
| AFIB | 1189 | 3.29878044 |
| STACH | 643 | 6.09992218 |
| SBRAD | 503 | 7.79771376 |

Dominacja SR wynosi około 85%, a stosunek SR/SBRAD około 26,55. **Rekomenduję włączyć ważony CrossEntropyLoss w pierwszym pełnym runie**, zgodnie z celem Macro F1 i nazwanych rzadkich rytmów. Wagi obliczono wyłącznie z całych foldów 1–8; nie z benchmarkowego podzbioru i nigdy z walidacji/testu. Zwiększają wkład rzadkich klas w loss, ale mogą zwiększać wrażliwość na szum i nie gwarantują poprawy recall lub precision. W walidacyjnym loss stosowane są te same stałe wagi treningowe; metryki klasyfikacji i Macro F1 pozostają nieważone.

Brak SR/AFIB w predykcjach małego, równo licznego baseline fazy 17 nie dowodzi, że same wagi rozwiążą problem. Nie wybierano opcji na podstawie nowych wyników jakościowych ani foldu 10. Learning rate **0.001** pozostaje uzasadnioną konfiguracją początkową Adam, bez strojenia na benchmarku. Architektura, preprocessing i seed pozostają takie same.

Nowy plik **configs/ptbxl_phase18_full_v1.json**: batch 16, 2 wątki CPU, Adam lr=.001, class weighting=true, seed42, maks.15 epok, patience4, min_delta0.0, workers0, brak limitów i podzbiorów train/val. Najlepszy checkpoint = maksimum **Macro F1 całego foldu 9** (przy remisie wcześniejsza epoka), nie test ani training loss. Start od świeżej inicjalizacji; nie od benchmarku i nie od małego modelu fazy 17. Wersja v1 została zapisana tylko raz i nie nadpisuje konfiguracji fazy 17.

## Checkpointy, przerwanie i wznowienie

Użyto istniejącego trenera bez zmian. Zapis jest atomowy i exclusive-create; gotowy katalog runu nie jest nadpisywany. Każda ukończona epoka ma model, Adam, RNG, stan early stopping, historię, najlepszy stan i hash config/kodu/manifestu. Resume wymaga identycznego planu i nowego katalogu wyjściowego. Częściowa przerwana epoka jest odtwarzana od ostatniej kompletnej; pierwszą bez checkpointu należy zacząć ponownie. Nie obsługujemy środka batcha.

Testy obejmują zachowanie ostatniej ukończonej epoki, poprawne przeładowanie i zgodność bitową resumed vs uninterrupted na kontrolowanych fixture'ach. Dodatkowy test weryfikuje planowane **2 wątki i włączoną ścieżkę wag**. To małe syntetyczne testy techniczne, nie pełne epoki rzeczywistych danych. Nie zmieniano starego source code, więc checkpointy fazy 17 pozostają odtwarzalne przy ich oryginalnym planie; nie można ich wznawiać jako innego, pełnego planu.

## Testy i zachowanie pracy

**69 istotnych przypadków testowych zaliczonych**: sesja 68 passed (2 ostrzeżenia importowe WFDB/pandas) oraz dodatkowy test wznowienia 1 passed (te same 2 ostrzeżenia). Nie przedstawiamy tego jako ponownego uruchomienia całego historycznego zestawu 532 testów — sprawdzono wszystkie istotne dla tego checkpointu moduły: readiness, metrics, training, protocol i Torch/model. Logi: relevant_verified_pytest.log oraz weighted_two_thread_resume_pytest.log. Wcześniejszy brak importu helpera w nowym teście I/O poprawiono; końcowe sesje są bez błędów i pominięć.

Pokrycie: pełna konfiguracja i brak caps walidacji, patient-safe splits, wagi train-only, metryki accuracy/F1/recall/specificity/precision/AP/CM z undefined przypadkami/remisami, inference timing, forward/backward, walidacja eval/BN, checkpoint/early stopping, deterministic resume, cap bez nadmiarowego pobrania, brak test reads w próbach i odmowa nadpisania wyników. Nie wyciszano warnings.

SHA256 poprzednich źródeł, raportów, konfiguracji, manifestów i wszystkich plików baseline fazy 17 porównano z nowym checkpointem zachowania: **bez zmian**. Istniejące niezatwierdzone pliki na feature/ptbxl-1dcnn zachowano. Nie wykonano commit/push/reset. Nowe duże artefakty i probe checkpoint są objęte istniejącym .gitignore.

## Nowe pliki i decyzja checkpoint A

- src/ptbxl_readiness.py i examples/benchmark_ptbxl_cpu.py — wyłącznie ograniczony benchmark i plan, bez wywołania pełnego trenera.
- configs/ptbxl_phase18_full_v1.json — konkretna konfiguracja do zatwierdzenia.
- tests/test_ptbxl_readiness.py — nowe kontrole; pozostałe testy bez zmian.
- PHASE18_READINESS_REPORT.md — niniejszy raport.
- results/model_training/phase18_checkpoint_a/: benchmark.json, readiness_plan.json, logi/probe, preservation i końcowy audyt.

**Gotowość techniczna: TAK, do oddzielnie zatwierdzonego pełnego treningu.** Nie jest to potwierdzenie docelowej skuteczności modelu. Proponowany nowy katalog `results/model_training/phase18_full_v1/` jeszcze nie istnieje. Polecenie poniżej jest wyłącznie przygotowane do przeglądu; **nie wykonano go i wymaga wyraźnej zgody użytkowniczki**:

```bash
MPLCONFIGDIR=/tmp/ecg_phase18_mpl .venv/bin/python -u -m examples.train_ptbxl_baseline --config configs/ptbxl_phase18_full_v1.json --output-dir results/model_training/phase18_full_v1 --allow-full-training
```

Domyślny CLI bez jawnego trybu pełnego nie rozpocznie takiego runu. Zatrzymujemy się na checkpoint A; nie używano final test, nie zmieniano datasetów/preprocessingu, nie wdrażano MIT-BIH CNN ani Streamlit.
