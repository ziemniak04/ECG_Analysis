# Faza 22 — pierwszy pełny trening CNN MIT-BIH

Status: **early_stopped**, 5 ukończonych epok. Wykonano wyłącznie jeden zatwierdzony eksperyment. Nie oceniano końcowego testu MIT-BIH i nie zmieniono zakończonego modelu ani wyników PTB-XL. Wyniki poniżej są **walidacyjne**, nie testowe ani kliniczne.

## Eksperyment i dane

Katalog: `results/model_training/phase22_mitbih_full/run_001`. Dokładna konfiguracja pochodzi z niezmienionego `PHASE21_MITBIH_TRAINING_CONFIG.json`, kopia w `configuration.json`. Adam, lr 0.001, batch 32, CPU 2 wątki, seed 42, max 15 epok, patience 4, min_delta 0. Architektura fazy 20: 17797 parametrów, wejście float32 `[B,2,252]`, wyjście `[B,5]` logitów; CE otrzymuje logity bez Softmax.

Klasy N=0, S=1, V=2, F=3, Q=4 są grupami uderzeń AAMI-style, nie pięcioma nazwanymi diagnozami rytmu. Użyto istniejących segmentów bez ponownej filtracji/normalizacji i bez zmian etykiet/odprowadzeń.

Trening: **62926 uderzeń / 29 grup** (N52254, S1715, V4389, F401, Q4167). Walidacja: **23134 / 9 grup** (N19094, S491, V1444, F38, Q2067). Zamrożony test: 23392 / 9 grup — wyłącznie liczby z manifestu; jego plików NPZ nie otwierano w treningu ani w analizie. 201 i 202 mają wspólną grupę, razem w walidacji. Grupy wszystkich trzech części są rozłączne.

Zastosowano pierwiastki treningowych wag balanced, znormalizowane do średniej 1: N 0.200587, S 1.107213, V 0.692118, F 2.289766, Q 0.710316. Sprawdzono je niezależnie ze statystyk wyłącznie treningu. Nie wykonywano oversamplingu/undersamplingu, augmentacji ani wyboru wag na teście.

## Czas i wybór checkpointu

- Ukończone epoki: **5**; status early stopping: **tak**.
- Całkowity czas `train_model`: **499.89 s (8.33 min)**, łącznie z loading/metrykami/zapisem; bez kontroli preflight.
- Suma zmierzonych przebiegów epok: 495.23 s, średnio 99.05 s/epokę. Czasy epok obejmują przejścia i metryki, a zapis checkpointu następuje po tym pomiarze.
- Najlepsza epoka: **1**, wybrana przez najwyższy pięcioklasowy Macro F1 na **kompletnych 23134 przykładach walidacyjnych**. Przy remisie wcześniejsza epoka.
- Każda epoka: pełne 62926 treningowych i 23134 walidacyjne, brak duplikatów/brakujących indeksów; 1967/723 batchy.

## Wyniki najlepszego checkpointu — wyłącznie walidacja

- **Macro F1: 0.354215**.
- **Macro AUPRC / average precision: 0.416958**.
- **Accuracy: 84.15%**. Dla porównania, stałe przewidywanie N dałoby 82.54% na tym samym rozkładzie — to opisowy punkt odniesienia, nie dodatkowy trenowany model.
- Training loss w wybranej epoce: 0.341721; validation loss: 1.780054.
- CPU forward validation: 5.549 s, 0.2399 ms/uderzenie; bez loading/loss/Softmax/metryk. To przepustowość batchy 32, nie opóźnienie pojedynczego badania.

| Klasa | Support | Precision | Recall | Specificity | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N | 19094 | 86.56% | 96.13% | 29.48% | 0.910963 | 18355 | 2849 | 739 |
| S | 491 | 7.69% | 0.20% | 99.95% | 0.003968 | 1 | 12 | 490 |
| V | 1444 | 79.31% | 66.90% | 98.84% | 0.725770 | 966 | 252 | 478 |
| F | 38 | 0.20% | 2.63% | 97.88% | 0.003781 | 1 | 490 | 37 |
| Q | 2067 | 69.23% | 6.97% | 99.70% | 0.126593 | 144 | 64 | 1923 |

### Macierz pomyłek

Wiersze = rzeczywista klasa, kolumny = argmax. Bez progów i abstencji.

| True / pred | N | S | V | F | Q |
|---|---:|---:|---:|---:|---:|
| N | 18355 | 10 | 199 | 482 | 48 |
| S | 460 | 1 | 19 | 7 | 4 |
| V | 464 | 2 | 966 | 1 | 11 |
| F | 28 | 0 | 8 | 1 | 1 |
| Q | 1897 | 0 | 26 | 0 | 144 |

Największe kierunki pomyłek: **Q → N: 1897; N → F: 482; V → N: 464; S → N: 460; N → V: 199; N → Q: 48**. Interpretować je łącznie z nierównowagą: N stanowi większość, a niewielka liczba błędów rzadkiej klasy może znacznie zmienić jej recall/precision. S: TP 1/491, FP 12; Q: TP 144/2067, FP 64.

**F pozostaje problematyczna: TP=1/38, recall 2.63%, precision 0.20%. Przy tak niskiej czułości i precyzji klasa F nie została wiarygodnie rozpoznana. Dodatkowym ograniczeniem jest tylko 38 walidacyjnych uderzeń i cztery grupy z F. Nie można na tej podstawie uznać klasy za niezawodnie rozpoznawaną.**

## Przebieg strat i metryk

| Epoka | Train loss | Validation loss | Val accuracy | Val Macro F1 | Val Macro AP | Czas s |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.341721 | 1.780054 | 84.15% | 0.354215 | 0.416958 | 116.38 |
| 2 | 0.178921 | 2.815819 | 53.49% | 0.287916 | 0.363386 | 96.53 |
| 3 | 0.149734 | 2.280909 | 74.33% | 0.320062 | 0.457688 | 95.17 |
| 4 | 0.133631 | 1.927857 | 73.50% | 0.346702 | 0.460038 | 93.61 |
| 5 | 0.121159 | 2.103351 | 76.22% | 0.343723 | 0.464979 | 93.55 |

Strata treningowa zmieniła się od 0.341721 do 0.121159; walidacyjna od 1.780054 do 2.103351. Najniższa validation loss wystąpiła w epoce 1; model wybierano jednak według Macro F1, nie minimum straty. Training Macro F1 online wzrósł od 0.795551 do 0.911735, bez analogicznej poprawy walidacji. Jest to zgodne z problemem generalizacji/przeuczenia, nie dowodzi jednej konkretnej przyczyny. Wahania walidacyjne mogą odzwierciedlać nierównowagę, różnice pacjentów i przeuczenie; sama krzywa nie identyfikuje przyczyny. Training metrics są online, przed kolejnymi aktualizacjami z dropout/BN, więc nie są oceną jednego zamrożonego modelu.

## Pacjenci/grupy i klasy rzadkie

Trening F: 372/401 uderzeń pochodzi z 208. Walidacja F ma cztery grupy, łącznie tylko 38 przykładów. Q w treningu jest zdominowana przez 102/107, a 2062 z 2067 walidacyjnych Q pochodzi z 104. Liczebność uderzeń nie równa się liczbie niezależnych pacjentów. Nie wykonano dodatkowego doboru podziału ani rezygnacji z trudnych pacjentów.

Poniżej metryki S/F/Q według grup. Brak prawdziwych przykładów klasy oznacza nieokreślony recall, nie zero; false positives pozostają policzone.

| Grupa | Klasa | Support | TP | FP | Recall | Precision |
|---|---|---:|---:|---:|---:|---:|
| mitdb_group_104 | S | 0 | 0 | 0 | nieokreślone | nieokreślone |
| mitdb_group_104 | F | 0 | 0 | 0 | nieokreślone | nieokreślone |
| mitdb_group_104 | Q | 2062 | 144 | 10 | 6.98% | 93.51% |
| mitdb_group_105 | S | 0 | 0 | 0 | nieokreślone | nieokreślone |
| mitdb_group_105 | F | 0 | 0 | 1 | nieokreślone | 0.00% |
| mitdb_group_105 | Q | 5 | 0 | 27 | 0.00% | 0.00% |
| mitdb_group_119 | S | 0 | 0 | 0 | nieokreślone | nieokreślone |
| mitdb_group_119 | F | 0 | 0 | 0 | nieokreślone | nieokreślone |
| mitdb_group_119 | Q | 0 | 0 | 0 | nieokreślone | nieokreślone |
| mitdb_group_121 | S | 1 | 0 | 0 | 0.00% | nieokreślone |
| mitdb_group_121 | F | 0 | 0 | 16 | nieokreślone | 0.00% |
| mitdb_group_121 | Q | 0 | 0 | 3 | nieokreślone | 0.00% |
| mitdb_group_201_202 | S | 183 | 0 | 0 | 0.00% | nieokreślone |
| mitdb_group_201_202 | F | 3 | 0 | 430 | 0.00% | 0.00% |
| mitdb_group_201_202 | Q | 0 | 0 | 4 | nieokreślone | 0.00% |
| mitdb_group_205 | S | 3 | 0 | 10 | 0.00% | 0.00% |
| mitdb_group_205 | F | 11 | 0 | 0 | 0.00% | nieokreślone |
| mitdb_group_205 | Q | 0 | 0 | 0 | nieokreślone | nieokreślone |
| mitdb_group_210 | S | 22 | 0 | 2 | 0.00% | 0.00% |
| mitdb_group_210 | F | 10 | 1 | 42 | 10.00% | 2.33% |
| mitdb_group_210 | Q | 0 | 0 | 3 | nieokreślone | 0.00% |
| mitdb_group_222 | S | 209 | 0 | 0 | 0.00% | nieokreślone |
| mitdb_group_222 | F | 0 | 0 | 0 | nieokreślone | nieokreślone |
| mitdb_group_222 | Q | 0 | 0 | 0 | nieokreślone | nieokreślone |
| mitdb_group_223 | S | 73 | 1 | 0 | 1.37% | 100.00% |
| mitdb_group_223 | F | 14 | 0 | 1 | 0.00% | 0.00% |
| mitdb_group_223 | Q | 0 | 0 | 17 | nieokreślone | 0.00% |

Koncentracja błędów: FN S: mitdb_group_222=209, mitdb_group_201_202=183, mitdb_group_223=72; FN F: mitdb_group_223=14, mitdb_group_205=11, mitdb_group_210=9; FN Q: mitdb_group_104=1918, mitdb_group_105=5; Najwięcej FP F: mitdb_group_201_202=430, mitdb_group_210=42, mitdb_group_121=16. To lokalizacja pomyłek, nie dowód ich klinicznej lub technicznej przyczyny.

Pełne macierze i metryki wszystkich klas/grup znajdują się w `analysis/patient_group_validation_metrics.json` oraz CSV. Nie podano beat-level przedziałów ufności sugerujących niezależność uderzeń.

## Weryfikacja i odtwarzalność

- Przed startem sprawdzono kształty, skończoność, oryginalne symbole/mapowanie, SHA256 wszystkich **39 treningowych/walidacyjnych** shardów; testowe NPZ nie były dekodowane ani oceniane.
- Dokładne populacje i sumy macierzy wszystkich epok oraz wybór najlepszego modelu zgadzają się z historią. Suma macierzy grup zgadza się z macierzą całej walidacji.
- Wszystkie checkpointy mają poprawne SHA256; najlepszy stan jest identyczny ze stanem best zapisanym w ostatnim checkpoincie. Dwa niezależnie odtworzone modele zwróciły dokładnie identyczne logity na tym samym małym batchu **treningowym** `[8,2,252]`; bez ponownej oceny testu ani dodatkowych aktualizacji.
- Źródła/model/preprocessing/dane i wcześniejsze wyniki: brak zmian. Zweryfikowano SHA256 612 wcześniejszych plików oraz niezmienność inwentarza/rozmiarów/mtime 19668 plików obu eksportów. Nowy, rosnący console.log tego runu jest jawnie wyłączony z zakresu wcześniejszych artefaktów.
- Seed 42, deterministyczne algorytmy, shuffle seed+epoch, num_workers=0; dokładne wersje zależności i platforma w `reproducibility.json`. Nie powtarzano eksperymentu, aby oszacować zmienność między seedami.
- Testy po treningu: **657 passed, 229 warnings, 249.21 s**. Ostrzeżenia pandas/WFDB pozostawiono widoczne. Testy mechaniki treningu używają małych fixture; historyczne QC nie oznacza oceny nowego modelu na końcowym teście.

## Pliki i bezpieczne wznowienie

- Najlepszy model: `results/model_training/phase22_mitbih_full/run_001/best_model.pt`, SHA256 `83f914122bf389fe0ddde1abe229cafa81abf1e8f8973dbd03cfc86eacb00d75`.
- Ostatni pełny checkpoint: `results/model_training/phase22_mitbih_full/run_001/epoch_005.pt`, obok `.sha256`; zawiera model, najlepszy stan, Adam, RNG i historię.
- Konfiguracja, plan, preflight, dependencies, console/training logs oraz `epoch_XXX_metrics.json` znajdują się w katalogu eksperymentu.
- `analysis/training_history.csv`, `analysis/training_history.json`, `analysis/per_class_validation_metrics.csv`, `analysis/patient_group_class_metrics.csv`, `analysis/confusion_matrix.csv`.
- Wykresy 300 dpi: `analysis/loss_curves.png`, `analysis/validation_macro_f1.png`, `analysis/validation_macro_auprc.png`, `analysis/validation_confusion_matrix.png`, `analysis/validation_confusion_normalized.png`.
- `analysis/checkpoint_reload_check.json`, `analysis/final_integrity_check.json`, `post_training_pytest.log`, sumy kontrolne.

Zapis epok jest nieprzepisywalny. Wznowienie jest możliwe wyłącznie z kompletnego poprawnego checkpointu nieukończonego runu, z identycznym planem/kodem/runtime, do nowego katalogu; częściowa epoka jest odtwarzana. **Obecny zakończony run nie jest automatycznie wznawiany ani przedłużany poza zatwierdzone 15 epok/early stopping.**

## Rekomendowany następny eksperyment — bez uruchamiania

Najbardziej czytelny kolejny kontrolowany punkt odniesienia to **unweighted CE** przy tym samym modelu, seedzie, pacjentach, optymalizatorze i limicie epok, po osobnym zatwierdzeniu. Porównać na tej samej kompletnej walidacji Macro F1/AP, S/F/Q recall i precision oraz błędy N, zamiast zakładać skuteczność wag. Wagi nie naprawią koncentracji F/Q w pojedynczych osobach. Analizę per-group i morfologii błędów można prowadzić bez końcowego testu; nie zmieniać mapowania/wyłączać trudnych grup w reakcji na wynik.

To propozycja, nie twierdzenie o poprawie i nie automatyczny kolejny run. Powtarzana selekcja na dziewięciu grupach walidacyjnych grozi przeuczeniem walidacji; ograniczyć liczbę porównań i z góry ustalić kryteria. Końcowy test pozostaje zamknięty do osobnej zgody na ocenę ostatecznie zamrożonego modelu. Projekt jest prototypem badawczym, nie klinicznie zwalidowanym urządzeniem diagnostycznym. **Koniec fazy 22.**
