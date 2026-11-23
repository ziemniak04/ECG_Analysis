# Faza 23A — kontrolowany trening unweighted i audyt błędów walidacyjnych MIT-BIH

**Status: ukończono dokładnie jeden nowy eksperyment.** Model B trenowano od zera. Model A fazy 22 nie był ponownie trenowany ani nadpisany. Oba zamrożone checkpointy oceniono na tej samej pełnej walidacji. Końcowy test MIT-BIH nie został otwarty do oceny ani audytu; nie wykonano nowej ewaluacji PTB-XL.

## Identyfikacja i kontrola eksperymentu

A: `results/model_training/phase22_mitbih_full/run_001/best_model.pt`, SHA256 `83f914122bf389fe0ddde1abe229cafa81abf1e8f8973dbd03cfc86eacb00d75`. B: `results/model_training/phase23a_mitbih_unweighted/run_001/best_model.pt`, SHA256 `6f1b46426fa5202ae7d958aafd1ea0bf918bf11be0eb41cc0cc53230cadbc4af`.

Jedyna zamierzona zmiana: `sqrt_balanced` → **unweighted CrossEntropyLoss**, `weight=None`. Wszystkie pozostałe pola konfiguracji, kod trainera/modelu/loadera/metryk, wersje torch/numpy, manifest, konfiguracja modelu oraz zamrożony split porównano programowo z planem fazy 22. Wagi fazy 22 nie służyły do inicjalizacji B: istniejący trainer tworzy nowy model po seed_everything(42), bez resume_from. Oba runy mają ten sam shuffle seed+epoch, Adam lr 0.001, batch 32, 2 wątki CPU, max 15 epok, patience 4 i best checkpoint według pełnego walidacyjnego Macro F1. Różne liczby epok/best epoch są dozwolonym skutkiem tej samej reguły early stopping, nie ręczną zmianą budżetu.

Model: niezmieniony CNN 17797 parametrów, X float32 `[B,2,252]`, y long, pięć logitów N/S/V/F/Q. Softmax stosowano dopiero do AP/probability audit, nigdy przed CE. Bez resamplingu, augmentacji, zmiany etykiet, normalizacji, filtrów, odprowadzeń czy inicjalizacji z nauczonego checkpointu.

Dane: 62926 treningowych uderzeń / 29 grup (N52254,S1715,V4389,F401,Q4167), 23134 walidacyjne / 9 grup (N19094,S491,V1444,F38,Q2067). 201 i 202 należą do jednej grupy walidacyjnej. Podział `configs/mitbih_patient_split_v1.json` SHA256 `8c5cc28d8dcae170fd222432d9f247fd19207ba01be60763241da4b7cae61831`; grupy są rozłączne. Liczby końcowego testu pozostają wyłącznie metadanymi zamrożonego manifestu; nie analizowano jego symboli, sygnałów ani błędów.

## Przebieg nowego runu

Katalog `results/model_training/phase23a_mitbih_unweighted/run_001`. Ukończono **10 epok**, best epoch **6**, status **early_stopped**, czas całego wywołania train_model **1872.77 s (31.21 min)**. Czas A: 499.89 s. Pomimo tych samych dwóch wątków obciążenie systemu i warunki termiczne nie były kontrolowane; różnicy czasu nie wolno przypisywać samej funkcji straty.

| Epoka B | Online train CE | Validation CE | Val accuracy | Val Macro F1 | Val Macro AP | Czas s |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.175406 | 0.781362 | 86.09% | 0.337939 | 0.400059 | 211.42 |
| 2 | 0.079908 | 1.436195 | 73.43% | 0.316647 | 0.324027 | 182.95 |
| 3 | 0.062449 | 1.034644 | 81.13% | 0.340556 | 0.400967 | 188.91 |
| 4 | 0.054210 | 1.518500 | 74.50% | 0.330159 | 0.341481 | 187.33 |
| 5 | 0.049845 | 1.505526 | 73.74% | 0.321188 | 0.344238 | 196.10 |
| 6 | 0.043334 | 1.138666 | 85.60% | 0.342391 | 0.414563 | 194.13 |
| 7 | 0.041956 | 1.519576 | 70.53% | 0.313813 | 0.392737 | 207.29 |
| 8 | 0.038989 | 1.207630 | 73.39% | 0.323337 | 0.414824 | 202.20 |
| 9 | 0.036401 | 1.265109 | 83.68% | 0.335904 | 0.370540 | 165.72 |
| 10 | 0.034490 | 1.716485 | 84.35% | 0.340237 | 0.344877 | 126.24 |

## Uczciwe porównanie zamrożonych modeli

Każdy wynik pochodzi z pełnych **23134 tych samych uderzeń walidacyjnych**, eval() i inference_mode(), argmax, bez progów/abstencji. Sprawdzono zgodność populacji, unikalność indeksów i odtworzenie metryk wybranej epoki. Dataset cache napełniono przed oboma przebiegami; czas forward nie obejmuje loading/CE/Softmax/metadanych. Czasy są opisowym pomiarem batchy 32 na bieżącym CPU, nie kontrolowanym benchmarkiem mikroarchitektury. Softmax nie jest deklaracją klinicznej kalibracji.

| Model | Accuracy | Macro F1 | Macro AP | Klasy z recall >0 | Wspólna val CE bez wag | Wspólna val CE sqrt | Forward ms/beat |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_sqrt_balanced | 84.15% | 0.354215 | 0.416958 | 5 | 0.763867 | 1.780053 | 0.3224 |
| B_unweighted | 85.60% | 0.342391 | 0.414563 | 4 | 1.138666 | 2.941174 | 0.3326 |

Według wcześniej używanego kryterium pełnego walidacyjnego Macro F1 lepszy jest **A_sqrt_balanced**. Zmiana B−A: Macro F1 -0.011824, Macro AP -0.002396, Accuracy +1.45 punktu procentowego. Nie oznacza to automatycznie poprawy każdej klasy; poniżej podano wszystkie trade-offy. To jeden seed i jeden podział, bez dowodu przewagi statystycznej ani klinicznej.

### Pełne metryki klas

Support/TP/FP/FN są beat-level, nie liczbami niezależnych pacjentów. Macro F1 i Macro AP obejmują wszystkie pięć klas, również z zerowym recall. AP to nieinterpolowana one-vs-rest average precision, nie pole trapezami. Nieokreślona precision przy braku predykcji to null, nie sztuczne zero. Klasy AAMI-style to grupy uderzeń, nie pięć nazwanych diagnoz rytmu.

| Model | Klasa | Support | Precision | Recall | Specificity | F1 | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A_sqrt_balanced | N | 19094 | 86.56% | 96.13% | 29.48% | 0.910963 | 2849 | 739 |
| A_sqrt_balanced | S | 491 | 7.69% | 0.20% | 99.95% | 0.003968 | 12 | 490 |
| A_sqrt_balanced | V | 1444 | 79.31% | 66.90% | 98.84% | 0.725770 | 252 | 478 |
| A_sqrt_balanced | F | 38 | 0.20% | 2.63% | 97.88% | 0.003781 | 490 | 37 |
| A_sqrt_balanced | Q | 2067 | 69.23% | 6.97% | 99.70% | 0.126593 | 64 | 1923 |
| B_unweighted | N | 19094 | 86.50% | 98.30% | 27.50% | 0.920256 | 2929 | 324 |
| B_unweighted | S | 491 | 0.00% | 0.00% | 100.00% | 0.000000 | 1 | 491 |
| B_unweighted | V | 1444 | 85.51% | 70.71% | 99.20% | 0.774071 | 173 | 423 |
| B_unweighted | F | 38 | 0.48% | 2.63% | 99.10% | 0.008097 | 208 | 37 |
| B_unweighted | Q | 2067 | 32.26% | 0.48% | 99.90% | 0.009533 | 21 | 2057 |

Zmiany klasowe: N: recall 96.13% → 98.30%, precision 86.56% → 86.50%, FP 2849 → 2929, FN 739 → 324; S: recall 0.20% → 0.00%, precision 7.69% → 0.00%, FP 12 → 1, FN 490 → 491; V: recall 66.90% → 70.71%, precision 79.31% → 85.51%, FP 252 → 173, FN 478 → 423; F: recall 2.63% → 2.63%, precision 0.20% → 0.48%, FP 490 → 208, FN 37 → 37; Q: recall 6.97% → 0.48%, precision 69.23% → 32.26%, FP 64 → 21, FN 1923 → 2057. Liczba klas z niezerowym recall jest tylko wskaźnikiem pokrycia — pojedynczy TP nie oznacza użytecznego rozpoznawania klasy. Wyższa Accuracy może wynikać z częstszego N przy spadku recall mniejszości; punkt odniesienia always-N na walidacji to 82.54%.

### A_sqrt_balanced

| True / pred | N | S | V | F | Q |
|---|---:|---:|---:|---:|---:|
| N | 18355 | 10 | 199 | 482 | 48 |
| S | 460 | 1 | 19 | 7 | 4 |
| V | 464 | 2 | 966 | 1 | 11 |
| F | 28 | 0 | 8 | 1 | 1 |
| Q | 1897 | 0 | 26 | 0 | 144 |
### B_unweighted

| True / pred | N | S | V | F | Q |
|---|---:|---:|---:|---:|---:|
| N | 18770 | 1 | 106 | 203 | 14 |
| S | 451 | 0 | 30 | 5 | 5 |
| V | 421 | 0 | 1021 | 0 | 2 |
| F | 25 | 0 | 12 | 1 | 0 |
| Q | 2032 | 0 | 25 | 0 | 10 |

## Audyt strat — brak mieszania różnych funkcji celu

W obu trybach train/validation istniejący trainer liczy dokładną stratę jako `sum(w_y * CE_i) / sum(w_y)`. Dla B wagi są 1, więc mianownik to liczba uderzeń. Sumowane są liczniki i mianowniki wszystkich batchy, nie nieważona średnia średnich. Te same **treningowe** wagi fazy 22 obowiązywały w jej train i validation. Nie dopasowywano wag do walidacji.

Wspólne CE obu modeli przeliczono z tych samych zamrożonych walidacyjnych logitów: raz bez wag, raz z oryginalnymi treningowymi wagami sqrt. Własna CE każdego modelu odtwarza zapis wybranej epoki z tolerancją numeryczną; nie wykryto błędu agregacji, pominiętego batcha ani innego zbioru. Ręczny test porównuje agregację z bezpośrednią PyTorch CrossEntropyLoss, również dla ostatniego niepełnego batcha.

Dodatkowo wykonano jedynie inferencję loss-only obu zamrożonych modeli na pełnych 62926 treningowych uderzeniach w eval/inference_mode. Nie było backward, optimizer step ani nowej selekcji modelu. Pozwala to porównywać tryb i funkcję celu identycznie z walidacją:

| Model | Frozen train CE bez wag | Frozen val CE bez wag | Frozen train CE sqrt | Frozen val CE sqrt |
|---|---:|---:|---:|---:|
| A_sqrt_balanced | 0.073295 | 0.763867 | 0.177355 | 1.780053 |
| B_unweighted | 0.040997 | 1.138666 | 0.147613 | 2.941174 |

Historia online i frozen train dotyczą różnych chwil/trybów; utrzymywanie się luki frozen train–validation wskazuje na brak generalizacji po wyrównaniu trybu, ale nie identyfikuje przyczyny. Sama obecność BatchNorm nie dowodzi jego błędu.

Historyczna train CE A i B ma inne znaczenie i skalę — nie można z samej różnicy wnioskować o poprawie B. Ponadto training loss/metrics są online, przed kolejnymi aktualizacjami z aktywnymi dropout/BatchNorm, a validation jest po epoce w eval/inference_mode. Sama luka strat nie dowodzi błędu obliczeń ani jednej przyczyny słabej generalizacji. Nie wykonano dodatkowego pełnego treningu ani oceny testu, by „naprawić” lukę.

## Symbole źródłowe i przesunięcie składu klas

Mapowanie pozostaje bez zmian: N←N/L/R/e/j, S←A/a/J/S, V←V/E, F←F, Q←`/`, `f`, `Q`. Duże F to inna etykieta niż małe f; markerów rytmu nie potraktowano jako uderzeń. Oryginalne symbole są zachowane w każdej predykcji CSV i audycie.

| Część | Symbol MIT-BIH | Grupa | Liczba |
|---|---|---|---:|
| train | `/` | Q | 4105 |
| train | `A` | S | 1676 |
| train | `F` | F | 401 |
| train | `J` | S | 31 |
| train | `L` | N | 6615 |
| train | `N` | N | 40287 |
| train | `Q` | Q | 6 |
| train | `R` | N | 5345 |
| train | `S` | S | 2 |
| train | `V` | V | 4389 |
| train | `a` | S | 6 |
| train | `f` | Q | 56 |
| train | `j` | N | 7 |
| validation | `/` | Q | 1378 |
| validation | `A` | S | 350 |
| validation | `E` | V | 1 |
| validation | `F` | F | 38 |
| validation | `J` | S | 2 |
| validation | `N` | N | 18856 |
| validation | `Q` | Q | 23 |
| validation | `V` | V | 1443 |
| validation | `a` | S | 139 |
| validation | `e` | N | 16 |
| validation | `f` | Q | 666 |
| validation | `j` | N | 222 |

Potwierdzone różnice:

- S: 1382 z 1715 treningowych S (80.58%) pochodzi z nagrania/grupy 232; liczba przykładów nie zastępuje różnorodności osób. `a` 6/1715 w treningu (0.35%) versus 139/491 w walidacji (28.31%); `A` 1676 versus 350. Ocena zbiorczej S miesza odmienne symbole i pacjentów.
- Q: `/` 4105, `f` 56, `Q` 6 w treningu; `/` 1378, `f` 666, `Q` 23 w walidacji. Q jest heterogeniczna; udział `f` rośnie z 1.34% do 32.22%. Nie relabelowano ani nie usuwano tych przykładów. Błędy według symbolu są w ORIGINAL_SYMBOL_ERRORS.csv.
- N: L/R występują w treningu (6615/5345), nie w tej walidacji; j/e mają inny udział. To opis składu konkretnego zamrożonego podziału, nie definicja całej bazy.
- F: trening 401 przykładów z 10 grup, ale 372 z 208; walidacja 38 z czterech grup. Q w treningu jest zdominowana przez 102/107, w walidacji przez 104.

Różnice te są zgodne z ryzykiem domain shift i zależności od pacjenta, ale nie identyfikują przyczyny konkretnych błędów bez dalszego audytu. Nie zmieniamy podziału po obejrzeniu wyników i nie analizujemy końcowego testu, by dostrajać dane.

## Odprowadzenia i grupa 201/202

| Część | Kanał 0 | Kanał 1 | Liczba nagrań |
|---|---|---|---:|
| train | MLII | V1 | 22 |
| train | MLII | V2 | 2 |
| train | MLII | V4 | 1 |
| train | MLII | V5 | 2 |
| train | V5 | MLII | 1 |
| train | V5 | V2 | 1 |
| validation | MLII | V1 | 9 |
| validation | V5 | V2 | 1 |

Indeks kanału nie gwarantuje identycznego anatomicznego odprowadzenia; zachowano nazwy i kolejność eksportu, bez harmonizacji. 201/202 mają MLII/V1, tak jak 22 treningowe nagrania. 104 ma V5/V2, reprezentowane też przez treningowe 102. Z samych nazw odprowadzeń nie wynika wyjaśnienie błędów; nie przypisujemy im przyczynowości.

| Model | Rekord | Uderzenia | F false positives |
|---|---|---:|---:|
| A_sqrt_balanced | 201 | 1963 | 155 |
| A_sqrt_balanced | 202 | 2135 | 275 |
| B_unweighted | 201 | 1963 | 54 |
| B_unweighted | 202 | 2135 | 31 |

FP F według symbolu w tych dwóch rekordach: A_sqrt_balanced / 201 / symbol A: 1; A_sqrt_balanced / 201 / symbol N: 147; A_sqrt_balanced / 201 / symbol a: 3; A_sqrt_balanced / 201 / symbol j: 4; A_sqrt_balanced / 202 / symbol A: 3; A_sqrt_balanced / 202 / symbol N: 272; B_unweighted / 201 / symbol N: 49; B_unweighted / 201 / symbol a: 4; B_unweighted / 201 / symbol j: 1; B_unweighted / 202 / symbol A: 1; B_unweighted / 202 / symbol N: 30. W grupie 210 FP F zmieniły się z 42 do 122; część problemu jest więc silniejsza w innej grupie mimo niższej sumy FP. Pełny audyt obejmuje wszystkie walidacyjne rekordy, nie tylko wybrane przykłady. Symbole i pozycje adnotacji pozwalają później obejrzeć te same rzeczywiste okna bez ich przesuwania czy zmiany etykiet.

### Metryki S/F/Q według grup

| Model | Grupa | Klasa | Support | TP | FP | Recall |
|---|---|---|---:|---:|---:|---:|
| A_sqrt_balanced | mitdb_group_104 | Q | 2062 | 144 | 10 | 6.98% |
| A_sqrt_balanced | mitdb_group_105 | F | 0 | 0 | 1 | nieokreślone |
| A_sqrt_balanced | mitdb_group_105 | Q | 5 | 0 | 27 | 0.00% |
| A_sqrt_balanced | mitdb_group_121 | S | 1 | 0 | 0 | 0.00% |
| A_sqrt_balanced | mitdb_group_121 | F | 0 | 0 | 16 | nieokreślone |
| A_sqrt_balanced | mitdb_group_121 | Q | 0 | 0 | 3 | nieokreślone |
| A_sqrt_balanced | mitdb_group_201_202 | S | 183 | 0 | 0 | 0.00% |
| A_sqrt_balanced | mitdb_group_201_202 | F | 3 | 0 | 430 | 0.00% |
| A_sqrt_balanced | mitdb_group_201_202 | Q | 0 | 0 | 4 | nieokreślone |
| A_sqrt_balanced | mitdb_group_205 | S | 3 | 0 | 10 | 0.00% |
| A_sqrt_balanced | mitdb_group_205 | F | 11 | 0 | 0 | 0.00% |
| A_sqrt_balanced | mitdb_group_210 | S | 22 | 0 | 2 | 0.00% |
| A_sqrt_balanced | mitdb_group_210 | F | 10 | 1 | 42 | 10.00% |
| A_sqrt_balanced | mitdb_group_210 | Q | 0 | 0 | 3 | nieokreślone |
| A_sqrt_balanced | mitdb_group_222 | S | 209 | 0 | 0 | 0.00% |
| A_sqrt_balanced | mitdb_group_223 | S | 73 | 1 | 0 | 1.37% |
| A_sqrt_balanced | mitdb_group_223 | F | 14 | 0 | 1 | 0.00% |
| A_sqrt_balanced | mitdb_group_223 | Q | 0 | 0 | 17 | nieokreślone |
| B_unweighted | mitdb_group_104 | Q | 2062 | 9 | 0 | 0.44% |
| B_unweighted | mitdb_group_105 | Q | 5 | 1 | 14 | 20.00% |
| B_unweighted | mitdb_group_121 | S | 1 | 0 | 0 | 0.00% |
| B_unweighted | mitdb_group_201_202 | S | 183 | 0 | 0 | 0.00% |
| B_unweighted | mitdb_group_201_202 | F | 3 | 1 | 85 | 33.33% |
| B_unweighted | mitdb_group_201_202 | Q | 0 | 0 | 5 | nieokreślone |
| B_unweighted | mitdb_group_205 | S | 3 | 0 | 0 | 0.00% |
| B_unweighted | mitdb_group_205 | F | 11 | 0 | 0 | 0.00% |
| B_unweighted | mitdb_group_210 | S | 22 | 0 | 0 | 0.00% |
| B_unweighted | mitdb_group_210 | F | 10 | 0 | 122 | 0.00% |
| B_unweighted | mitdb_group_222 | S | 209 | 0 | 1 | 0.00% |
| B_unweighted | mitdb_group_222 | F | 0 | 0 | 1 | nieokreślone |
| B_unweighted | mitdb_group_223 | S | 73 | 0 | 0 | 0.00% |
| B_unweighted | mitdb_group_223 | F | 14 | 0 | 0 | 0.00% |
| B_unweighted | mitdb_group_223 | Q | 0 | 0 | 2 | nieokreślone |

**38 walidacyjnych uderzeń F, skupionych w niewielkiej liczbie pacjentów, nie wystarcza do wiarygodnych samodzielnych wniosków o tej klasie.** Nie traktować skorelowanych uderzeń jako niezależnych pacjentów, nie interpretować skoków jednego TP jako trwałej poprawy. Zachowano obie macierze, także zera, bez ukrywania nieskutecznych klas.

## Integralność, testy i artefakty

Preflight zweryfikował SHA256 i zgodność sygnałów/etykiet wszystkich 39 treningowych/walidacyjnych shardów. Testowe NPZ pozostały zamknięte; inwentarz obu eksportów sprawdzano wyłącznie stat dla ochrony niezmienności. Po pracy: **664 wcześniejszych plików SHA256 bez zmian**, **19668 plików eksportów bez zmiany rozmiaru/mtime/inwentarza**. Zachowano wszystkie artefakty PTB-XL i MIT-BIH fazy 22. Oba checkpointy i ich parametry/bufory pozostały identyczne po inferencji. Dodatkowy reload B na małym batchu treningowym daje identyczne logity; nie jest to ocena testowa.

Testy po treningu: **666 passed, 229 warnings, 295.24 s**. Uruchomiono nowe testy loss-only config, dokładnej ważonej/nieważonej CE, odmowy dostępu do testu, kompletu walidacji, metadanych i niezmienności modelu, a także cały istniejący zestaw. Deprecacje pandas/WFDB pozostawiono widoczne. Historyczne QC i syntetyczne testy nie oznaczają oceny nowego modelu na końcowym teście.

Pliki:

- `results/model_training/phase23a_mitbih_unweighted/run_001/best_model.pt`, `results/model_training/phase23a_mitbih_unweighted/run_001/epoch_010.pt` i ich SHA256; configuration.json, training_plan.json, training.log, epoch metrics, execution.json. Ostatni checkpoint zawiera Adam/RNG i historię; nie przedłużać automatycznie zakończonego runu.
- `analysis/`: historia CSV/JSON, własne krzywe strat/F1/AP B, macierze, metryki klas i grup, kontrola reload.
- `comparison/`: MODEL_COMPARISON.csv, CLASS_COMPARISON.csv, PATIENT_GROUP_COMPARISON.csv, ORIGINAL_SYMBOL_AAMI_AUDIT.csv, ORIGINAL_SYMBOL_ERRORS.csv, F_FALSE_POSITIVE_SYMBOL_AUDIT.csv, osobne predykcje CSV i NPZ obu modeli, macierze rekordów 201/202, integrity.json, porównawcze wykresy confusion_matrices.png i training_comparison.png.
- Preflight: `results/model_training/phase23a_mitbih_unweighted/preflight_001`, w tym per-record odprowadzenia/liczebności i oryginalne symbole. Suma kontrolna wszystkich nowych plików jest zapisana osobno po zakończeniu.

## Wniosek i następny krok — bez uruchamiania

Według wcześniej używanego kryterium pełnego walidacyjnego Macro F1 lepszy jest **A_sqrt_balanced**. Zmiana B−A: Macro F1 -0.011824, Macro AP -0.002396, Accuracy +1.45 punktu procentowego. Nie oznacza to automatycznie poprawy każdej klasy; poniżej podano wszystkie trade-offy. To jeden seed i jeden podział, bez dowodu przewagi statystycznej ani klinicznej.

Następnie warto przejrzeć błędy **według oryginalnego symbolu, grupy i rzeczywistych okien** (zwłaszcza S typu a, Q typu f oraz FP F w 201/202 i 210), a także wyjaśnić pozostałą lukę między zmierzoną frozen train CE i validation CE, uwzględniając koncentrację pacjentów i statystyki BatchNorm. Audyt loss-only już wyrównał tryb eval; obecny eksperyment nadal nie rozstrzyga jednej przyczyny generalizacji. Dopiero po takim audycie i osobnym zatwierdzeniu planować jedną zmianę modelu lub strategii; nie zakładać, że silniejsze wagi rozwiążą koncentrację pacjentów.

Nie uruchomiono dodatkowego wariantu ani nowego podziału. Powtarzana selekcja na tej samej walidacji może ją przeuczyć; ograniczyć liczbę porównań i ustalić kryteria z góry. Nie oceniać końcowego testu przed ostatecznym zamrożeniem rozwiązania i osobną zgodą. **Koniec fazy 23A.**
