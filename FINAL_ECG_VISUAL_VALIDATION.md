# Finalna wizualna walidacja preprocessingu ECG

Wyłącznie rzeczywiste, lokalne MIT-BIH i PTB-XL. Nowe figury demonstracyjne; nie zmieniono danych, metod preprocessingu ani wcześniejszych raportów. Nie pobierano danych i nie trenowano modeli.

## Metoda i wybór przykładów

MIT-BIH: pełne nagranie przetworzono istniejącym compare_recording, potem wycięto 10–20 s. Z-score jest liczony per nagranie i kanał, dlatego 10-sekundowy fragment nie musi mieć średniej 0 i std 1. Uderzenia skopiowano bez zmian z wyeksportowanych NPZ; wybrano pierwsze kompletne okno po 10 s dla każdej klasy według kolejności rekordów 100,200,208,102. Każdy symbol i pozycję sprawdzono w oryginalnym .atr i istniejącym mapowaniu AAMI.

PTB-XL: pierwszy kwalifikujący się ecg_id każdej klasy, dokładnie jeden kod spośród wszystkich 12 rozpoznanych kodów rytmu. Nie wybierano rekordu według wyglądu ani likelihood. NaN/brak kodu nie oznacza SR; zero score może oznaczać nieznaną pewność. Odczyt przez WFDB, walidowane filtrowanie i normalizacja; przeliczony wynik float32 jest identyczny z gotowym eksportem. Oryginalne SCP, ID i foldy są w manifeście.

Filtry: istniejące Butterworth SOS 0,5 Hz high-pass / 45 Hz low-pass, rząd 4, zero-phase offline. Notch wyłączony; 50/60 Hz jest na lub ponad Nyquistem przy 100 Hz. Normalizacja Z-score niezależna per rekord/per lead; bez wspólnych parametrów z innych pacjentów.

Reguła wyboru jest deterministyczna i służy ilustracji. Pojedynczy przykład nie jest statystycznie reprezentatywną próbką klasy, oceną skuteczności ani potwierdzeniem rozpoznania klinicznego.

## Figury i podpisy

### Rysunek 1

MIT-BIH 100, oba oryginalne kanały, 10–20 s. Czerwone punkty i symbole wskazują niezmienione pozycje eksperckich adnotacji. Raw i filtered mają wspólne granice amplitudy w mV; Z-score jest bezwymiarowy, liczony na całym nagraniu przed wyborem fragmentu.

![Rysunek 1: mit_stages](results/preprocessing/final_visual_validation/mitbih_100_stages.png)

[Wektorowy PDF](results/preprocessing/final_visual_validation/mitbih_100_stages.pdf)

### Rysunek 2

MIT-BIH 200, oba oryginalne kanały, 10–20 s. Czerwone punkty i symbole wskazują niezmienione pozycje eksperckich adnotacji. Raw i filtered mają wspólne granice amplitudy w mV; Z-score jest bezwymiarowy, liczony na całym nagraniu przed wyborem fragmentu.

![Rysunek 2: mit_stages](results/preprocessing/final_visual_validation/mitbih_200_stages.png)

[Wektorowy PDF](results/preprocessing/final_visual_validation/mitbih_200_stages.pdf)

### Rysunek 3

Po jednym rzeczywistym wyeksportowanym uderzeniu N/S/V/F/Q, oba kanały w oryginalnej kolejności. Linia t=0 oznacza pozycję adnotacji, nie automatycznie wyznaczony nowy R-peak. Q pokazuje konkretny zachowany symbol, nie całą heterogeniczną klasę. Okna [p−90,p+162) mają 252 próbki przy 360 Hz.

![Rysunek 3: mit_beats](results/preprocessing/final_visual_validation/mitbih_aami_heartbeat_examples.png)

[Wektorowy PDF](results/preprocessing/final_visual_validation/mitbih_aami_heartbeat_examples.pdf)

### Rysunek 4

PTB-XL ecg_id=1, SR (Sinus rhythm), odprowadzenia II/V1/V5. Etykieta wynika z jedynego kodu rytmu spośród wszystkich 12, nie z oceny wykresu. Całe 10 s, filtrowanie 0,5–45 Hz bez notch i Z-score per rekord/per lead; wynik identyczny z gotowym eksportem.

![Rysunek 4: ptb_stages](results/preprocessing/final_visual_validation/ptbxl_SR_00001_stages.png)

[Wektorowy PDF](results/preprocessing/final_visual_validation/ptbxl_SR_00001_stages.pdf)

### Rysunek 5

PTB-XL ecg_id=152, AFIB (Atrial fibrillation), odprowadzenia II/V1/V5. Etykieta wynika z jedynego kodu rytmu spośród wszystkich 12, nie z oceny wykresu. Całe 10 s, filtrowanie 0,5–45 Hz bez notch i Z-score per rekord/per lead; wynik identyczny z gotowym eksportem.

![Rysunek 5: ptb_stages](results/preprocessing/final_visual_validation/ptbxl_AFIB_00152_stages.png)

[Wektorowy PDF](results/preprocessing/final_visual_validation/ptbxl_AFIB_00152_stages.pdf)

### Rysunek 6

PTB-XL ecg_id=103, STACH (Sinus tachycardia), odprowadzenia II/V1/V5. Etykieta wynika z jedynego kodu rytmu spośród wszystkich 12, nie z oceny wykresu. Całe 10 s, filtrowanie 0,5–45 Hz bez notch i Z-score per rekord/per lead; wynik identyczny z gotowym eksportem.

![Rysunek 6: ptb_stages](results/preprocessing/final_visual_validation/ptbxl_STACH_00103_stages.png)

[Wektorowy PDF](results/preprocessing/final_visual_validation/ptbxl_STACH_00103_stages.pdf)

### Rysunek 7

PTB-XL ecg_id=2, SBRAD (Sinus bradycardia), odprowadzenia II/V1/V5. Etykieta wynika z jedynego kodu rytmu spośród wszystkich 12, nie z oceny wykresu. Całe 10 s, filtrowanie 0,5–45 Hz bez notch i Z-score per rekord/per lead; wynik identyczny z gotowym eksportem.

![Rysunek 7: ptb_stages](results/preprocessing/final_visual_validation/ptbxl_SBRAD_00002_stages.png)

[Wektorowy PDF](results/preprocessing/final_visual_validation/ptbxl_SBRAD_00002_stages.pdf)

### Rysunek 8

Cztery etykiety rytmu PTB-XL, po jednym rzeczywistym 10-sekundowym rekordzie, odprowadzenia II i V1. Wykresy pokazują dokładne tablice gotowego eksportu po filtracji i Z-score. Nie jest to diagnoza, estymacja jakości klasyfikacji ani dowód reprezentatywności populacji.

![Rysunek 8: ptb_overview](results/preprocessing/final_visual_validation/ptbxl_four_rhythm_examples.png)

[Wektorowy PDF](results/preprocessing/final_visual_validation/ptbxl_four_rhythm_examples.pdf)

## Wizualna walidacja i ograniczenia

W każdej parze raw/filtered użyto wspólnych granic osi amplitudy i tej samej osi czasu. Redukcja składowych wolnozmiennych i wysokoczęstotliwościowych może zmieniać amplitudę oraz kształt przebiegu. Z-score zmienia skalę i offset; nie jest dodatkowym filtrem. Dla pełnych 10 s PTB-XL statystyki są per lead; dla fragmentu MIT-BIH wynik odnosi się do całego nagrania.

Markery MIT-BIH i początek względnej osi okna pozostają dokładnie w pozycjach eksperckich. Nie szukano nowych R-peaków ani nie przesuwano sygnału. Zero-phase nie dodaje stałego opóźnienia filtru, ale nie gwarantuje, że wszystkie lokalne ekstrema surowego przebiegu zachowają dokładnie ten sam kształt lub indeks po filtracji.

Q jest heterogeniczną grupą AAMI; jeśli wybrany symbol to /, jest to uderzenie stymulowane, nie przykład każdego rodzaju unknown. N nie oznacza zdrowego pacjenta. F to grupa fusion beat, nie migotanie przedsionków. AFIB/STACH/SBRAD są etykietami całego rekordu PTB-XL i nie zastępują klas uderzeń. Oznaczenia są potwierdzone metadanymi/adnotacjami, nie rozpoznawane na podstawie wizualnej morfologii.

Filtracja offline może mieć efekty brzegowe, usuwać informacje o amplitudzie i subtelnej morfologii; wizualna kontrola nie dowodzi zachowania wszystkich P/QRS/T ani przydatności klinicznej. Sygnały 100 Hz nie zawierają całej informacji dostępnej przy 500 Hz. Nie wykonano delineacji fal ani oceny medycznej.

## Odtwarzanie i artefakty

PNG: 400 dpi; dodatkowo PDF z osadzonymi fontami i wektorowymi przebiegami. Osie i jednostki w języku angielskim do wykorzystania w publikacji. Trzy wybrane leads PTB-XL zamiast wszystkich 12; MIT-BIH pokazuje oba oryginalne kanały, bez zamiany nazw.

Pliki: results/preprocessing/final_visual_validation/. Manifest zawiera wybrane beat indices, symbole, pozycje, ecg_id/patient_id/fold, konfiguracje, SHA256 i podpisy. CSV opisuje dokładnie pokazane fragmenty MIT-BIH, nie globalne statystyki populacji.

Uruchomienie z katalogu projektu: `.venv/bin/python -m examples.final_ecg_visual_validation`. Skrypt odmawia nadpisania istniejącego raportu/katalogu; kolejne wykonanie wymaga nowych ścieżek przez --output-dir i --report-path.

## Wynik przeglądu wizualnego wszystkich ośmiu figur

Sprawdzono wszystkie osiem PNG: kompletność paneli, czytelność opisów, jednostki, wspólne osie czasu, oryginalne nazwy kanałów i brak uciętych głównych wychyleń. W MIT-BIH 100 widoczna jest zmiana offsetu po high-pass; w kanale V1 nagrania 200 redukcja szybkich oscylacji jest bardziej widoczna. W PTB-XL AFIB 152 i STACH 103 linia bazowa po filtracji jest bardziej wycentrowana niż w raw. Są to obserwacje tych konkretnych wykresów, nie ilościowy benchmark filtrów.

Panel Z-score zachowuje układ czasowy przebiegu filtered, zmieniając amplitudę i offset. Nie stosowano przesunięć ani automatycznej detekcji nowych szczytów. Znacznik t=0 w segmentach nie musi być maksimum w każdym odprowadzeniu — pokazuje oryginalną adnotację źródłową. Nie należy porównywać bezpośrednio amplitud Z-score między pacjentami, kanałami lub panelami o różnych zakresach y. Wartości większe od 3 odchyleń standardowych nie oznaczają automatycznie błędu; nie stosowano clippingu.

Wybrany przykład Q ma oryginalny symbol **Q** z rekordu **208**; nie jest to przykład symbolu stymulatorowego `/`. Nie można nim reprezentować wszystkich podtypów grupy Q. Podobnie wizualne podobieństwo niektórych przykładów N i S nie unieważnia oryginalnej adnotacji i nie jest dowodem zdolności ich rozróżnienia przez przyszły model.

### Dokładne źródła pięciu segmentów

| Klasa | Oryginalny symbol | Rekord | Próbka adnotacji | Wiersz eksportu (od 0) |
|---|---|---|---:|---:|
| N | `N` | 100 | 3862 | 12 |
| S | `A` | 100 | 66792 | 229 |
| V | `V` | 100 | 546792 | 1905 |
| F | `F` | 200 | 218881 | 880 |
| Q | `Q` | 208 | 454171 | 2086 |

PTB-XL: SR → ECG 00001; AFIB → ECG 00152; STACH → ECG 00103; SBRAD → ECG 00002. Dla każdego zachowano pełne oryginalne SCP, patient_id i strat_fold w manifeście, a obecność tylko jednego z wszystkich 12 kodów rhythm zweryfikowano istniejącym adapterem.

## Testy i końcowa kontrola plików

**101 passed, 0 failed, 0 skipped**, 65 ostrzeżeń deprecacyjnych WFDB/pandas, 74,67 s. W tym 13 nowych testów: kompletność ośmiu figur, rozdzielczość PNG i format PDF, podpisy, zgodność pięciu okien z .atr/NPZ, poprawność czterech wyłącznych etykiet SCP, patient_id/fold, nazwy i jednostki kanałów, pozycje na osi czasu oraz SHA256 źródeł i kodu. Pozostałe testy obejmują dotychczasowe porównania preprocessingu, wizualizację, adapter PTB-XL i mapowanie AAMI.

Polecenie: `.venv/bin/python -m pytest tests/test_final_visual_validation.py tests/test_preprocessing_comparison.py tests/test_visualization.py tests/test_ptbxl_rhythm.py tests/test_label_mapping.py -ra`. Log: `results/preprocessing/final_visual_validation/pytest.log`. Nie wyciszano ostrzeżeń.

Niezależnie od testów potwierdzono metadane 400 dpi w każdym PNG i osadzenie fontów we wszystkich ośmiu PDF przez pdffonts. Renderer użył dostępnej grubości bold 700 zamiast semibold (informacja w generation.log); nie dotyczy to danych ani preprocessingu. Zapisano sumy SHA256 wszystkich 16 plików graficznych. Skrypt potwierdził niezmienność rozmiaru i mtime 98826 wcześniej istniejących plików, a SHA256 użytych eksportów i źródeł zgadzały się z istniejącymi manifestami. Nie nadpisano zbiorów, wcześniejszych raportów ani modułów preprocessingu.

Nowe pliki kodu: `examples/final_ecg_visual_validation.py` (lokalny, odtwarzalny generator), `tests/test_final_visual_validation.py` (kontrole na rzeczywistych danych). Szczegóły: visual_validation_manifest.json, final_verification.json, pdf_fonts.json i mitbih_display_statistics.csv w katalogu figur.
