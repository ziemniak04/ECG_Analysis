# ECG — wczytywanie i wizualizacja (fazy 1–3)

Zakres Magdaleny: inżynieria danych i preprocessing ECG. Obecna implementacja obejmuje wyłącznie lokalny odczyt MIT-BIH i wykresy surowych sygnałów. Nie zawiera filtrowania, segmentacji, etykiet klas, eksportu tensorów ani części Anny.

## Lokalizacja i inspekcja

Projekt znaleziono w `~/Pulpit/ECG-APP`; oczekiwany `~/ECG-APP` nie istnieje. Istniejący `main.py` pozostawiono bez zmian.

Dane względem katalogu projektu:

```
data/mitdb/physionet.org/files/mitdb/1.0.0/
```

Inspekcja lokalna: 48 nagłówków `.hea`, 48 plików sygnału `.dat`, 49 plików adnotacji `.atr`. Wszystkie 48 rekordów mają komplet plików; dodatkowy `102-0.atr` nie jest osobnym rekordem. Żaden z tych plików nie jest pusty. Nie pobieramy bazy ponownie.

Nagłówek opisuje częstotliwość próbkowania, kanały i sposób zapisu. `.dat` zawiera próbki, a `.atr` — eksperckie adnotacje. Nie każda adnotacja oznacza uderzenie serca, np. `+` jest znacznikiem zmiany rytmu.

## Środowisko

Systemowy Python: 3.12.3, brak aktywnego venv. Brakuje numpy, wfdb, scipy, pandas i matplotlib; systemowy pytest jest dostępny. `python -m venv .venv` utworzył niepełne środowisko, ponieważ system nie ma ensurepip. Próba instalacji przez `pip --python .venv/bin/python install wfdb matplotlib pytest` nie powiodła się z powodu błędu DNS. Nie zmieniano pakietów systemowych.

Po udostępnieniu pakietu systemowego `python3.12-venv` oraz dostępu do pakietów można utworzyć nowe środowisko pod nową nazwą (bez nadpisywania niepełnego `.venv`):

```bash
cd ~/Pulpit/ECG-APP
python3 -m venv .venv-preprocessing
source .venv-preprocessing/bin/activate
python -m pip install -r requirements_preprocessing.txt
python -m pytest -q
python -m examples.inspect_ecg
```

W VS Code należy wybrać interpreter nowego środowiska. Dodaj jego katalog do lokalnych wykluczeń Git, jeśli używasz nazwy innej niż `.venv`.

## Interfejs danych dla Anny

```python
from src.data_loader import load_recording, list_recordings
record = load_recording("100")
print(record.signals.shape)  # [samples, channels], jeszcze nie wejście CNN
print(record.sampling_frequency)
print(record.channel_names)
print(record.annotation_positions, record.annotation_symbols)
```

Domyślna ścieżka jest wyznaczana względem modułu za pomocą pathlib, niezależnie od bieżącego katalogu. Można przekazać `root=...` wskazujący inny lokalny katalog. Rekurencyjne wyszukiwanie odrzuca niejednoznaczną lokalizację. Odczyt używa `wfdb.rdrecord()` i `wfdb.rdann(..., "atr")`, bez `pn_dir` i bez sieci. Niekompletne rekordy są raportowane przez `inspect_recordings()` i pomijane przez `list_recordings()`; bezpośredni odczyt zgłasza błąd.

Sygnał jest float64 w jednostkach fizycznych nagłówka; dla rekordu 100 są to mV. Sprawdzane są wymiary, wartości skończone, metadane kanałów i zakres oraz kolejność adnotacji. Zachowujemy ID rekordu, pozycje, symbole i notatki pomocnicze. ID rekordu nie jest deklarowane jako ID pacjenta. Ustalenie grup pacjentów i walidacja należą do dalszej integracji; nie wykonujemy podziałów danych.

## Wykresy

`python -m examples.inspect_ecg` drukuje metadane rekordu 100 i zapisuje:

- `results/preprocessing/record_100_first_10s.png`: osobny panel dla każdego kanału, pierwsze 10 sekund; czerwone znaczniki oznaczają adnotacje uderzeń, szare pozostałe adnotacje. Symbole są zachowane bez mapowania do klas.
- `results/preprocessing/records_100_101_200_comparison.png`: pierwsze kanały trzech rekordów na wspólnej osi czasu. Nazwy odprowadzeń i jednostki są na osiach; kanały nie muszą być równoważnymi odprowadzeniami.

Oś czasu wynika z indeksu próbki podzielonego przez częstotliwość próbkowania. Dwa kanały pokazują różne spojrzenia na tę samą aktywność elektryczną serca. Różnice między rekordami mogą wynikać z rytmu, odprowadzeń i zakłóceń; sam wykres nie stanowi diagnozy.

Argumenty `--data-dir` i `--output-dir` pozwalają zmienić lokalizacje. Zapis odmawia nadpisania istniejącego obrazu; przy ponownym uruchomieniu wybierz nowy katalog wynikowy. Backend Agg pozwala zapisywać obrazy bez interfejsu graficznego.

## Testy i ograniczenia

Testy obejmują lokalne 48 rekordów, eksperckie adnotacje rekordu 100, wymiary, niekompletne/uszkodzone pliki, błędne identyfikatory i katalogi, wykresy, zakres czasu oraz ochronę plików przed nadpisaniem. Testy integracyjne wymagają lokalnej bazy.

Na obecnym etapie zależności blokują uruchomienie testów i generowanie wykresów. Nie potwierdzono odczytu przez WFDB i nie wygenerowano obrazów; kontrola plików i nagłówka jest niezależna od WFDB. Nie podajemy wyników eksperymentalnych.

Baza, duże dane przetworzone, środowisko i wyniki obrazowe są wykluczone przez `.gitignore`. Fazy 4–12, w tym filtrowanie i format `[samples, channels, sequence_length]`, pozostają poza aktualnie zatwierdzonym zakresem.


## Aktualizacja — faza 4: filtrowanie ECG

Powyższe informacje o niedostępnych bibliotekach i braku filtrowania opisują wcześniejszy stan projektu. Środowisko `.venv` obecnie działa; fazy 1–3 zostały zweryfikowane (65 testów). Aktualny zakres obejmuje także filtrowanie, bez normalizacji i segmentacji. Nowe zależności opisuje `requirements_phase4.txt` (scipy oraz wcześniejsze biblioteki).

### Metody i interfejs

Moduł `src/preprocessing.py` udostępnia:

- `highpass_filter(values, sampling_frequency, cutoff_hz=0.5, order=4)`: filtr Butterwortha górnoprzepustowy tłumi powolne zmiany linii bazowej, w tym składową stałą.
- `lowpass_filter(values, sampling_frequency, cutoff_hz=45.0, order=4)`: filtr Butterwortha dolnoprzepustowy ogranicza szybkie zakłócenia; wygładza też część ostrych szczegółów sygnału.
- `notch_filter(values, sampling_frequency, frequency_hz=50.0, quality_factor=30.0)`: wąskopasmowy filtr zaporowy dla 50 lub 60 Hz. Większe Q oznacza węższe pasmo tłumienia, w przybliżeniu f/Q dla pojedynczego przejścia.
- `filter_ecg(...)`: domyślnie high-pass 0,5 Hz, następnie low-pass 45 Hz, oba rzędu 4. `notch_hz=None` domyślnie wyłącza notch. Jeśli zostanie jawnie włączony, działa PRZED pozostałymi filtrami. Każdy filtr Butterwortha można wyłączyć, przekazując `None` jako jego częstotliwość.
- `filter_recording(record, **kwargs)`: używa powyższej funkcji i zwraca nowy ECGRecording, zachowując częstotliwość próbkowania, ID, kanały, jednostki i oryginalne adnotacje.
- `powerline_diagnostics(...)`: pomocnicza ocena widma Welcha w pobliżu 50/60 Hz; nie włącza automatycznie notch.

```python
from src.data_loader import load_recording
from src.preprocessing import filter_recording, filter_ecg

raw = load_recording("100")
filtered = filter_recording(raw)  # notch wyłączony
single_channel = filter_ecg(raw.signals[:, 0], raw.sampling_frequency)
# Tylko po analizie zakłócenia można jawnie użyć:
# filtered = filter_recording(raw, notch_hz=60, quality_factor=30)
assert filtered.sampling_frequency == raw.sampling_frequency
assert filtered.signals.shape == raw.signals.shape
```

Wejście: `[samples]` albo `[samples, channels]`. Próbki zawsze leżą na osi 0. Wynik float64 ma identyczny kształt; wejście nie jest modyfikowane, nie ma resamplingu. Działanie na wielu kanałach jest niezależne dla każdego kanału. Dla danych o kolejności osi innej niż loadera należy najpierw zmienić kolejność osi.

Współczynniki Butterwortha powstają przez `scipy.signal.butter(..., output="sos")`. Notch korzysta z `iirnotch` i konwersji pojedynczego biquadu do SOS. Wszystkie etapy używają `sosfiltfilt`, czyli filtracji w przód i wstecz z dopełnieniem brzegów. To ogranicza przesunięcie fazy, ale jest **nieprzyczynowe**: korzysta z przyszłych próbek i nie nadaje się bezpośrednio do przetwarzania w czasie rzeczywistym. Dwa przejścia podwajają efektywny rząd charakterystyki amplitudowej; na nominalnej częstotliwości odcięcia Butterwortha tłumienie wynosi około 6 dB po obu przejściach, zamiast około 3 dB pojedynczego przejścia.

Parametry muszą być skończonymi liczbami, fs > 0, częstotliwości w (0, fs/2), high-pass < low-pass, rząd dodatni całkowity, Q > 0. NaN, nieskończoności, puste i nienumeryczne dane są odrzucane. Sygnał za krótki dla dopełnienia `sosfiltfilt` powoduje czytelny ValueError, bez cichego przejścia na filtr przyczynowy. Stały sygnał jest obsługiwany. Samo spełnienie minimalnej długości technicznej nie gwarantuje dobrej jakości bardzo krótkiego zapisu przy odcięciu 0,5 Hz.

### Czy notch jest potrzebny?

Low-pass 45 Hz tłumi również 50/60 Hz, choć nie usuwa ich całkowicie. Nie należy automatycznie dokładać notch. Przykład porównuje widma pełnych rekordów przed filtracją i po filtracji bez notch. Welch używa 4-sekundowych okien; porównujemy maksimum PSD w ±0,5 Hz z medianą tła oddaloną o 2–5 Hz. Wyniki są osobne dla kanałów. Wymagane są przynajmniej 4 sekundy danych i zapas pasma przy Nyquiście. Stosunek pik/tło jest heurystyką; może reagować również na aktywność fizjologiczną i zmienia się wraz z charakterystyką low-pass. Ważna jest także bezwzględna moc resztkowa, nie tylko sam stosunek.

W badanych pełnych rekordach pik PSD w okolicy 60 Hz po filtracji bez notch zmniejszył się o około 23,65–23,67 dB względem surowego zapisu. Względny pik nadal występuje (szczególnie w V1 rekordu 101); nie oznacza to, że zakłócenie całkowicie zniknęło. W okolicy 50 Hz stosunek pik/tło surowych nagrań wynosi około 1,02–1,29, więc nie ma wyraźnego wąskiego piku według tej heurystyki. Nie włączano notch na podstawie samego względnego piku 60 Hz: do oceny potrzeby dodatkowej filtracji trzeba uwzględnić pozostałą moc i wpływ na morfologię. Wybór pozostaje jawnie konfigurowalny; brak notch w przykładach nie jest dowodem jego zbędności dla wszystkich fragmentów i kanałów.

Dane liczbowe i parametry są w `results/preprocessing/phase4/powerline_diagnostics.json`. To diagnostyka przetwarzania sygnału, bez oceny modelu.

### Odtworzenie i obrazy

Z katalogu projektu:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m examples.filter_ecg
```

Skrypt korzysta z istniejącego loadera i filtruje pełne rekordy 100, 101, 200. Zapisuje w `results/preprocessing/phase4/` sześć PNG:

- `record_100_raw_filtered_10s.png` i `record_100_raw_filtered_detail.png`;
- `record_101_raw_filtered_10s.png` i `record_101_raw_filtered_detail.png`;
- `record_200_raw_filtered_10s.png` i `record_200_raw_filtered_detail.png`.

Każdy obraz pokazuje oba kanały na wspólnej osi czasu: surowy sygnał jest niebieski, filtrowany pomarańczowy, amplituda w mV. Pierwszy wariant pokazuje sekundy 10–20, szczegół sekundy 10–13. Filtracja następuje przed wycięciem fragmentu; nie filtrujemy osobno tych krótkich okien. Zapis odmawia nadpisania obrazów i JSON. Przy ponownym uruchomieniu użyj nowego katalogu, np. `--output-dir results/preprocessing/phase4_run2`. Obrazy można otworzyć w eksploratorze VS Code.

### Morfologia i ograniczenia

Obejrzano szczegółowe porównania wszystkich trzech rekordów. W MLII nadal rozpoznawalne są zespoły QRS oraz mniejsze wychylenia przed i po QRS, odpowiadające falom P i T; najlepiej widać je w przykładzie 101. W 100 i 200 ocena drobnych struktur bywa mniej jednoznaczna. Drugie kanały V1 w 101 i 200 pozostają trudniejsze do oceny z powodu niewielkiej amplitudy i zakłóceń. Filtracja nie gwarantuje widoczności wszystkich załamków w każdym kanale ani każdym uderzeniu.

Zmiana poziomu bazowego powoduje pionowe przesunięcie przebiegu. Low-pass wygładza szczegóły i może zmienić amplitudę szczytów; high-pass może zmienić wolniejsze składowe, w tym ST/T. Zachowanie widocznych P/QRS/T w kilku fragmentach nie dowodzi zachowania diagnostycznych amplitud, odstępów ani morfologii w całej bazie. Nie wykonywano automatycznej delineacji ani walidacji klinicznej. Odcięcie 0,5 Hz jest punktem wyjścia dla projektu, nie gwarancją przydatności do analizy ST. Nadmierne filtrowanie może usunąć informację istotną dla klasyfikacji.

Dopełnienie nie eliminuje artefaktów na początku i końcu nagrania; ich zakres zależy od charakterystyki filtru i sygnału, nie tylko padlen. Parametry należy utrzymywać i dokumentować spójnie. Filtry nie uczą się statystyk z innych pacjentów. Jeśli zapis został podzielony na niezależne części, nie należy filtrować przez granicę takiego podziału; przy przyszłej integracji segmentacja powinna używać odpowiedniego kontekstu z tego samego dozwolonego zapisu.

### Testy fazy 4

Dodano `tests/test_preprocessing.py`: tłumienie syntetycznej linii bazowej 0,1 Hz i zakłóceń 90 Hz, zachowanie 10 Hz, notch 50/60 Hz, brak przesunięcia fazy w paśmie, kształty dla jednego i wielu kanałów, brak mutacji wejścia, sygnały stałe i krótkie, parametry błędne, wartości niefinitywne, kolejność i wyłączanie filtrów, diagnostyka widmowa oraz pełne rekordy 100/101/200 z zachowaniem metadanych. Dane MIT-BIH nie są zapisywane ani modyfikowane. Nie zaimplementowano normalizacji, segmentacji ani zadań Anny.

Wynik końcowy `.venv/bin/python -m pytest`: **127 passed, 62 warnings in 23.11s** (65 wcześniejszych testów i 62 nowe). Ostrzeżenia dotyczą użycia przestarzałego argumentu pandas wewnątrz WFDB; nie wystąpiły błędy. Skrypt generowania sześciu PNG i raportu JSON zakończył się kodem 0. Kontrola składni nowych modułów przez compileall również przeszła.


## Aktualizacja — faza 5: normalizacja ECG

Aktualny zakres obejmuje wczytywanie, wykresy, filtrowanie i normalizację. Wcześniejsze fragmenty o braku normalizacji opisują stan z fazy 4. Nie implementujemy jeszcze segmentacji, CNN, treningu, walidacji ani Streamlit. Używamy istniejących filtrów bez zmiany ich parametrów domyślnych.

### Co oznacza normalizacja?

Normalizacja zmienia poziom i skalę amplitudy, aby kanały miały porównywalną skalę liczbową. Nie zmienia liczby próbek, ich kolejności ani czasu. Wynik jest bezwymiarowy: nie należy opisywać go jako mV ani używać bezpośrednio do pomiarów amplitud diagnostycznych. Oryginalne ECG i jednostki są zachowane w obiekcie źródłowym; funkcje zwracają nową tablicę float64 i nie modyfikują wejścia.

Dostępne metody w `src/preprocessing.py`:

- `zscore_normalize(values, ...)`: `(x - mean) / std`, gdzie std to odchylenie populacyjne (`ddof=0`). Dla zwykłego kanału dopasowane dane mają średnią około 0 i odchylenie około 1. Nie wymusza rozkładu normalnego ani ograniczonego przedziału.
- `minmax_normalize(values, feature_range=(0, 1), ...)`: `low + (x - min) / (max - min) * (high - low)`. Zakres docelowy jest konfigurowalny.
- `normalize_ecg(values, method="zscore", ...)`: wspólny interfejs obu metod.
- `ECGNormalizer`: jawne `fit`, `transform`, `to_dict`, `from_dict`, `save`, `load` dla statystyk wielokrotnego użytku.

Obsługiwane kształty to `[samples]` i `[samples, channels]`. Każdy kanał jest normalizowany niezależnie wzdłuż osi 0. Nie wolno bezpośrednio przekazywać przyszłego tensora CNN `[beats, channels, length]`; jego obsługa nie jest częścią fazy 5.

### Wybór domyślny do późniejszej integracji

Przyjęty punkt wyjścia to **Z-score per nagranie i per kanał, po filtracji i przed przyszłą segmentacją**. Całe nagranie otrzymuje jedną transformację na kanał; nie normalizujemy obecnie każdego uderzenia osobno. Z-score centruje sygnał wokół zera i skaluje go odchyleniem, zamiast wiązać zakres z samymi dwoma ekstremami. To uzasadnienie projektowe, nie wynik porównania modeli. Obie metody są wrażliwe na artefakty i obserwacje odstające; nie wykazano przewagi Z-score w klasyfikacji tego projektu. Zobacz [dokumentację skalowania i wpływu outlierów](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html). Nie dodano zależności scikit-learn.

Normalizacja niezależnych kanałów usuwa informację o ich bezwzględnej skali oraz stosunkach amplitud między kanałami; per-recording usuwa też część różnic amplitud pomiędzy nagraniami. Należy przekazać tę decyzję Annie. Parametry i tryb należy utrzymać identyczne w przygotowaniu wejścia i inferencji. Wybrany default jest przygotowany w API, ale segmentacja i eksport wejścia CNN nie zostały zaimplementowane.

### Zakres statystyk i ochrona przed leakage

`normalize_ecg` i funkcje pomocnicze domyślnie wyliczają statystyki wyłącznie z przekazanej tablicy **jednego nagrania** (`scope="recording"`). Przykłady i wykresy dopasowują 100, 101 i 200 osobno, bez łączenia ich statystyk. Nie tworzymy globalnego skalera z całej bazy i nie wyznaczamy podziału pacjentów.

`scope="segment"` jest dostępne, jeśli wywołujący świadomie przekazuje pojedynczy fragment; nie tworzy ani nie wykrywa segmentów. `scope` jest deklaracją zakresu i metadanymi, a nie funkcją sprawdzającą tożsamość pacjentów. Przy funkcjach pomocniczych warto jawnie podać `source_ids` zamiast domyślnego `("unspecified",)`.

Per-recording korzysta z całego dostępnego nagrania, również późniejszych próbek. Jest to procedura offline, wymagająca takiego samego dostępu w inferencji. W niezależnych nagraniach nie przenosi statystyk między pacjentami; nie jest jednak odpowiednikiem skalera dopasowanego tylko na treningu. Nie wolno wyliczać statystyk przez granicę czasowego podziału danych. Gdy inferencja ma działać na pojedynczym nowym oknie, należy świadomie wybrać inny, spójny tryb — np. zamrożone statystyki treningowe — zamiast udawać dostęp do całego przyszłego zapisu.

Opcjonalny wariant treningowy wymaga `ECGNormalizer.fit(..., scope="training", source_ids=...)`. Wywołujący potwierdza tym, że przekazane próbki należą wyłącznie do zbioru treningowego, już ustalonego przez Annę. ID rekordu nie jest automatycznie ID pacjenta. Moduł nie weryfikuje splitu i nie potrafi zapobiec podaniu błędnych źródeł. Podczas walidacji, testowania i inferencji wykonuje się **tylko transform** z zamrożonymi parametrami; nie należy ponownie wywoływać `fit` ani skrótu `normalize_ecg` w tym wariancie. Jest to zgodne z [zasadą uczenia transformacji wyłącznie na treningu](https://github.com/scikit-learn/scikit-learn/blob/main/doc/common_pitfalls.rst).

```python
from src.data_loader import load_recording
from src.preprocessing import filter_recording, normalize_ecg, ECGNormalizer

record = filter_recording(load_recording("100"))
x = normalize_ecg(record.signals, source_ids=[record.record_id],
                  channel_names=record.channel_names)  # default: per-record Z-score
assert x.shape == record.signals.shape

# Zachowanie parametrów pojedynczego nagrania:
scaler = ECGNormalizer("zscore").fit(
    record.signals, scope="recording", source_ids=[record.record_id],
    channel_names=record.channel_names,
)
scaler.save("results/preprocessing/record_100_scaler.json")  # nie nadpisuje pliku
restored = ECGNormalizer.load("results/preprocessing/record_100_scaler.json")
x_again = restored.transform(record.signals, channel_names=record.channel_names)
```

Szablon dla Anny, do użycia dopiero po ustaleniu zbioru treningowego (zmienne nie są tworzone w fazie 5):

```python
# training_samples: [samples, channels] wyłącznie z treningu,
# ta sama kolejność i znaczenie kanałów w każdym źródle.
scaler = ECGNormalizer("zscore").fit(
    training_samples, scope="training", source_ids=training_record_ids,
    channel_names=channel_names,
)
scaler.save("results/preprocessing/training_normalizer.json")
# W innym procesie / podczas inferencji, bez refit:
scaler = ECGNormalizer.load("results/preprocessing/training_normalizer.json")
x_new = scaler.transform(new_filtered_samples, channel_names=channel_names)
```

Fit przyjmuje jedną tablicę; nie implementuje akumulacji online ani zbierania wielu plików. Dużych zbiorów nie należy bezrefleksyjnie kopiować do pamięci. Statystyki są liczone z równą wagą każdej próbki, nie każdego pacjenta. JSON zawiera wersję formatu, metodę, epsilon, zakres, clipping, offset, skalę, zakres statystyk, ID źródeł, nazwy kanałów i liczbę próbek. Przy transform wymagane są te same nazwy/kolejność kanałów, jeśli podano je przy fit. Jeśli nie zapisano nazw, sprawdzana jest wyłącznie liczba kanałów — zgodność odprowadzeń musi zapewnić wywołujący. JSON nie zawiera ECG i nie zapisuje parametrów filtracji: filtrację należy konfigurować tak samo osobno.

### Przypadki brzegowe i polityka błędów

- Domyślny `epsilon=1e-8` jest bezwzględnym progiem w jednostkach wejścia (po filtracji MIT-BIH: mV), konfigurowalnym dla innych skal danych. Gdy std lub zakres jest <= epsilon, kanał daje stałe 0 w Z-score albo dolną granicę przedziału w Min-Max. Nie wzmacniamy tak drobnego szumu.
- Reguła dotyczy statystyk z fit: kanał uznany za stały pozostaje wyzerowany/dolny także przy transform nowych danych. Może to ukryć późniejszy sygnał w takim kanale; należy kontrolować te przypadki w integracji. To świadoma polityka, nie dokładna kopia zachowania sklearn.
- Sygnały jednoelementowe i stałe są poprawne. NaN/Inf są odrzucane czytelnym ValueError; nie imputujemy ani nie maskujemy braków w sygnale. Puste tablice, dane zespolone/nienumeryczne i nieprawidłowe wymiary są odrzucane.
- Sprawdzamy parametry oraz skończoność wyniku. Przepełnienie dla skrajnych liczb powoduje ValueError, zamiast zwrócenia NaN/Inf.
- Min-Max z zamrożonymi statystykami może dawać wartości poza zadanym zakresem dla nowych danych. Domyślne `clip=False` ich nie obcina. `clip=True` jest dostępne tylko dla Min-Max i może zniekształcić amplitudy. Z-score nowych danych nie musi mieć średniej 0 ani std 1.

### Wykresy i uruchomienie

```bash
cd ~/Pulpit/ECG-APP
.venv/bin/python -m pytest
.venv/bin/python -m examples.normalize_ecg
```

`examples/normalize_ecg.py` używa istniejącego loadera i `filter_recording` (0,5–45 Hz, rząd 4, bez notch). Normalizuje pełne nagrania osobno i pokazuje sekundy 10–13 każdego kanału w trzech kolumnach: po filtracji przed normalizacją (mV), Z-score, Min-Max (wyniki bezwymiarowe). Średnia i std pokazanego wycinka nie muszą wynosić 0 i 1: statystyki pochodzą z całego nagrania.

W `results/preprocessing/phase5/` zapisujemy:

- `record_100_normalization.png`, `record_101_normalization.png`, `record_200_normalization.png`;
- sześć plików `record_<ID>_<zscore|minmax>_parameters.json` — statystyki **per-recording**, nie parametry treningowe;
- `normalization_summary.json` — statystyki wyników, konfiguracja i wynik sprawdzenia odtworzenia parametrów.

Pliki otworzysz kliknięciem w eksploratorze VS Code. Przy ponownym generowaniu podaj nowy katalog, np. `--output-dir results/preprocessing/phase5_run2`. Nie nadpisujemy istniejących obrazów ani JSON. Baza pozostaje bez zmian, a wyniki są wykluczone z Git istniejącą regułą `results/preprocessing/`.

### Testy normalizacji

`tests/test_normalization.py` sprawdza znane wzory, niezależność kanałów, kształty i brak mutacji, sygnały stałe i niemal stałe, konfigurowalny zakres i epsilon, NaN/Inf i błędne parametry, brak refit podczas transform, stałe parametry treningowe, clipping, zgodność kanałów, walidację JSON i dokładne odtworzenie wyników po zapisie/odczycie. Obie metody sprawdzono także na pełnych przefiltrowanych rekordach 100, 101 i 200.


Weryfikacja końcowa fazy 5: `.venv/bin/python -m pytest` — **186 passed, 68 warnings in 28.87s** (127 wcześniejszych i 59 nowych). Ostrzeżenia pochodzą z przestarzałego użycia pandas wewnątrz WFDB; nie wystąpiły błędy. Kontrola składni przez compileall przeszła. Skrypt wygenerował trzy PNG, sześć JSON z parametrami i raport JSON, kończąc pracę kodem 0.

Dla obu kanałów każdego z trzech pełnych nagrań Z-score dał średnie bliskie 0 (bezwzględnie poniżej 2e-17) i std bliskie 1; Min-Max dał minimum 0 i maksimum 1. Odtworzenie parametrów JSON dało dokładnie identyczne wyniki transform dla wszystkich sześciu przypadków. Obrazy obejrzano: przebieg w czasie pozostaje taki sam, zmieniają się przesunięcie i skala amplitudy. W Min-Max lokalne fragmenty bywają skupione w wąskim zakresie przez ekstrema pełnego zapisu; nie utożsamiamy ich automatycznie z artefaktami. Nie są to wyniki klasyfikacji ani ocena kliniczna. Baza MIT-BIH pozostała bez zmian; prace zatrzymano po fazie 5.


## Faza 6 — zsynchronizowane porównanie preprocessingu

Dodano `examples/compare_preprocessing.py`, moduł wizualizacji `src/preprocessing_comparison.py` i testy `tests/test_preprocessing_comparison.py`. Wcześniejsze funkcje loadera, filtracji i normalizacji pozostają bez zmian; nie powielono ich implementacji. Nie wprowadzono segmentacji uderzeń ani mapowania klas.

### Procedura i odtworzenie

```bash
cd ~/Pulpit/ECG-APP
.venv/bin/python -m examples.compare_preprocessing
.venv/bin/python -m pytest
```

Skrypt odczytuje lokalne rekordy 100, 101 i 200. Używa istniejącego `filter_recording()` z ustawieniami domyślnymi: Butterworth high-pass 0,5 Hz i low-pass 45 Hz, rząd 4, SOS, zero-phase, notch wyłączony. Następnie wywołuje istniejące `normalize_ecg()` z domyślnym Z-score (epsilon 1e-8), per nagranie i per kanał. Statystyki nie są współdzielone pomiędzy nagraniami i nie powstaje globalny skaler.

Całe nagranie jest filtrowane i normalizowane PRZED wybraniem odcinka do wyświetlenia. Domyślnie pokazujemy sekundy 10–20: powtarzalny, ilustracyjny fragment wewnątrz zapisu, oddalony od początku. Wybór nie był optymalizowany według etykiet i nie jest dowodem reprezentatywności całego rekordu ani wszystkich arytmii. To wyłącznie przedział do wizualizacji, nie segmentacja uderzeń. Można zmienić przedział i katalog:

```bash
.venv/bin/python -m examples.compare_preprocessing \
  --start-seconds 30 --duration-seconds 10 \
  --output-dir results/preprocessing/phase6_run2
```

Opcja `--data-dir` wskazuje inną lokalną lokalizację bazy. Dla ponownego uruchomienia wybierz nowy katalog wynikowy; istniejące PNG i CSV nie są nadpisywane. Żądany przedział musi być dostępny w całości; nie skracamy go po cichu na końcu nagrania.

### Obrazy

W `results/preprocessing/phase6/` zapisujemy trzy obrazy PNG:

- `record_100_preprocessing_comparison.png`;
- `record_101_preprocessing_comparison.png`;
- `record_200_preprocessing_comparison.png`.

Każdy obraz ma trzy kolumny: surowe ECG, filtrowane ECG, filtrowane i znormalizowane Z-score ECG. Wiersze odpowiadają wszystkim dostępnym kanałom (dla tych nagrań: dwóm). Nazwy odprowadzeń i jednostki są na osiach Y; etapy nad panelami, a fs w tytule. Dla danych surowych i filtrowanych używamy jednostek z nagłówka (tu mV), a wynik Z-score jest bezwymiarowy. Nie wymuszamy wspólnego zakresu amplitud pomiędzy mV i skalą bezwymiarową.

Wszystkie panele mają dokładnie tę samą oś czasu: oryginalny indeks próbki / fs. Dla 360 Hz zakres [10, 20) s zawiera 3600 próbek o indeksach 3600–7199; prawa krawędź osi jest opisana jako 20 s. Nie ma resamplingu ani przesuwania tablic. Obrazy 18×8 cali zapisujemy w 300 dpi (5400×2400 pikseli dla dwóch kanałów). Otworzysz je kliknięciem w eksploratorze VS Code.

Czerwone punkty i oryginalne symbole oznaczają wyłącznie adnotowane uderzenia według istniejącej listy `BEAT_SYMBOLS`. Nie wyznaczamy nowych szczytów i nie przesuwamy markerów do lokalnego maksimum. Każdy panel używa tych samych oryginalnych pozycji; wysokość punktu pochodzi z danego etapu i kanału. Pozycje na lewej krawędzi wchodzą do przedziału, na prawej nie. Znaczniki inne niż uderzenia, np. `+`, są w tym porównaniu pomijane; nie zostały zamienione na klasy.

### Tabela CSV

`preprocessing_statistics.csv` zawiera 18 wierszy: 3 nagrania × 2 kanały × 3 etapy. Średnia, odchylenie populacyjne (`ddof=0`), minimum i maksimum odnoszą się **dokładnie do widocznego fragmentu**, a nie całego nagrania. Tabela zapisuje także ID nagrania, indeks i nazwę kanału, etap, jednostkę, fs, początek i koniec przedziału (koniec wyłączny), liczbę próbek oraz zakres statystyk i normalizacji.

To rozróżnienie jest istotne: Z-score był dopasowany do pełnego nagrania, więc sam wycinek może mieć średnią różną od 0 i std różne od 1. Nie są to błędy normalizacji. Tabela nie zawiera metryk klasyfikacji ani oceny modelu. Nie wyznaczamy podziału trening/walidacja/test ani ID pacjentów.

### Jak czytać zmiany

High-pass ogranicza powolną składową bazową; może przesunąć poziom przebiegu i zmienić wolne składowe ECG. Low-pass wygładza szybkie oscylacje, ale może także zmienić amplitudy ostrych szczegółów. Filtracja nie jest gwarancją zachowania wszystkich parametrów morfologii. Filtrowanie w przód i wstecz ogranicza przesunięcie fazy i działa offline; efekty brzegowe nadal istnieją, dlatego nie filtrujemy krótkiego fragmentu osobno.

Z-score zmienia przesunięcie i skalę amplitudy w każdym kanale. Dla kanału nieuznanego za stały jest to rosnąca transformacja liniowa: zachowuje indeksy ekstremów filtrowanego sygnału i kolejność próbek. Nie usuwa kolejnych zakłóceń i nie przywraca jednostek mV. Dla kanałów stałych/niemal stałych obowiązuje udokumentowana wcześniej reguła zerowania.

Indeksy czasu i pozycje adnotacji są zgodne między etapami. Nie oznacza to, że każde lokalne maksimum po filtracji musi wypaść na identycznej próbce: wygładzenie może zmienić lokalną morfologię i położenie maksimum. Nie wykonywano detekcji R ani ilościowej oceny przesunięć szczytów w rzeczywistych nagraniach; znaczniki są niezmienionymi punktami odniesienia ekspertów, nie wynikiem detektora.

### Testy fazy 6

Nowe testy sprawdzają zgodność z dotychczasowymi filtrami i normalizacją, kształty i brak mutacji, identyczne osie czasu wszystkich paneli, niezmienione współrzędne adnotacji, granice przedziału, pomijanie nieuderzeń, brak adnotacji, jeden/dwa kanały, etykiety i jednostki, zapis PNG w 300 dpi, odmowę nadpisania, poprawność wartości CSV oraz odczyt i porównanie rekordów 100/101/200. Test zachowania maksimum dotyczy etapu normalizacji na niezmiennym syntetycznym sygnale, nie twierdzenia o doskonałym zachowaniu maksimów przez filtr.


Weryfikacja fazy 6: `.venv/bin/python -m pytest` — **203 passed, 71 warnings in 36.33s** (186 wcześniejszych i 17 nowych). Ostrzeżenia pochodzą z pandas używanego wewnątrz WFDB; nie wystąpiły błędy. Kontrola składni nowych plików przez compileall przeszła. Skrypt porównania zakończył się kodem 0, zapisując trzy PNG i CSV z 18 wierszami.

Obejrzano wszystkie trzy wygenerowane porównania. W MLII rekordu 100 średnia widocznego fragmentu zmieniła się z około -0,3184 mV do -0,00147 mV po filtracji, ilustrując zmianę poziomu bazowego. W drugim kanale rekordu 101 nadal widoczne są oscylacje mimo wygładzenia; nie interpretujemy filtracji jako usunięcia całego szumu. W rekordzie 200 różne oryginalne symbole uderzeń pozostają w tych samych współrzędnych czasu we wszystkich etapach. Normalizacja przeskalowała amplitudy, zachowując czasowy przebieg filtrowanego sygnału. Są to obserwacje ilustracyjne, bez walidacji klinicznej. Nie zmieniono bazy ani domyślnych parametrów filtracji i normalizacji; prace zatrzymano po fazie 6.


## Faza 7 — segmentacja uderzeń według eksperckich adnotacji

Dodano `src/segmentation.py`, `examples/segment_ecg.py` i `tests/test_segmentation.py`. Nie zmieniono loadera, filtrów, normalizacji ani danych MIT-BIH. Nie szukamy nowych pików R: punktem odniesienia jest zawsze pozycja eksperta odczytana przez istniejący loader, używający `wfdb.rdann(..., "atr")`.

### Dokładna definicja okna

Dla fs i żądanych czasów `before_seconds` i `after_seconds`:

```text
before_samples = floor(before_seconds * fs + 0.5)
after_samples  = floor(after_seconds  * fs + 0.5)
start = annotation_position - before_samples
stop  = annotation_position + after_samples
window = signal[start:stop]  # stop jest wyłączny
length = before_samples + after_samples
```

Każdy czas jest zaokrąglany osobno do najbliższej próbki, z połówkami w górę. Wymagamy dodatnich czasów dających co najmniej jedną próbkę po każdej stronie. Przy 360 Hz, 0,25 s i 0,45 s otrzymujemy 90 + 162 = **252 próbki**, bez dodatkowego `+1`. Adnotacja wypada zawsze pod indeksem 90 (indeksowanie od zera), nie w geometrycznym środku asymetrycznego okna. Ostatnia próbka ma czas względny 161/360 s, a prawa granica wyłączna 0,45 s. Przy innych fs rzeczywiste czasy po zaokrągleniu mogą minimalnie różnić się od żądanych; liczby próbek i fs są zachowane w konfiguracji.

Jeśli start < 0 albo stop > liczba próbek nagrania, pomijamy całe okno. Dopuszczamy start == 0 i stop == długość nagrania. Nie dopełniamy zerami, nie skracamy okien i nie zmieniamy punktu adnotacji. Raport zawiera łączną liczbę pominięć brzegowych i osobno naruszenia lewej oraz prawej granicy. Dla zapisu krótszego od całego okna obie przyczyny mogą dotyczyć tej samej adnotacji — w liczbie łącznej liczymy ją raz.

### Wybór adnotacji

Używamy istniejącej listy `BEAT_SYMBOLS`: N, L, R, B, A, a, J, S, V, r, F, e, j, n, E, /, f, Q. Opisy symboli sprawdzono w tabeli adnotacji zainstalowanej biblioteki WFDB. Zachowujemy oryginalne symbole bez mapowania do klas AAMI i bez tworzenia etykiet dla nieadnotowanego ECG. Znaczniki rytmu `+`, jakości `~`, artefaktu `|`, fale P/T, pacer spike bez uderzenia i inne nieuderzenia są wykluczane. Lista nie jest schematem klas i może wymagać jawnego rozszerzenia dla innych baz/customowych adnotacji.

Błędne pozycje (niecałkowite, ujemne, poza nagraniem, nieuporządkowane), niespójne metadane i NaN/Inf powodują ValueError. Puste adnotacje i brak prawidłowych uderzeń dają poprawną pustą tablicę `[0, channels, length]`. Powtórzone pozycje zachowujemy w oryginalnej kolejności wraz z indeksami adnotacji, bez automatycznego usuwania duplikatów.

### API i struktura dla Anny

```python
from src.data_loader import load_recording
from src.segmentation import segment_recording, segment_heartbeats

record = load_recording("100")
beats = segment_recording(record)  # pełna filtracja, Z-score per recording, potem okna
print(beats.X.shape)              # (2271, 2, 252)
print(beats.annotation_symbols)   # oryginalne symbole, jeszcze bez klas AAMI
print(beats.report)
```

`segment_heartbeats(...)` wycina okna z dostarczonej tablicy `[samples]` albo `[samples, channels]` i NIE filtruje/normalizuje ponownie. Należy podać zgodne nazwy kanałów i jednostki; bez nich zapisujemy ogólne nazwy i jednostkę `unspecified`. `segment_recording(...)` jest wygodnym wrapperem dla istniejącego ECGRecording: przetwarza pełny zapis przez `filter_recording` oraz `ECGNormalizer`, dopiero potem wycina okna. Obsługuje także rekord jednokanałowy. Parametry okna są konfigurowalne; `filter_kwargs` pozwala świadomie użyć istniejących opcji filtrowania.

Wynik to `SegmentedHeartbeats` w pamięci:

- `X`: float64, `[number_of_beats, number_of_channels, sequence_length]`, dla obecnego wrappera bezwymiarowe;
- `record_ids`, `annotation_positions`, `annotation_indices`, `annotation_symbols`, `annotation_aux_notes`: jeden element na okno, w tej samej kolejności;
- `patient_group_ids`: jawnie przekazana grupa (`patient_group_id=...`) albo pusty string oznaczający **nieznaną grupę**, nigdy automatycznie ID nagrania;
- `sampling_frequency`, `channel_names`, `units`, `before_samples`, `after_samples`;
- `report`: liczby ekstrakcji, nieuderzeń i pominięć brzegowych, liczby wykluczonych symboli oraz oryginalne indeksy pominiętych adnotacji;
- `configuration`: czasy żądane, rzeczywiste liczby próbek, sposób zaokrąglania, polityka granic, lista symboli uderzeń, pochodzenie grupy pacjenta, parametry filtracji i dopasowane statystyki normalizacji.

`annotation_positions` to oryginalne indeksy PRÓBEK w nagraniu; `annotation_indices` to indeksy w oryginalnej tablicy ADNOTACJI. Anna może użyć record_ids do późniejszego dołączenia wiarygodnej mapy pacjentów. Same record_ids nie gwarantują rozłączności pacjentów pomiędzy nagraniami; niewypełnionych grup nie wolno traktować jako gotowego podziału. Nie ustalaliśmy grup pacjentów ani splitów.

Nie dodano `y`, grupowania AAMI, balansowania, augmentacji ani finalnego eksportu datasetu. Przyszłe użycie PyTorch będzie wymagało jawnej konwersji dtype na uzgodniony typ (np. float32); tu nie dodajemy zależności PyTorch.

### Normalizacja i odtwarzalność

Domyślnie wrapper używa istniejących ustawień: Butterworth 0,5–45 Hz, rząd 4, zero-phase, bez notch; Z-score per kanał pełnego nagrania, epsilon 1e-8. Nie liczymy wspólnych statystyk dla 100, 101 i 200 i nie normalizujemy każdego uderzenia osobno. Konfiguracja zachowuje dokładne statystyki i metadane użytego normalizatora.

Per-recording korzysta z całego nagrania, w tym przyszłych próbek; jest metodą offline. Inferencja musi mieć ten sam kontekst lub używać świadomie ustalonego innego wariantu. Nie wolno filtrować ani wyliczać statystyk przez granicę niezależnych części czasowych. Okna mogą zachodzić na siebie i zawierać części sąsiednich uderzeń; to nie jest detekcja granic morfologicznych. Przyszłe podziały nie mogą losowo rozdzielać okien tego samego pacjenta.

Opcjonalnie `segment_recording(record, normalizer=fitted_scaler)` stosuje WYŁĄCZNIE `transform` dostarczonego skalera, bez refit. Anna może przekazać wcześniej zapisany i wczytany ECGNormalizer dopasowany wyłącznie na jej zbiorze treningowym. Zgodność kanałów i konfiguracji musi zostać zachowana. Wrapper nie sprawdza tożsamości pacjentów ani uprawnień do użycia źródeł skalera. Testy potwierdzają, że przekazane statystyki pozostają niezmienione.

Filtracja offline nadal może powodować efekty brzegowe. Pominięcie niepełnych okien nie jest gwarancją braku takich efektów w kompletnych oknach blisko początku/końca. Krótkie zapisy mogą zostać odrzucone przez istniejący filtr przed segmentacją; funkcja niskopoziomowa nie wymaga filtracji i poprawnie raportuje brak kompletnych okien.

### Uruchomienie i wyniki lokalne

```bash
cd ~/Pulpit/ECG-APP
.venv/bin/python -m examples.segment_ecg
.venv/bin/python -m pytest
```

Skrypt przetwarza po jednym pełnym nagraniu na raz i nie zapisuje macierzy X do plików. Dla dużej bazy nie należy niepotrzebnie gromadzić wyników wszystkich nagrań w RAM. Parametry CLI: `--data-dir`, `--output-dir`, `--before-seconds`, `--after-seconds`. Powtórne generowanie wymaga nowego katalogu, np. `--output-dir results/preprocessing/phase7_run2`: nie nadpisujemy istniejących obrazów ani JSON.

| Rekord | Wszystkie adnotacje | Wycięte uderzenia | Nieuderzenia | Niepełne okna | Lewa / prawa granica |
|---|---:|---:|---:|---:|---:|
| 100 | 2274 | 2271 | 1 | 2 | 1 / 1 |
| 101 | 1874 | 1864 | 9 | 1 | 1 / 0 |
| 200 | 2792 | 2600 | 191 | 1 | 0 / 1 |

Powody wykluczenia nieuderzeń: 100 — `+`: 1; 101 — `+`: 1, `~`: 4, `|`: 4; 200 — `+`: 148, `~`: 43. Zatem łącznie pominięto odpowiednio 3, 10 i 192 adnotacje, z dwóch rozłącznych kategorii (nieuderzenia i niepełne okna). Szczegółowe indeksy są w raporcie.

`results/preprocessing/phase7/` zawiera dla każdego ID 100, 101, 200:

- `record_<ID>_raw_annotations.png`: oryginalne ECG obu kanałów w sekundach 10–20 z niezmienionymi pozycjami i symbolami eksperckimi (istniejący wykres pokazuje też szare nieuderzenia);
- `record_<ID>_heartbeat_segments.png`: trzy pierwsze kompletne uderzenia z tego przedziału, oba kanały, po filtracji i Z-score. Oś X to czas względem eksperckiej pozycji; czerwony punkt i linia przy 0 s wskazują tę adnotację. Numer oryginalnej próbki i symbol są w tytule panelu;
- `segmentation_summary.json`: diagnostyczne liczby, konfiguracja, kształty i pozycje pokazanych przykładów; **nie jest finalnym eksportem datasetu** i nie zawiera macierzy X.

Obrazy zapisano w 300 dpi. Można je otworzyć kliknięciem w eksploratorze VS Code. Są to ilustracje wycięcia według adnotacji, bez oceny klinicznej czy wyników modelu.


### Weryfikacja fazy 7

`.venv/bin/python -m pytest`: **239 passed, 74 warnings in 38.55s** — 203 wcześniejsze i 36 nowych testów. Ostrzeżenia dotyczą pandas wewnątrz WFDB; nie wystąpiły błędy. Testy obejmują długość i wyrównanie okien, granice dokładne i niepełne, zaokrąglanie przy innym fs, różne liczby kanałów, oryginalne symbole i notatki, ID źródła i opcjonalną grupę pacjenta, puste wyniki, odrzucenie NaN/Inf i błędnych danych, brak mutacji i refit skalera treningowego, konfigurację domyślną, współrzędne wykresów oraz pełne rekordy 100/101/200. Kontrola składni nowych plików przez compileall przeszła.

Skrypt przykładów zakończył się kodem 0, zapisując sześć PNG i diagnostyczny JSON. Obejrzano wykresy segmentów wszystkich trzech nagrań: pozycja 0 s odpowiada oryginalnej adnotacji. W drugim kanale nie zawsze jest to maksimum wychylenia; nie przesuwamy tego punktu. Nie wyciągamy z przykładów wniosków o poprawności klinicznej ani skuteczności przyszłego modelu. Nie zapisano macierzy datasetu i nie rozpoczęto fazy 8.


## Faza 8 — mapowanie oryginalnych symboli do pięciu klas AAMI-style

Dodano `src/label_mapping.py`, `examples/map_heartbeat_labels.py` i `tests/test_label_mapping.py`. Moduł korzysta z istniejącego `SegmentedHeartbeats`, `segment_recording` i całego poprzedniego preprocessingu bez powielania lub zmiany ich implementacji. Tablice `annotation_symbols` zawierają oryginalne, rozróżniające wielkość liter symbole WFDB; pozostają nienaruszone.

### Cel, konwencja i źródła

Grupowanie łączy szczegółowe rodzaje uderzeń w pięć wspólnych kategorii wejściowych do późniejszej klasyfikacji. Nie jest diagnozą rytmu ani oceną modelu. Definicje symboli sprawdzono w [oficjalnej tabeli PhysioNet](https://archive.physionet.org/physiobank/annotations.shtml), a przyporządkowanie w tabeli 1 pracy [Das i Ari, 2014, ECG Beats Classification Using Mixture of Features](https://onlinelibrary.wiley.com/doi/10.1155/2014/178436). Nazywamy je **AAMI-style**: nie przeprowadzono audytu pełnego, licencjonowanego tekstu normy EC57 ani nie deklarujemy zgodności całego protokołu ewaluacji z normą. W publikacjach występują różne zestawy rekordów, w tym wyłączenia nagrań stymulatorowych; tutaj żadne takie wyłączenia nie są wykonywane automatycznie.

| Klasa | Indeks | Oryginalne symbole | Znaczenie grupy |
|---|---:|---|---|
| N | 0 | N, L, R, e, j | Uderzenia normalne i powiązane: także bloki odnóg i pobudzenia zastępcze przedsionkowe/węzłowe |
| S | 1 | A, a, J, S | Pobudzenia nadkomorowe ektopowe |
| V | 2 | V, E | Pobudzenia komorowe, w tym zastępcze |
| F | 3 | F | Fuzja pobudzenia komorowego i normalnego |
| Q | 4 | /, f, Q | Stymulowane, fuzja stymulowanego i normalnego oraz nieklasyfikowalne |

Kolejność `CLASS_NAMES = ('N', 'S', 'V', 'F', 'Q')` jest stała. Indeksy są umową interfejsu projektu, nie kodami numerycznymi WFDB. N nie oznacza wyłącznie zdrowego serca, a Q nie oznacza dowolnego błędu lub nieznanego symbolu. `J` i `j`, `F` i `f`, `A` i `a` mają różne znaczenia. Normalizacja nie zmienia etykiet.

### Nieznane, nieobjęte mapą i nieuderzenia

`map_symbol(symbol)` i domyślne `map_segments(segments)` zatrzymują się z ValueError, jeśli symbol nie ma przyporządkowania. Nie ma automatycznego przypisania do N ani Q. `B`, `n`, `r` są rozpoznanymi symbolami uderzeń w fazie 7, ale nie należą do wskazanej przez projekt mapy domyślnej. Są jawnie nieprzyporządkowane. `?` (learning/unclassified during learning) nie jest oryginalnym `Q`; jest pomijany przez istniejącą selekcję fazy 7 i nie został tutaj dodany do Q. Ta polityka może różnić się od rozszerzonych map spotykanych w innych pracach.

`unknown_policy="mark"` zachowuje KAŻDY segment i oznacza brak przyporządkowania `y=-1`, `mapped_classes="UNMAPPED"`. To znacznik techniczny, **nie szósta klasa i nie poprawny target CNN**. Przed przyszłym użyciem Anna musi sprawdzić `mapped_mask` i świadomie rozstrzygnąć takie przypadki. Nie usuwamy ich automatycznie ani nie wykonujemy balansowania. Raport podaje `unmapped_symbol_counts` i jawne mianowniki procentów.

Nieuderzenia takie jak `+` (rytm), `~` (jakość) czy `|` (artefakt) są wykluczone wcześniej w segmentacji. Nie można dodać ich do własnej mapy jako uderzeń przez to API. Niepełne okna brzegowe także zostały już pominięte w fazie 7. Raporty zachowują te powody oddzielnie od symboli nieprzyporządkowanych podczas mapowania.

Konfiguracja: `mapping={symbol: class_name, ...}` w `map_symbol`, `map_segments` i `summarize_recordings` zastępuje całą domyślną mapę, a nie rozszerza jej po cichu. Klasy i ich indeksy pozostają stałe. Przykładowe świadome rozszerzenie to `custom = {**DEFAULT_MAPPING, "r": "V"}`; nie jest używane w naszych wynikach. Własna mapa jest walidowana, kopiowana i porządkowana deterministycznie; należy ją zachować wraz z danymi.

### Rzeczywiste liczebności wybranych nagrań

Poniższe liczby odnoszą się do kompletnych segmentów po fazie 7, bez balansowania:

| Rekord | N | S | V | F | Q | Razem segmentów | Unmapped |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | 2237 | 33 | 1 | 0 | 0 | 2271 | 0 |
| 101 | 1859 | 3 | 0 | 0 | 2 | 1864 | 0 |
| 200 | 1742 | 30 | 826 | 2 | 0 | 2600 | 0 |
| Łącznie wybrane | 5838 | 66 | 827 | 2 | 2 | 6735 | 0 |

| Rekord | N [%] | S [%] | V [%] | F [%] | Q [%] |
|---|---:|---:|---:|---:|---:|
| 100 | 98,5029 | 1,4531 | 0,0440 | 0 | 0 |
| 101 | 99,7318 | 0,1609 | 0 | 0 | 0,1073 |
| 200 | 67,0000 | 1,1538 | 31,7692 | 0,0769 | 0 |

Procenty są udziałem w przyporządkowanych segmentach; dla tych nagrań jest to również udział we wszystkich segmentach, ponieważ nie ma unmapped. Dane nie są estymacją częstości chorób w populacji. Nie liczymy wag klas ani współczynników do balansowania.

Oryginalne symbole kompletnych segmentów: 100 — N=2237, A=33, V=1; 101 — N=1859, A=3, Q=2; 200 — N=1742, A=30, V=826, F=2. Szczegółowe liczby wszystkich surowych adnotacji są osobno w `original_symbol_counts.csv` i JSON; nie należy utożsamiać ich z liczbą gotowych okien.

Zgodność z fazą 7:

- 100: 2274 adnotacje = 2271 segmentów + 1 nieuderzenie (`+`) + 2 okna brzegowe;
- 101: 1874 = 1864 + 9 nieuderzeń (`+`:1, `~`:4, `|`:4) + 1 okno brzegowe;
- 200: 2792 = 2600 + 191 nieuderzeń (`+`:148, `~`:43) + 1 okno brzegowe.

Mapowanie nie odrzuciło żadnego segmentu w tych trzech nagraniach. Suma klas zgadza się dokładnie z wyjściem segmentacji.

### Struktura gotowa do dalszej pracy Anny

```python
from src.data_loader import load_recording
from src.segmentation import segment_recording
from src.label_mapping import map_segments, class_distribution

segments = segment_recording(load_recording("100"))
labeled = map_segments(segments)  # default: błąd przy symbolu spoza mapy
print(labeled.X.shape)           # (2271, 2, 252), to samo X co w segments
print(labeled.y.shape)           # (2271,), int64, klasy 0..4
print(labeled.class_names)       # ('N', 'S', 'V', 'F', 'Q')
print(labeled.mapped_classes)    # oryginalne segmenty dostają etykiety grup
print(labeled.segments.annotation_symbols)
print(class_distribution(labeled))
```

`LabeledHeartbeats` jest wrapperem zawierającym dotychczasowy `SegmentedHeartbeats` (bez kopiowania dużej tablicy X), `y`, `mapped_classes`, `class_names`, wybraną mapę i politykę unknown. `mapped_mask` informuje o przyporządkowaniu. Dla każdego indeksu i zachowane są bez zmian w `.segments`: oryginalny symbol, ID nagrania, pozycja próbki, indeks adnotacji, notatka pomocnicza, dostępna grupa pacjenta, nazwy i jednostki kanałów, fs, raport i konfiguracja preprocessingu. Nic nie jest sortowane ani przestawiane względem X. Nie dodano zależności PyTorch, konwersji X do float32 ani finalnego eksportu datasetu.

Grupy pacjentów nadal są **nierozwiązane** (pusty string, zgodnie z fazą 7); raport podaje liczbę takich segmentów. Nie tworzymy nowych ID pacjentów. Same ID nagrań nie gwarantują rozłączności pacjentów — oficjalna [dokumentacja MIT-BIH](https://physionet.org/content/mitdb/1.0.0/) opisuje 48 zapisów 47 osób i wspólnego pacjenta rekordów 201/202. Zachowujemy informacje dla Anny, ale nie implementujemy jej mapy pacjentów ani walidacji.

### Zestawienie wszystkich dostępnych rekordów

```python
from src.label_mapping import summarize_recordings
summary = summarize_recordings()  # domyślnie wszystkie kompletne lokalne rekordy
# Alternatywnie ograniczony zakres:
summary = summarize_recordings(record_ids=["100", "101", "200"])
```

Funkcja przetwarza po jednym nagraniu, używa rzeczywistej istniejącej segmentacji i zwraca małe zestawienia per nagranie oraz sumę. Nie łączy macierzy ECG w RAM i niczego nie eksportuje. Raportuje wszystkie symbole źródłowych adnotacji osobno od symboli segmentów i unknown. Powtarzające się ID na liście są odrzucane, aby zapobiec podwójnemu liczeniu. Wyjątki przy niepoprawnych danych nie są ignorowane. Obecne zapisane wyniki dotyczą wyłącznie 100, 101, 200; nie deklarujemy wykonania pełnego zestawienia wszystkich 48 nagrań.

Procenty `class_percentages_of_mapped` mają mianownik mapped_count; `class_percentages_of_segments` — wszystkie segmenty, w tym unmapped. Suma tych drugich może być <100%, gdy istnieją symbole nieprzyporządkowane. Dla pustych danych procenty i liczebności wynoszą 0; kolejność i obecność wszystkich pięciu klas pozostaje stała.

### Odtworzenie i pliki wynikowe

```bash
cd ~/Pulpit/ECG-APP
.venv/bin/python -m examples.map_heartbeat_labels
.venv/bin/python -m pytest
```

W `results/preprocessing/phase8/`:

- `original_symbols.png`: symbole wszystkich adnotacji i kompletnych segmentów osobno dla każdego nagrania;
- `aami_class_counts.png`: liczby pięciu klas per nagranie i łącznie;
- `available_class_examples.png`: rzeczywisty segment każdej obecnej klasy, oba kanały, oryginalny symbol, ID i pozycja; tu wszystkie pięć klas występuje łącznie (Q pochodzi z 101, F z 200);
- `class_distribution.csv`: 20 wierszy (trzy nagrania i suma × pięć klas), liczby i procenty z jawnymi mianownikami;
- `original_symbol_counts.csv`: wszystkie symbole źródłowe, liczba adnotacji i segmentów oraz liczba wykluczeń przed mapowaniem;
- `label_mapping_summary.json`: mapa, indeksy, liczebności, segmentacyjne powody wykluczeń, unresolved groups i metadane przykładów; nie zawiera X i nie jest finalnym eksportem datasetu.

Obrazy zapisujemy w 300 dpi. Wybór przykładu jest deterministyczny: pierwszy dostępny kompletny segment klasy w kolejności rekordów 100, 101, 200. Nie wyszukujemy najładniejszych przebiegów; Q może przedstawiać silnie zaburzony sygnał. Jeśli klasa nie występuje, nie tworzymy sztucznego przykładu — skrypt odnotowuje brak. Otwórz pliki w eksploratorze VS Code. Powtórne generowanie wymaga nowego katalogu, np. `--output-dir results/preprocessing/phase8_run2`. Opcja `--data-dir` wskazuje lokalną bazę. Zapis nie nadpisuje istniejących wyników.

### Ograniczenia

To grupowanie uderzeń, nie rozpoznawanie AF ani diagnoza pacjenta. N i Q są heterogeniczne; scalenie symboli traci szczegółowość, dlatego zachowujemy oryginały. Liczby zależą od wersji adnotacji, wyboru rekordów i okien brzegowych. Statystyki normalizacji pozostają per nagranie, bez łączenia walidacyjnych/testowych pacjentów w globalnym skalerze; ograniczenia offline i późniejszej inferencji z fazy 5/7 nadal obowiązują. Brak finalnego eksportu, balansowania, augmentacji, modelu, treningu, metryk i walidacji pacjentów. Prace kończymy na fazie 8.


### Weryfikacja fazy 8

Końcowe `.venv/bin/python -m pytest`: **283 passed, 77 warnings in 68.46s** (239 wcześniejszych i 44 nowe testy). Ostrzeżenia pochodzą z pandas wewnątrz WFDB; nie wystąpiły błędy. Nowe testy weryfikują każdy z 15 symboli domyślnej mapy, indeksy, deterministyczność, nieznane symbole i sentinel, własne mapy, zachowanie metadanych i kolejności, unresolved patient groups, liczby i oba mianowniki procentów, puste wejścia, błędne metadane, agregację bez podwójnego liczenia oraz prawdziwą segmentację rekordów 100/101/200.

Kontrola składni przez compileall przeszła. Skrypt przykładów zakończył się kodem 0 i wygenerował trzy PNG, dwie tabele CSV i diagnostyczny JSON. Wszystkie trzy obrazy obejrzano. Łącznie jest 6735 kompletnych segmentów i 6735 przyporządkowanych etykiet; nie wystąpiły symbole nieprzyporządkowane. Przykłady wszystkich pięciu klas pochodzą z rzeczywistych adnotowanych okien. Nie zmieniono plików MIT-BIH ani poprzednich modułów preprocessingu. Nie rozpoczęto fazy 9.


## Faza 9 — niezrównoważenie klas i opcjonalne narzędzia tylko dla treningu

Dodano `src/class_imbalance.py`, `examples/analyze_class_imbalance.py` i `tests/test_class_imbalance.py`. Używamy istniejących loadera, filtrów, normalizacji, segmentacji i domyślnej mapy z fazy 8 bez zmian. Nowe funkcje analizy przyjmują `LabeledHeartbeats`; nie modyfikują jego ECG, etykiet ani metadanych.

### Pełna analiza rzeczywistych danych lokalnych

Przetworzono **wszystkie 48 kompletnych lokalnych nagrań** — także nagrania stymulatorowe, bez dodatkowych wyłączeń. Każdy rekord został rzeczywiście wczytany, przefiltrowany i znormalizowany dotychczasowymi ustawieniami, podzielony na okna 252 próbek i zmapowany do dotychczasowych pięciu klas. Nie jest to zestawienie samych surowych adnotacji ani wynik symulacji. Nagrania przetwarzano pojedynczo, bez łączenia całej macierzy X w RAM.

| Klasa | Liczba kompletnych segmentów | Udział [%] |
|---|---:|---:|
| N | 90594 | 82,77053 |
| S | 2781 | 2,54084 |
| V | 7235 | 6,61020 |
| F | 802 | 0,73274 |
| Q | 8040 | 7,34569 |
| Razem | 109452 | 100 |

Klasa większościowa: **N**. Najmniejsza niepusta klasa: **F**. Iloraz N/F = **112,96009975**, czyli około 113:1. Nie ma klas pustych w całej bazie. Descriptory `majority_classes` i `minority_nonempty_classes` zwracają listy, aby poprawnie obsługiwać remisy.

Domyślny, konfigurowalny znacznik `rare_count_threshold=1000` oznacza **0 < liczba < 1000**, osobno dla analizowanej całości lub pojedynczego rekordu. Globalnie spełnia go F. Nie jest to próg kliniczny ani reguła doboru balansowania. Klasy puste są raportowane osobno, nie wchodzą do mianownika majority/minority. Dla pustego wejścia iloraz wynosi `None`, a procenty 0; przy jednej niepustej klasie iloraz wynosi 1. Procenty analizy dotyczą wyłącznie zmapowanych segmentów. Nieprzyporządkowane etykiety -1 są zliczane osobno, nigdy jako N/Q.

Globalny audyt liczebności:

```text
112647 wszystkich adnotacji
= 109452 kompletnych segmentów
+ 3153 znaczników niebędących uderzeniami
+ 42 niepełne okna brzegowe

109452 kompletnych segmentów = 109452 etykiet N/S/V/F/Q + 0 unmapped
```

Sprawdzono ten związek w każdym nagraniu oraz zgodność liczebności po indeksach całkowitych z istniejącym raportem mapowania. Globalne liczby oryginalnych symboli w segmentach: `/`=7025, A=2546, E=106, F=802, J=83, L=8072, N=75022, Q=33, R=7255, S=2, V=7129, a=150, e=16, f=982, j=229. Q jako grupa obejmuje więc 8007 uderzeń `/` lub `f` i tylko 33 oryginalne Q. Nie wolno utożsamiać całej grupy Q wyłącznie z nieklasyfikowalnymi uderzeniami lub zakłóceniami.

### Różnice pomiędzy nagraniami

`recording_class_distribution.csv` zawiera wszystkie 48 rekordów, pięć liczebności i pięć procentów na rekord, klasy rzadkie i brakujące, większość/mniejszość oraz iloraz dla niepustych klas. Nie zakładamy, że rekord jest pacjentem.

Klasy występują w następującej liczbie nagrań: N — 47, S — 32, V — 37, F — 17, Q — 9. W **46 z 48 rekordów** brakuje przynajmniej jednej klasy. Dla F aż 734 z 802 okien pochodzi z rekordów 208 i 213; taka koncentracja nie oznacza dużej różnorodności pacjentów i nie jest rozwiązywana samym powtarzaniem próbek. Lista nagrań zawierających rzadkie klasy oraz lista brakujących klas są w JSON i tabeli per rekord.

Wszystkie 109452 grup pacjentów pozostają nierozwiązane tak jak wcześniej. Zachowujemy oryginalne ID nagrań, a `recording_patient_groups.csv` oznacza grupy jako unresolved, bez wymyślania identyfikatorów. Raporty per rekord zachowują konfigurację preprocessingu i segmentacyjne powody wykluczeń; istniejące `LabeledHeartbeats` zachowuje pełne pozycje i metadane źródłowe. Same ID rekordów nie gwarantują rozłączności pacjentów. Anna musi najpierw ustalić wiarygodne grupy pacjentów, w tym powiązanie rekordów 201/202 opisane w dokumentacji MIT-BIH.

### Dlaczego imbalance ma znaczenie?

Duża liczba uderzeń N może zdominować uczenie, podczas gdy pomyłki na rzadkich S/F pozostaną słabo widoczne w wyniku zbiorczym. Accuracy liczy poprawne decyzje bez uwzględnienia tego, że większość przykładów może należeć do jednej klasy: wysoka wartość nie gwarantuje rozpoznawania mniejszości. Nie implementowano ani nie obliczano tutaj Accuracy, F1, AUPRC ani innych metryk modelu. Ich dobór i późniejsza ocena należą do Anny. Globalne częstości bazy nie są estymacją częstości arytmii w populacji klinicznej.

### Możliwe strategie — analiza, nie zastosowany pipeline treningowy

| Strategia | Zalety | Wady i ryzyka dla ECG |
|---|---|---|
| Wagi klas | Zachowują wszystkie próbki; nie powiększają macierzy ECG | Wzmacniają także błędne/zakłócone etykiety rzadkich klas; duże wagi mogą utrudniać uczenie; nie tworzą różnorodności pacjentów |
| Losowy oversampling | Prosty, odtwarzalny; zwiększa częstość prezentacji rzadkich uderzeń bez zmiany ich morfologii | Powtarza te same okna i pacjentów; ryzyko przeuczenia; więcej kroków/próbek; nie tworzy nowej informacji |
| Losowy undersampling | Zmniejsza liczbę próbek i koszt obliczeń | Może usunąć wiele użytecznych i różnorodnych uderzeń N; zrównanie do najmniejszej klasy może być bardzo stratne |
| Augmentacja | Potencjalnie zwiększa różnorodność sygnałów treningowych | Szum, skalowanie lub deformacje czasu mogą zmienić P/QRS/T i znaczenie etykiety; wymagają osobnego uzasadnienia i kontroli. Nie zaimplementowano augmentacji ani syntetycznych uderzeń |

Dla późniejszych eksperymentów rozsądnym punktem odniesienia są wagi klas albo kontrolowany oversampling treningu; nie wykazano ich przewagi w tym projekcie. Agresywny undersampling całej większości może usuwać ważne dane. Nie należy automatycznie łączyć silnych wag i pełnego oversamplingu, ponieważ obie metody mogą nadmiernie zwiększyć wpływ tej samej mniejszości. Wybór strategii, jej intensywności i sposobu użycia w funkcji straty/samplerze należy do Anny. Nie ustalamy ich na podstawie wyników testowych.

Resampling całej bazy przed podziałem grozi przeciekiem — kopie tego samego uderzenia i pacjenta mogą znaleźć się w różnych częściach. Wagi i rozkład samplera także muszą wynikać wyłącznie z treningu. Walidacja/test zachowują oryginalny rozkład. Zobacz [oficjalne omówienie resamplingu i leakage](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/doc/common_pitfalls.rst). Nie dodano zależności imbalanced-learn ani sklearn.

### API analizy

```python
from src.class_imbalance import analyze_database, analyze_labeled
report = analyze_database()  # rzeczywista pełna lokalna baza, bez balansowania
# report = analyze_database(record_ids=["100", "101", "200"])
# analyze_labeled(labeled) analizuje istniejący obiekt bez ponownego preprocessingu.
```

`describe_counts` realizuje czystą arytmetykę i może być testowane na małych fixture'ach. Raporty produkcyjne skryptu pochodzą jednak wyłącznie z `analyze_database` i rzeczywistych okien. Sumy z fazy 8 dla 100/101/200 pozostają niezmienione. Parametr `progress` pozwala obserwować przetwarzanie nagrań. Nie eksportujemy końcowego datasetu.

### Narzędzia treningowe: warunki użycia i ograniczenia zabezpieczeń

Nie ma domyślnego wejścia do balansowania ani funkcji automatycznie uruchamianej przez analizę. Dodano:

- `make_training_subset(labeled, indices, *, partition, patient_split_confirmed)` — wymaga jawnej niepustej listy indeksów, `partition="train"`, `patient_split_confirmed=True`, rozstrzygniętych grup pacjentów i poprawnych etykiet 0..4. Odrzuca duplikaty, indeksy poza zakresem, całą tablicę źródłową i nierozwiązane grupy. Nie wybiera indeksów ani pacjentów. Zapisuje kopie metadanych i etykiet tylko wybranych próbek jako tablice read-only, bez kopiowania/modyfikowania X.
- `training_class_weights(subset)` — liczy wyłącznie z przekazanego treningu `n_train / (k_present * count_class)`. Dla klas nieobecnych zwraca 0, bez dzielenia przez zero. Jest to wariant [wzoru inverse-frequency](https://sklearn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html), gdzie k to liczba klas obecnych w treningu; nie liczymy wag z globalnej tabeli. Wynik to słownik indeks klasy → float, bez implementacji loss.
- `training_sampling_indices(subset, strategy=..., random_seed=...)` — zwraca indeksy w ŹRÓDŁOWYM `LabeledHeartbeats`, nie nowe ECG i nie indeksy względne podzbioru. Wymaga jawnego nieujemnego seedu. Oversample zachowuje oryginały i dodaje losowe powtórzenia do liczebności największej obecnej klasy; undersample losuje bez zwracania do najmniejszej niepustej. Wynik jest przemieszany deterministycznie dla ustalonego seedu/wersji NumPy. Klasy puste nie są tworzone.

Klasa `TrainingSubset` przechowuje też ID rekordów, grupy pacjentów, pozycje adnotacji i rozmiar źródła. Obie funkcje ponownie sprawdzają kontrakt. Surowe y, brak podzbioru, etykiety unmapped, walidacja/test albo brak potwierdzenia są odrzucane. Gdy wszystkie wybrane próbki należą do jednej klasy, sampler nie naprawi braku pozostałych.

**Ograniczenie:** API opiera się również na deklaracji wywołującego. Nie może udowodnić, że źródło nie zostało wcześniej ograniczone do samego treningu, że grupy są prawdziwe lub że nie ma wspólnego pacjenta w walidacji/test. Odrzucenie pełnej tablicy jest zabezpieczeniem przed przypadkowym użyciem globalnego zbioru, a nie kryptograficzną gwarancją braku leakage. Przygotowanie i sprawdzenie splitu należy wyłącznie do Anny; nie implementujemy go. W szczególności nie wolno dostarczać wymyślonych grup tylko po to, aby przejść walidację API.

Szablon integracyjny (zmienne ma wcześniej ustalić Anna; faza 9 ich nie tworzy):

```python
from src.class_imbalance import (
    make_training_subset, training_class_weights, training_sampling_indices,
)
train = make_training_subset(
    labeled_source_with_resolved_patient_groups,
    anna_train_indices,
    partition="train", patient_split_confirmed=True,
)
# Wybór strategii należy do Anny, to nie polecenie łączenia obu metod:
weights = training_class_weights(train)
sampled_source_indices = training_sampling_indices(
    train, strategy="oversample", random_seed=42,
)
# Późniejszy kod Anny może użyć source_indices do dostępu do X i y.
# Nie używaj tego na validation/test i nie przeliczaj wag z tych części.
```

Nie wykonywano tych operacji na rzeczywistym kompletnym zbiorze ani nie tworzono splitu. Testy narzędzi korzystają z małych syntetycznych fixture'ów; rzeczywiste liczebności analizy nie pochodzą z tych fixture'ów. Brak klasy w treningu jest problemem pokrycia danych, którego waga 0 ani powtarzanie próbek nie rozwiąże. Użycie wag, długość epoki, sampler i późniejsza ocena to obowiązki Anny. Statystyki preprocessingu nie powinny być dopasowywane globalnie do niezależnych pacjentów; dotychczasowe ograniczenia normalizacji per nagranie nadal obowiązują.

### Reprodukcja i pliki wynikowe

```bash
cd ~/Pulpit/ECG-APP
.venv/bin/python -m examples.analyze_class_imbalance
.venv/bin/python -m pytest
```

Parametry: `--data-dir`, `--output-dir`, `--rare-count-threshold`. Przy powtórnym generowaniu wybierz nowy katalog, np. `--output-dir results/preprocessing/phase9_run2`; nie nadpisujemy istniejących plików.

W `results/preprocessing/phase9/` zapisano:

- `class_counts.png` — rzeczywiste globalne liczebności;
- `class_percentages.png` — udziały procentowe;
- `class_counts_log.png` — liczebności na osi logarytmicznej; puste klasy są pomijane w tym wykresie, ale pozostają w tabeli;
- `recording_class_heatmap.png` — 48 rekordów × pięć klas, kolor ln(1+count), a liczby w komórkach są surowymi liczebnościami; zero pozostaje zerem danych;
- `class_distribution.csv` — 5 wierszy liczebności, udziałów i flag zero/rare;
- `recording_class_distribution.csv` — 48 wierszy z liczebnościami i procentami per rekord oraz klasami rzadkimi/brakującymi;
- `recording_patient_groups.csv` — zachowane rekordy i jawnie nierozwiązane grupy pacjentów;
- `segmentation_audit.csv` — zgodność sum adnotacji, segmentów, mapped/unmapped i powodów wykluczeń w każdym rekordzie;
- `imbalance_summary.json` — pełny raport opisowy i konfiguracje (bez ECG i bez globalnych wag treningowych).

Obrazy zapisano w 300 dpi; otworzysz je w eksploratorze VS Code. Wszystkie wyniki są wykluczone istniejącą regułą Git `results/preprocessing/`. Nie zmieniono bazy ani mapy AAMI i nie rozpoczęto fazy 10.


### Weryfikacja fazy 9

Końcowe `.venv/bin/python -m pytest`: **321 passed, 80 warnings in 34.72s** — 283 wcześniejsze i 38 nowych testów. Ostrzeżenia pochodzą z użycia przestarzałego API pandas wewnątrz WFDB, bez błędów. Testy obejmują liczebności/procenty, remisy i mniejszość niepustą, zera, puste wejścia, unmapped, deterministyczność, zachowanie metadanych, walidację selekcji treningowych, wzór wag, obecne/nieobecne klasy samplera, źródłowe indeksy oraz integrację z prawdziwymi rekordami 100/101/200.

Pełny skrypt analizy 48 rekordów zakończył się kodem 0, bez wywoływania narzędzi balansowania. Zweryfikowano również wszystkie per-record sumy i procenty zapisane w JSON oraz liczby wierszy czterech CSV (5, 48, 48, 48). Obejrzano cztery obrazy. Kontrola składni nowych plików przez compileall przeszła. Nie wdrożono modelu, treningu, podziału danych, ewaluacji, augmentacji, SMOTE ani finalnego eksportu. Baza i istniejąca mapa AAMI pozostały bez zmian. Prace zakończono po fazie 9.


## Faza 10 — finalny eksport i przekazanie datasetu Annie

Eksport zakończono i niezależnie odczytano. Gotowe dane są w **`data/processed/ecg_dataset_v1/`**. Pełna instrukcja interfejsu, działający przykład Python, grupy osób, normalizacja, ograniczenia kanałów i reprodukcja są w **`DATASET_FOR_ANNA.md`**.

### Wynik końcowy

- 109452 kompletne segmenty z 48 nagrań i 47 grup osób w obrębie MIT-BIH;
- logiczne X `[109452, 2, 252]`, float32, amplituda bezwymiarowa; y `[109452]`, int64;
- klasy N=90594, S=2781, V=7235, F=802, Q=8040, dokładnie zgodne z fazą 9;
- wszystkie ECG skończone, etykiety 0–4, oryginalne symbole i pozycje ekspertów zachowane;
- dane nie są zbalansowane ani podzielone na zbiory;
- 48 NPZ (po jednym na nagranie), dwa JSON i dwa CSV: łącznie 52 pliki, około 186,46 MiB;
- 3153 nieuderzenia i 42 niepełne okna wykluczone zgodnie z wcześniejszą segmentacją, bez dopisywania brakujących próbek.

Zapis jest shardowany, aby uniknąć łączenia dużych tablic w RAM. Per-beat pola mają tę samą liczbę wierszy; metadane nagrania (np. dwie nazwy kanałów i pięć nazw klas) mają osobne kształty. NPZ można wczytać bez pickle przez `load_record_shard` albo `np.load(..., allow_pickle=False)`. X i y są od razu typami zgodnymi z wejściem NumPy do PyTorch; nie dodano kodu modelu.

### Zweryfikowane grupy osób

Dodano wersjonowalny plik **`metadata/patient_groups.json`** i jego kopię w eksporcie. Źródła: [oficjalny opis PhysioNet](https://physionet.org/content/mitdb/1.0.0/) oraz [opis oryginalny MIT-BIH](https://archive.physionet.org/physiobank/database/html/mitdbdir/intro.htm). Podają 48 nagrań od 47 osób oraz wspólnego pacjenta 201/202.

201 i 202 mają `mitdb_group_201_202`; inne rekordy — odrębne techniczne tokeny grup. To nie wymyślone rzeczywiste ID pacjentów. Odrębność pozostałych grup wynika z dokumentowanej liczby osób i jawnego powtórzenia pary, przy jednym pacjencie na rekord. Plik odróżnia relację bezpośrednio opisaną (`explicit_same_subject_201_202`) od odrębności wywnioskowanej z liczebności (`distinct_by_documented_cardinality`), podaje źródła i ograniczenie do tej bazy. Nie znamy rzeczywistych tożsamości ani powiązań poza MIT-BIH.

Grupowanie jest gotowe dla Anny; split pozostaje jej zadaniem. Historyczne raporty faz 7–9 prawidłowo opisują wcześniejszy brak mapy. Finalne patient_groups nie są unresolved. Nie tworzymy podziałów pacjentów ani walidacji.

### Niezmieniony preprocessing i audyt kanałów

Eksporter wywołuje istniejące `load_recording` → `segment_recording` (filtracja, Z-score, okna) → `map_segments` → istniejącą analizę opisową → zapis. Nie powielono algorytmów i nie zmieniono domyślnych ustawień. Dokładna konfiguracja przy każdym shardzie zawiera parametry Butterwortha 0,5–45 Hz / rząd 4 / SOS / zero-phase / notch off, Z-score per recording i per channel po filtracji (epsilon 1e-8, mean/std), okno 90+162, politykę brzegową oraz wersję mapy `mitdb-aami-style-v1`.

Nie dopasowywano globalnych parametrów do połączonych pacjentów. Per-recording jest procedurą offline, korzystającą z całego nowego nagrania. Strategia global training-only wymaga przyszłego splitu Anny i innego jawnego fit — nie została zastosowana. Reprodukcja nowych pełnych nagrań wymaga tego samego kontekstu filtracji, ustawień i zgodności fs/kanałów. Krótkie okno w czasie rzeczywistym nie jest równoważnym kontekstem; nie wdrażamy modelowej inferencji.

Zachowano wszystkie nazwy z nagłówków bez relabelingu: MLII/V1 — 40 nagrań; MLII/V5 — 2; V5/V2 — 2; MLII/V2 — 2; V5/MLII — 1; MLII/V4 — 1. Szczegóły są w `lead_configuration.csv`, JSON i NPZ. Rekord 114 ma odwrócony porządek względem typowego MLII na indeksie 0. Indeksy kanałów nie mają jednolitej semantyki anatomicznej w całej bazie. Decyzja o obsłudze różnic w przyszłym CNN należy do Anny, bez cichego przepinania kanałów przez preprocessing.

### Nowe pliki i odtworzenie

Utworzono `src/prepare_dataset.py`, `tests/test_prepare_dataset.py`, `metadata/patient_groups.json` i `DATASET_FOR_ANNA.md`; uzupełniono ten README. Nie zmieniono wcześniejszych modułów.

```bash
cd ~/Pulpit/ECG-APP
# Kontrola istniejącego eksportu — nie zapisuje danych:
.venv/bin/python -m src.prepare_dataset --verify-only

# Reprodukcja do NOWEGO katalogu:
.venv/bin/python -m src.prepare_dataset \
  --output-dir data/processed/ecg_dataset_reproduced

.venv/bin/python -m pytest
```

Domyślny eksport wymaga pełnych 48 nagrań, zgodnego grupowania i liczebności fazy 9; rozbieżność zatrzymuje go z błędem, bez inventowania lub usuwania segmentów dla dopasowania sum. API pozwala także na jawne małe eksporty kontrolne `record_ids=["100"]` — manifest oznacza wtedy `scope="explicit_subset"` i nie deklaruje zgodności z całą fazą 9.

Istniejący katalog wyjściowy, nawet pusty, jest odrzucany. Nie nadpisujemy go i nie usuwamy częściowych plików po awarii: należy je obejrzeć i wskazać nową ścieżkę. `dataset_metadata.json` jest zapisywany jako ostatni, z `status="complete"`. Ścieżki shardów są względne względem folderu eksportu, więc można przenieść go razem z manifestem i metadanymi. Baza ani paczki nie są pobierane podczas eksportu.

Manifest zapisuje wersje Python i bibliotek, hashe kodu, plików wyjściowych i **676 plików źródłowych** z całego drzewa bazy (206 to tylko liczba plików bezpośrednio w jej głównym katalogu). Hashowano źródła przed i po przygotowaniu — bez zmiany treści. `verify-only` ponownie sprawdza wszystkie shardy, klasy, grupy, oryginalne adnotacje WFDB, nagłówki i źródłowe hashe. Eksporty i baza pozostają wykluczone z Git; małą mapę pacjentów i dokumentację można wersjonować.

### Weryfikacja finalna

Pełne `.venv/bin/python -m pytest`: **342 passed, 132 warnings in 72.49s** — 321 wcześniejszych i 21 nowych testów. Ostrzeżenia pochodzą z przestarzałego użycia pandas wewnątrz WFDB; nie wystąpiły błędy.

Testy fazy 10 obejmują dtype i kształty, skończoność wszystkich shardów, wspólną długość pól per beat, wymagane metadane, zachowanie oryginalnych adnotacji i wartości preprocessingu po konwersji float32, poprawne klasy, konfigurację, zgodność liczebności z fazą 9, wspólną grupę 201/202, odczyt oraz wyjątki kanałowe, ochronę przed nadpisaniem, deterministyczne powtórzenie eksportu w tym samym środowisku i niezmienność całej bazy.

Niezależna kontrola CLI zakończyła się sukcesem: 109452 segmenty / 48 nagrań / 47 grup i zgodne klasy. Działający przykład odczytu dla Anny przetestowano: X dla 100 ma `(2271, 2, 252)` float32, y `(2271,)` int64. Zgodność powtórnego eksportu dotyczy identycznego wejścia i wersji środowiska; nowe wersje bibliotek mogą wprowadzić małe różnice numeryczne. Konwersja float64 → float32 jest świadomym formatem końcowym.

### Podsumowanie faz 1–10 i granica odpowiedzialności

| Faza | Ukończony zakres |
|---|---|
| 1 | Inspekcja projektu, środowiska i lokalnej bazy |
| 2 | Odczyt sygnałów i adnotacji WFDB, walidacja danych |
| 3 | Wykresy kanałów i oryginalnych adnotacji |
| 4 | Konfigurowalne filtry offline, notch opcjonalny |
| 5 | Z-score/Min-Max oraz zapis i odtwarzanie parametrów |
| 6 | Zsynchronizowane porównania etapów i statystyki |
| 7 | Okna uderzeń według ekspertów z metadanymi i raportem granic |
| 8 | Jawna mapa AAMI-style i kontrola symboli nieprzyporządkowanych |
| 9 | Pełna analiza klas i opcjonalne utilities wyłącznie dla treningu |
| 10 | Zweryfikowane grupy osób, audyt kanałów, pełny eksport i handoff |

Zakres Data Engineering & ECG Preprocessing Magdaleny w fazach 1–10 jest ukończony i przetestowany. Pozostają jawne ograniczenia danych: offline/per-recording normalization, różne odprowadzenia, szum i artefakty, możliwe efekty brzegowe, nierównowaga i mała różnorodność niektórych klas, techniczny zakres grupowania osób oraz brak gwarancji klinicznej. Te ograniczenia nie są ukrywane przez eksport. Anna odpowiada dalej za decyzje integracyjne modelu, podział patient-separated, trening, balansowanie tylko treningu, ewaluację i Streamlit. Żadnego z tych komponentów nie implementowano.


### Ponowna weryfikacja fazy 10 — zakres MIT-BIH

Istniejący eksport zachowano bez nadpisania. Ponowna weryfikacja potwierdziła 109452 segmenty z 48 nagrań i 47 grup oraz identyczny rozkład klas z fazą 9. Źródłowe hashe wszystkich 676 plików i hashe kodu są zgodne z manifestem. Pełne testy: **342 passed, 132 warnings in 64.10s**; ostrzeżenia pochodzą z pandas wewnątrz WFDB. Przykład ładowania X/y i metadanych wykonano poprawnie.

`DATASET_FOR_ANNA.md` uzupełniono o zakres wyłącznie MIT-BIH heartbeat classification oraz jawne nazwy pól odpowiadające specyfikacji. Eksport nie zawiera PTB-XL ani etykiet klasyfikacji rytmu; przyszłe rozszerzenia wymagają osobnych kontraktów danych i nie zostały zaimplementowane. Nie zmieniono walidowanych konfiguracji. Raport końcowy zapisano w `FINAL_MITBIH_EXPORT_REPORT.md`.
