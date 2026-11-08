# Dataset ECG dla Anny — przekazanie po fazie 10

Gotowy eksport: `data/processed/ecg_dataset_v1/` względem katalogu projektu `~/Pulpit/ECG-APP`. Obejmuje wszystkie 48 lokalnych nagrań, **109452 segmenty**, 47 grup osób, dwa kanały i 252 próbki na segment przy 360 Hz. Dane nie są zbalansowane i nie mają podziału trening/walidacja/test. Nie przygotowano modelu ani ewaluacji.

## Układ plików i pamięć

```text
data/processed/ecg_dataset_v1/
├── recordings/
│   ├── 100.npz
│   ├── 101.npz
│   └── ... 48 plików, jeden na nagranie
├── dataset_metadata.json
├── patient_groups.json
├── class_distribution.csv
└── lead_configuration.csv
```

Wybrano shardy NPZ zamiast jednego dużego NPZ: przygotowanie i odczyt wymagają pamięci tylko na jedno nagranie. Łączny rozmiar eksportu to około 186,46 MiB; sama logiczna tablica X bez kompresji ma około 210,43 MiB. Nie trzeba jej sklejać w RAM. NPZ nie zapewnia efektywnego mmap skompresowanej tablicy; pojedynczy shard jest rozpakowywany do pamięci. Kolejność globalna to rosnące ID nagrania i oryginalna kolejność adnotacji wewnątrz niego, zgodnie z manifestem. X logicznie ma kształt `[109452, 2, 252]`, ale pliki przechowują `[liczba_uderzeń_w_nagraniu, 2, 252]`.

Manifest `dataset_metadata.json` powstaje jako ostatni plik i `status="complete"` oznacza zakończony eksport. Katalog częściowego/nieudanego eksportu bez tego manifestu nie jest gotowym datasetem. Nie nadpisujemy istniejących katalogów.

## Działający przykład odczytu w Pythonie

Uruchom z katalogu projektu, wybierając interpreter `.venv/bin/python`:

```python
from pathlib import Path
import json
import numpy as np
from src.prepare_dataset import load_record_shard

root = Path("data/processed/ecg_dataset_v1")
metadata = json.loads((root / "dataset_metadata.json").read_text())
assert metadata["segment_count"] == 109452

entry = metadata["shards"][0]
data = load_record_shard(root / entry["file"])  # walidacja, allow_pickle=False
X = data["X"]                         # float32, [2271, 2, 252] dla 100
Y = data["y"]                         # int64, [2271]
record_ids = data["record_ids"]
patient_groups = data["patient_groups"]
channel_names = data["channel_names"]
config = json.loads(str(data["preprocessing_config"]))
print(X.shape, X.dtype, Y.shape, Y.dtype)
print(channel_names, np.unique(patient_groups))
print(metadata["class_names"])
```

Standardowy NumPy także wystarcza: `with np.load(path, allow_pickle=False) as f: X = f["X"]; y = f["y"]`. Każdy shard jest samodzielnie opisany. Do iteracji bez scalania:

```python
for entry in metadata["shards"]:
    data = load_record_shard(root / entry["file"])
    X, y = data["X"], data["y"]
    # Tutaj Anna może przekazać tablice do własnego interfejsu danych.
    # Nie ma tutaj modelu, treningu, splitu ani inferencji modelu.
```

W środowisku Anny z PyTorch tablice numeryczne są zgodne z `torch.from_numpy(X)` i `torch.from_numpy(y)` bez zmiany osi. Nazwy/ID są tablicami Unicode i pozostają metadanymi, nie tensorami cech. Przy współdzieleniu pamięci przez `from_numpy` nie należy modyfikować źródłowego X. Projekt preprocessingu nie wymaga instalowania PyTorch.

## Pola każdego pliku NPZ

| Pole | Typ i kształt | Znaczenie |
|---|---|---|
| X | float32 `[n, 2, 252]` | ECG po filtracji i Z-score, amplituda bezwymiarowa |
| y | int64 `[n]` | Indeksy klas 0–4 |
| record_ids | Unicode `[n]` | Oryginalne trzycyfrowe ID źródłowego nagrania |
| patient_groups | Unicode `[n]` | Zweryfikowane grupy osób w obrębie MIT-BIH |
| annotation_positions | int64 `[n]` | Oryginalne indeksy próbek ekspertów w nagraniu |
| original_symbols | Unicode `[n]` | Oryginalne symbole WFDB, z rozróżnieniem wielkości liter |
| annotation_indices | int64 `[n]` | Indeksy w oryginalnej tablicy adnotacji .atr |
| annotation_aux_notes | Unicode `[n]` | Zachowane notatki pomocnicze adnotacji |
| class_names | Unicode `[5]` | N, S, V, F, Q w kolejności indeksów |
| sampling_frequency | float, skalar | Oryginalne 360 Hz, bez resamplingu |
| channel_names | Unicode `[2]` | Nazwy w oryginalnej kolejności nagłówka, wspólne dla danego sharda |
| preprocessing_config | Unicode, skalar JSON | Filtry, normalizacja i jej statystyki, okno, mapa i wersje |

Tylko pola per uderzenie mają pierwszy wymiar n; class_names, fs, channel_names i config są metadanymi całego nagrania. Manifest zawiera wszystkie shardy, liczby, grupy, kanały, SHA-256 oraz wersje bibliotek i kodu.

## Etykiety i liczby

| Indeks | Klasa | Symbole MIT-BIH | Liczba |
|---|---|---|---:|
| 0 | N | N, L, R, e, j | 90594 |
| 1 | S | A, a, J, S | 2781 |
| 2 | V | V, E | 7235 |
| 3 | F | F | 802 |
| 4 | Q | /, f, Q | 8040 |

To dokładnie liczby z fazy 9. Mapowanie `mitdb-aami-style-v1` jest zapisane w konfiguracji. N nie oznacza wyłącznie zdrowego pacjenta; Q obejmuje również uderzenia stymulowane. Nie ma etykiet -1 w finalnym eksporcie. Nieznany symbol powoduje błąd zamiast cichego przypisania. Nie wymyślano brakujących segmentów. 112647 adnotacji = 109452 kompletne okna + 3153 nieuderzenia + 42 niepełne okna brzegowe.

## Grupy pacjentów: co jest zweryfikowane, a co oznacza identyfikator?

Źródła: [oficjalna karta PhysioNet](https://physionet.org/content/mitdb/1.0.0/) oraz [oryginalny opis MIT-BIH](https://archive.physionet.org/physiobank/database/html/mitdbdir/intro.htm). Dokumentują 48 nagrań od 47 osób i wspólnego pacjenta rekordów 201/202.

- 201 i 202 mają wspólny token **`mitdb_group_201_202`**, na podstawie jawnie opisanej relacji.
- Pozostałe rekordy otrzymują odrębne tokeny `mitdb_group_<record_id>`. Ich odrębność wynika z opisanej liczby osób i jedynego powtórzenia 201/202, przy jednym pacjencie w nagraniu; nie pochodzi z wymyślonej listy rzeczywistych ID osób.
- Tokeny są technicznymi identyfikatorami grup datasetu, nie rzeczywistymi identyfikatorami pacjentów. Nie znamy nazwisk ani powiązań z innymi bazami. Singletony mają jawny `relationship_basis="distinct_by_documented_cardinality"`, para — `explicit_same_subject_201_202`.

Przejrzysta mapa z pochodzeniem i zakresem pewności jest w wersjonowalnym `metadata/patient_groups.json` oraz identycznej kopii w eksporcie. Łącznie jest 47 grup. Grupowanie jest gotowe do wykorzystania przez Annę, ale **nie utworzono żadnego splitu**.

Anna musi dzielić dane według patient_groups, aby okna tej samej osoby nie znalazły się po obu stronach podziału. 201 i 202 muszą pozostać razem. Losowy podział samych uderzeń, które dodatkowo mogą zachodzić na siebie, grozi leakage. Wybór podziału i walidacji należy do Anny. Wcześniejsze raporty faz 7–9 z unresolved groups dokumentują dawny stan; finalny eksport używa nowej jawnej mapy.

## Preprocessing i normalizacja

Sekwencja nie została zmieniona:

1. WFDB odczytuje lokalne .hea/.dat i eksperckie .atr.
2. Butterworth SOS high-pass 0,5 Hz, następnie low-pass 45 Hz, rząd 4; filtracja dwukierunkowa zero-phase; notch wyłączony.
3. Z-score pełnego nagrania **osobno dla każdego kanału**, po filtracji, przed wycięciem okien; std populacyjne ddof=0, epsilon 1e-8; kanały niemal stałe zerowane zgodnie z dotychczasową polityką.
4. Okno `[annotation-90, annotation+162)`, 252 próbki, adnotacja pod indeksem 90. Czasy 0,25/0,45 s są zaokrąglane osobno metodą floor(t*fs+0.5). Nie szukamy nowych pików i nie dopełniamy niepełnych okien.
5. Dotychczasowa mapa AAMI-style, zachowanie metadanych, dopiero na końcu konwersja X z float64 do float32.

Dokładne parametry i policzone mean/std (offset/scale) są w `preprocessing_config` każdego NPZ. Klasa `ECGNormalizer` już umożliwia odtworzenie parametrów ze słownika; nie dodano modelu inferencyjnego.

**Per-recording** oznacza, że każde nagranie ustala swoją średnią i skalę niezależnie. Nie estymowano wspólnych parametrów z połączonego treningu i testu. Jednak używa to całego dostępnego zapisu, również przyszłych próbek: procedura jest offline i nie odpowiada predykcji w czasie rzeczywistym z krótkiego okna. Przy nowych pełnych nagraniach trzeba konsekwentnie stosować ten sam preprocessing, z nowymi statystykami tego nagrania. Dla odtwarzania już przetworzonego rekordu należy użyć zapisanych parametrów i tego samego kontekstu filtracji.

**Global training-only** to inna strategia: wspólne parametry dopasowane wyłącznie na pacjentach treningowych, a potem zamrożone dla nowych danych. Zależy od przyszłego splitu Anny. Nie wykonywano takiego fit i nie należy mieszać obu strategii bez ponownego, jawnego przygotowania danych. Zmiana polityki wejścia to decyzja integracyjna z Anną. Normalizacja per kanał usuwa bezwzględną skalę mV i relacje amplitud między kanałami; to nie jest dataset do pomiarów diagnostycznych amplitud.

## Audyt kanałów — ważne ograniczenie wejścia CNN

| Oryginalna kolejność kanałów | Liczba nagrań |
|---|---:|
| MLII, V1 | 40 |
| MLII, V5 | 2 |
| V5, V2 | 2 |
| MLII, V2 | 2 |
| V5, MLII | 1 |
| MLII, V4 | 1 |

Wyjątki od MLII/V1: 100 i 123 — MLII/V5; 102 i 104 — V5/V2; 103 i 117 — MLII/V2; **114 — V5/MLII**; 124 — MLII/V4. Wszystkie nazwy pochodzą z nagłówków i są zachowane bez zamiany kanałów. Lead configuration jest w `lead_configuration.csv`, manifeście i NPZ. Te dwa kanały nie są anatomicznie równoważnymi cechami między wszystkimi zapisami. Anna musi uwzględnić to w projekcie modelu/danych; nie harmonizowano ani nie przemianowano odprowadzeń. Archiwalny opis dokumentuje też możliwy analogowy skew między kanałami. Nie wyrównywano ich przez przesuwanie adnotacji.

## Imbalance i zakres zadań Anny

N dominuje, a F stanowi mniej niż 1% okien. Globalne proporcje nie są wagami treningowymi. Anna powinna najpierw ustalić patient-separated split, sprawdzić pokrycie klas w treningu i dopiero wtedy ewentualnie skorzystać z narzędzi fazy 9. Walidacji i testu nie należy balansować. Powielanie uderzeń nie zwiększa liczby pacjentów. Nie wykonano oversamplingu, undersamplingu, augmentacji, wyboru wag ani tworzenia splitu.

## Weryfikacja i odtworzenie eksportu

```bash
cd ~/Pulpit/ECG-APP
# Istniejący eksport: tylko sprawdzenie (bez nadpisania)
.venv/bin/python -m src.prepare_dataset --verify-only

# Nowy eksport — podaj NOWY, nieistniejący katalog
.venv/bin/python -m src.prepare_dataset \
  --output-dir data/processed/ecg_dataset_reproduced

.venv/bin/python -m pytest
```

Dodatkowo można wskazać `--data-dir`. Domyślne uruchomienie wymaga wszystkich 48 lokalnych nagrań i zgodności mapy pacjentów. Rozbieżność z liczebnościami fazy 9 zatrzymuje pełny eksport z błędem; nie dodaje/uszczupla segmentów dla uzyskania zgodności. Eksport nie pobiera danych.

Zapisano SHA-256 wszystkich 676 lokalnych plików źródłowych (w tym plików bocznych) przed i po przygotowaniu; treści pozostały identyczne. Manifest zawiera też sumy kontrolne NPZ/metadata, hashe kodu oraz wersje Python i bibliotek. `verify-only` niezależnie czyta wszystkie shardy, sprawdza skończoność, dtype, etykiety i grupy, porównuje symbole i pozycje z WFDB, nagłówki z kanałami oraz ponownie sprawdza źródłowe hashe. Zgodność liczbową powtórnego eksportu sprawdzamy w tej samej wersji środowiska; inne wersje bibliotek mogą dać drobne różnice zmiennoprzecinkowe. Konwersja do float32 jest zamierzonym zaokrągleniem, nie zmianą metody preprocessingu.

Oryginalna baza i eksporty są wykluczone przez `.gitignore`. Do Git można dodać kod, dokumentację i `metadata/patient_groups.json`, bez dużych danych.

## Znane ograniczenia

- Dataset ma adnotacje eksperckie i okna o stałej długości, nie automatyczną detekcję/delineację ani ocenę kliniczną.
- Niektóre okna obejmują fragmenty sąsiednich uderzeń, szum i artefakty; kompletny kształt okna nie gwarantuje jakości sygnału.
- Filtracja może zmienić morfologię i powodować efekty brzegowe; zero-phase nie jest przyczynowe.
- Q obejmuje paced/fusion/unknown, N obejmuje także niektóre zaburzenia przewodzenia; nie są to diagnozy pacjenta.
- Zachowano nagrania stymulatorowe, więc nie należy porównywać liczebności bezpośrednio z pracami używającymi 44 rekordów.
- Grupy są zweryfikowane tylko w obrębie tej bazy, a odrębność singletonów jest wywnioskowana z dokumentowanych liczebności.
- Kanały różnią się pomiędzy zapisami; nie należy zakładać jednakowej semantyki indeksu 0/1.
- Model, podziały pacjentów, trening, ewaluacja i aplikacja nadal należą wyłącznie do Anny.


## Zakres tego eksportu: MIT-BIH heartbeat classification

Ten eksport jest przeznaczony wyłącznie do klasyfikacji **pojedynczych, ekspercko adnotowanych uderzeń MIT-BIH** do N/S/V/F/Q. Nie zawiera celów klasyfikacji rytmu, etykiet PTB-XL ani przykładów przygotowanych z PTB-XL. Symbole rytmu z adnotacji .atr nie zostały zamienione na targety tego datasetu.

Planowana klasyfikacja rytmu będzie odrębnym zadaniem: etykieta rytmu opisuje aktywność w przedziale czasu, nie jest tożsama z typem pojedynczego uderzenia. Przyszła obsługa PTB-XL będzie wymagała osobnego kontraktu danych i etykiet. Nie należy automatycznie przenosić na nią obecnych okien 252 próbek, dwóch kanałów, fs=360 Hz, mapy AAMI ani grup osób MIT-BIH. Nie wdrażamy ani nie ustalamy teraz preprocessingu tych rozszerzeń.

### Nazwy pól a nazwy używane w specyfikacji fazy 10

Dla zachowania gotowego eksportu nie przemianowano jego kluczy:

- `original_annotation_symbols` ze specyfikacji są zapisane jako `original_symbols`;
- `preprocessing_configuration` jest zapisana jako JSON w `preprocessing_config`;
- `segmentation_configuration` jest częścią tego samego JSON: czasy żądane, before_samples, after_samples, sequence_length, rounding, interval i boundary_policy;
- `label_mapping_version` jest zapisana pod kluczem `mapping_version` w tym JSON oraz w dataset_metadata.json.

Wszystkie te informacje są obecne; nie jest potrzebna ponowna filtracja ani przeliczenie etykiet. Przykład jawnego odczytu pod nazwami ze specyfikacji:

```python
import json
from src.prepare_dataset import load_record_shard

data = load_record_shard("data/processed/ecg_dataset_v1/recordings/100.npz")
X, y = data["X"], data["y"]
original_annotation_symbols = data["original_symbols"]
preprocessing_configuration = json.loads(str(data["preprocessing_config"]))
segmentation_configuration = {
    key: preprocessing_configuration[key]
    for key in ("before_seconds_requested", "after_seconds_requested", "before_samples",
                "after_samples", "sequence_length", "rounding", "interval", "boundary_policy")
}
label_mapping_version = preprocessing_configuration["mapping_version"]
```

Powyższy fragment tylko odczytuje dane i metadane. Nie zmienia formatu ani nie implementuje klasyfikacji rytmu, modelu, treningu, splitu lub inferencji modelu.
