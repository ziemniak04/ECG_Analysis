# Dane ECG — finalne przekazanie dla Anny

**Zweryfikowano oba kompletne eksporty.** Projekt zawiera dwa osobne zadania: klasyfikację uderzeń MIT-BIH i nazwanych rytmów PTB-XL. Nie łączyć ich tablic X, indeksów y ani identyfikatorów pacjentów. Dokument potwierdza inżynierię danych, nie skuteczność ani kliniczną przydatność przyszłego modelu.

## Lokalizacja i reprezentacja

Wszystkie ścieżki względem `~/Pulpit/ECG-APP`; interpreter `.venv/bin/python`.

| Dataset | Lokalizacja i manifest | Logiczny X | y | fs |
|---|---|---|---|---|
| MIT-BIH heartbeat | `data/processed/ecg_dataset_v1/`, `dataset_metadata.json` | float32 `[109452,2,252]` | int64 `[109452]` | 360 Hz |
| PTB-XL rhythm | `data/processed/ptbxl_rhythm_v1/`, `runs/90b449bf54b649509519747f417b9e5e/PTBXL_DOWNLOAD_MANIFEST.json` | float32 `[19605,12,1000]` | int64 `[19605]` | 100 Hz |

Nie istnieje jedna scalona tablica N w pamięci: MIT-BIH ma 48 shardów `recordings/<record_id>.npz`, każdy `[n,2,252]`; PTB-XL ma 19605 plików `recordings/<ecg_id:05d>_rhythm.npz`, każdy `[12,1000]` i skalarny y. Duże archiwa należy czytać po shardzie/rekordzie. Skompresowane NPZ nie zapewniają efektywnego mmap; odczyt rozpakowuje pojedynczy plik. Wszystkie loadery używają `allow_pickle=False`.

**Wskazany run PTB-XL jest kompletny:** 19605 sukcesów, 0 failed, 0 not_attempted, 17223 pacjentów. Starszy run `4760b72c29934d1ab406cf7b9ded933a`, rootowy manifest checkpoint A oraz dawne raporty o DNS są historyczne. Nie używać ich jako aktualnej listy przykładów. Nie zmieniano tych wcześniejszych artefaktów.

## Etykiety

MIT-BIH: pięć klas AAMI-style uderzeń, bez końcowych celów rytmu.

| Indeks | Klasa | Uderzenia | Udział |
|---|---|---:|---:|
| 0 | N | 90594 | 82.771% |
| 1 | S | 2781 | 2.541% |
| 2 | V | 7235 | 6.610% |
| 3 | F | 802 | 0.733% |
| 4 | Q | 8040 | 7.346% |

Mapowanie symboli: N ← N,L,R,e,j; S ← A,a,J,S; V ← V,E; F ← F; Q ← /,f,Q. To zachowane grupowanie `mitdb-aami-style-v1`, nie diagnozy pacjenta. N obejmuje część zaburzeń przewodzenia, Q również paced/fusion/unknown. Oryginalne symbole, pozycje i indeksy adnotacji pozostają dostępne. Segmenty bazują na eksperckich pozycjach, bez szukania nowych R-peaków: 90 próbek przed i 162 od adnotacji, przedział `[p-90,p+162)`, 252 próbki. Niepełne okna pominięto w pierwotnym przygotowaniu.

PTB-XL: dokładnie jeden kod spośród **wszystkich 12** kodów rytmu i musi należeć do czterech celów. Zachowano wszystkie pozostałe SCP i likelihood. Brak kodu nie oznacza SR, zero likelihood może oznaczać nieznaną pewność. Nie wybierano najwyższego score, nie nadawano priorytetu nakładającym się rytmom.

| Indeks | Rytm | Zapisy | Pacjenci per klasa | Udział zapisów |
|---|---|---:|---:|---:|
| 0 | SR | 16687 | 14986 | 85.116% |
| 1 | AFIB | 1484 | 1220 | 7.569% |
| 2 | STACH | 805 | 763 | 4.106% |
| 3 | SBRAD | 629 | 608 | 3.208% |

SR = rytm zatokowy, nie NORM ani „zdrowy pacjent”; AFIB = migotanie przedsionków; STACH/SBRAD = odpowiednio tachykardia/bradykardia **zatokowa**, nie wszystkie szybkie/wolne rytmy. Etykiety dotyczą całego 10-sekundowego rekordu. Jedna zapisana etykieta nie dowodzi jednolitego rytmu każdej próbki. Ten sam pacjent może mieć różne rytmy w różnych ECG; sumy osób per klasa przewyższają 17223 unikalnych pacjentów. PAC/PVC i AAMI nie są zamiennikami tych rytmów.

## Minimalny odczyt i gotowość tensorów

```python
from pathlib import Path
import json
from src.prepare_dataset import load_record_shard
from src.ptbxl_rhythm import load_pilot_record

# MIT-BIH: one recording shard, multiple beats
mit_root = Path("data/processed/ecg_dataset_v1")
mit_manifest = json.loads((mit_root / "dataset_metadata.json").read_text())
beat = load_record_shard(mit_root / mit_manifest["shards"][0]["file"])
X_beat, y_beat = beat["X"], beat["y"]  # float32 [n,2,252], int64 [n]
patient_groups = beat["patient_groups"]

# PTB-XL: one 10-second record, add a batch dimension
rhythm_root = Path("data/processed/ptbxl_rhythm_v1")
run = rhythm_root / "runs/90b449bf54b649509519747f417b9e5e"
manifest = json.loads((run / "PTBXL_DOWNLOAD_MANIFEST.json").read_text())
assert manifest["status"] == "complete"
rhythm = load_pilot_record(rhythm_root / manifest["records"][0]["file"])
X_rhythm, y_rhythm = rhythm["X"][None, ...], rhythm["y"].reshape(1)
# float32 [1,12,1000], int64 [1]
patient_id, fold = int(rhythm["patient_id"]), int(rhythm["strat_fold"])
scp = json.loads(str(rhythm["original_scp_codes"]))
```

W środowisku Anny z już zainstalowanym PyTorch:

```python
import torch
X_beat_tensor, y_beat_tensor = torch.from_numpy(X_beat), torch.from_numpy(y_beat)
X_rhythm_tensor, y_rhythm_tensor = torch.from_numpy(X_rhythm), torch.from_numpy(y_rhythm)
```


Przykłady NumPy wykonano i sprawdzono: pierwszy shard MIT-BIH `[2271,2,252]` / y `[2271]`; PTB-XL z osią batch `[1,12,1000]` / y `[1]`. PyTorch nie jest zainstalowany w środowisku preprocessingu, więc samej konwersji tensorów tutaj nie wykonano i niczego nie instalowano. To standardowy krok w środowisku Anny, bez architektury, treningu lub splitu. `from_numpy` współdzieli pamięć — nie modyfikować wczytanych tablic przez tensor. Funkcja `load_pilot_record` jest istniejącym walidatorem pojedynczego NPZ używanym również dla pełnego eksportu, mimo historycznej nazwy.

## Metadane i separacja pacjentów

MIT-BIH zachowuje wyrównane `record_ids`, `patient_groups`, `annotation_positions`, `annotation_indices`, `original_symbols`, `annotation_aux_notes`. Nazwy kanałów, fs, klasy i JSON `preprocessing_config` są metadanymi sharda, nie dodatkowymi przykładami. Jest 48 nagrań i 47 **technicznych grup osób**; rekordy **201 i 202 należą do tej samej grupy** i muszą pozostać razem. Nie są to rzeczywiste dane identyfikacyjne. Pozostałe singletony wynikają z udokumentowanej struktury bazy, nie z niezależnego rozpoznawania osób.

PTB-XL zachowuje `ecg_id`, `patient_id`, `strat_fold`, `rhythm_code`, `original_scp_codes`, pełny JSON `original_metadata`, nazwy odprowadzeń, 100 Hz, źródłowe SHA256 oraz `preprocessing_configuration` z dopasowanymi statystykami. Nie wykryto pacjenta w wielu oficjalnych foldach — zarówno w całych metadanych, jak i wyeksportowanej kohorcie. Oficjalne zalecenie: foldy **1–8 trening, 9 walidacja, 10 test**; tutaj zachowano wyłącznie metadane, nie utworzono splitów. Rozkłady per fold są w CSV wskazanego ukończonego runu i w artefaktach fazy 15.

Anna odpowiada za patient-separated validation: wszystkie zapisy/uderzenia danej osoby pozostają w tej samej części. Nie wykonywać losowego splitu uderzeń MIT-BIH ani okien tego samego pacjenta. ID pacjentów obu baz mają osobne przestrzenie nazw; ich równość nie oznacza wspólnego człowieka. Nie ustalano tożsamości między bazami.

## Preprocessing, kanały i ograniczenia

Oba eksporty: high-pass Butterworth 0,5 Hz, low-pass 45 Hz, order 4, SOS, zero-phase offline; konfiguracje są zapisane w danych. PTB-XL nie używa notch 50/60 Hz przy 100 Hz (Nyquist 50 Hz). Nic nie zmieniano w fazie 15.

Z-score liczony niezależnie na każdym kanale po filtracji: MIT-BIH na całym nagraniu przed segmentacją, PTB-XL na każdym pełnym 10-sekundowym rekordzie. Nie dopasowano wspólnych parametrów z połączonych pacjentów. Jednak normalizacja całego nagrania korzysta również z próbek późniejszych niż bieżące uderzenie — to procedura offline, nie przyczynowa inferencja. Kolejne nowe dane wymagają zgodnego kontekstu okna i kolejności operacji. Z-score usuwa skalę amplitudy, zależy od artefaktów; filtracja zmienia pasmo/morfologię i może powodować efekty brzegowe. Poprawne wymiary i brak NaN nie gwarantują jakości klinicznej. Ewentualna globalna normalizacja wymaga odrębnych parametrów dopasowanych wyłącznie na treningu i osobnej zatwierdzonej wersji danych.

PTB-XL ma stałe 12 odprowadzeń w kolejności I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6; oryginalne jednostki mV, wyjście bezwymiarowe. MIT-BIH ma dwa kanały o różnej anatomii między nagraniami (np. MLII/V1 lub V5/MLII); nie zakładać, że indeks kanału oznacza zawsze to samo odprowadzenie. Patrz `lead_configuration.csv` i nazwy w shardach. Nie harmonizowano ich ani nie zmieniano kolejności. Bazy różnią się też domeną: ambulatoryjne długie ECG kontra spoczynkowe 12-lead 10 s.

## Niezbalansowanie i odpowiedzialność Anny

Dominują N (MIT-BIH) i SR (PTB-XL); najmniejsze klasy to F=802 uderzenia oraz SBRAD=629 ECG/608 pacjentów. Globalne liczby służą opisowi, nie są gotowymi wagami treningu. Wagi, sampling i augmentacje można rozważać dopiero wewnątrz patient-separated treningu; walidacji i testu nie balansować. Powielanie przykładów nie zwiększa różnorodności pacjentów. Accuracy samo może ukrywać słabe rozpoznawanie rzadkich klas; nie obliczano żadnych metryk modelu.

## Reprodukowalność i kontrola

Wszystkie 19605 PTB-XL i 48 shardów MIT-BIH przeszły istniejące loadery, kontrolę dtype/kształtu/skończoności, identyfikatorów, etykiet i SHA256. Zweryfikowano także 39210 źródłowych plików PTB-XL i 676 plików MIT-BIH. Wyniki: `FINAL_ECG_DATA_VALIDATION_REPORT.md`, `results/preprocessing/phase15_final_validation/dataset_validation.json` i `verified_export_checksums.csv`. Wszystkie zapisane hashe kodu odpowiadają obecnym modułom.

Odtworzenie **wyłącznie walidacji**, bez pobierania ani reprocessingu:

```bash
cd ~/Pulpit/ECG-APP
.venv/bin/python -m src.prepare_dataset --verify-only
PYTHONPATH=. .venv/bin/python results/preprocessing/phase15_final_validation/loading_examples.py
.venv/bin/python -m pytest -ra
```

Pełny skrypt audytu zapisuje nowe artefakty wyłącznie w swoim katalogu i używa trybu exclusive-create; ponowne uruchomienie wymaga skopiowania go do **nowego** katalogu audytu, aby nie nadpisać obecnych wyników. Nie uruchamiać eksportu ani downloadera w celu „sprawdzenia” gotowego zbioru.

Sesja fazy 15: **465 passed, 222 warnings**. Wszystkie ostrzeżenia to deprecacje pandas wywoływane przez WFDB, opisane w raporcie; nie zostały wyciszone. Środowisko: WFDB 4.3.1, pandas 3.1.0, NumPy 2.5.4, SciPy 1.18.1, pytest 9.2.0. Przed przyszłą zmianą zależności należy ponowić walidację. Nie wdrożono sieci, trenowania, ewaluacji, aplikacji ani podziałów. Finalna inżynieria i przekazanie tych dwóch wersji danych są zakończone.
