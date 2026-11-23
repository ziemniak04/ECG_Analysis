# Faza 16 — PTB-XL: środowisko, Dataset i pierwsza architektura 1D-CNN

**Kod fazy 16 przygotowany; runtime PyTorch i smoke test nie zostały wykonane z powodu brakującego pakietu i DNS. Projekt nie jest jeszcze zweryfikowany do rozpoczęcia fazy 17.** Nie ma wyników accuracy ani innego pomiaru jakości modelu. Nie przeprowadzono pełnego treningu ani oceny finalnego testu.

## Rzeczywisty eksport i zakres inspekcji

Użyto wyłącznie `data/processed/ptbxl_rhythm_v1/` i ukończonego runu `90b449bf54b649509519747f417b9e5e`. Manifest: `runs/90b449bf54b649509519747f417b9e5e/PTBXL_DOWNLOAD_MANIFEST.json`; SHA256 `ec81bfdfcaf8037591f65fa8999ae1f11a4c4d2930b4346fbf9a0bd1e3c6e9e1`. Potwierdzono 19605 wpisów, 17223 pacjentów, brak failed/not_attempted, rosnącą kolejność ecg_id, unikalne ścieżki/ID i obecność wszystkich NPZ. Etykiety i patient_id/strat_fold porównano z oryginalnymi CSV i istniejącą regułą wyłącznego kodu spośród wszystkich 12 rhythm SCP. Klasy: SR=0, AFIB=1, STACH=2, SBRAD=3.

Nowy indeks przechowuje tylko metadane. Nie ładuje całej tablicy X do RAM. Istniejący reader NPZ zapewnia float32 `[12,1000]`, int64 y, 100 Hz, oryginalne kanały i wartości skończone. Na dostępie rekord jest sprawdzany względem SHA256, etykiety, pacjenta/fold i zapisanej konfiguracji. Błędy są jawne, próbki nie są usuwane ani zastępowane.

W tej fazie wykonano niezależny od PyTorch odczyt NumPy czterech rzeczywistych **treningowych** ECG: SR=1, AFIB=282, STACH=103, SBRAD=2. Każdy ma `[12,1000]`, float32, int64 y i wartości skończone; metadane i SCP są zgodne z eksportem/CSV. Nie powtarzano pełnego odczytu wszystkich waveformów ani preprocessingu — pełna walidacja eksportów jest udokumentowana w fazie 15, a nowy Dataset będzie sprawdzał każdy NPZ przy odczycie. Aktualna weryfikacja tensorów pozostaje niewykonana.

## Oficjalny protokół pacjentów

| Partycja | Oryginalne foldy | ECG | Pacjenci |
|---|---|---:|---:|
| train | 1,2,3,4,5,6,7,8 | 15689 | 13749 |
| validation | 9 | 1958 | 1763 |
| test | 10 | 1958 | 1711 |

Wszystkie rekordy są pokryte rozłącznie. Nie ma pacjenta w kilku foldach ani nakładania między train/validation/test. To deterministyczny wybór na podstawie istniejącego strat_fold, bez losowego dzielenia. Indeksy i ecg_id każdej partycji zapisano w `partition_protocol.json` wraz z checksum manifestu. Sama inspekcja metadata test fold nie jest oceną modelu; etykiety testowe nie służyły żadnej decyzji treningowej lub doborowi wag.

Training-only class counts i opcjonalne wagi n_train/(4*n_class):

| Klasa | Training ECG | Waga balanced |
|---|---:|---:|
| SR | 13354 | 0.29371348 |
| AFIB | 1189 | 3.29878044 |
| STACH | 643 | 6.09992218 |
| SBRAD | 503 | 7.79771376 |

API odrzuca indeksy walidacji/testu, pełnej kohorty, puste/niecałkowite indeksy, powtórzenia i brak klasy. Wagi są opcjonalne (domyślny smoke bez wag); nie stosowano oversamplingu, undersamplingu ani augmentacji.

## Model

`PTBXLRhythmCNN`: Conv1d 12→32 k15 + BN + ReLU + MaxPool2; 32→64 k9 + BN + ReLU + MaxPool2; 64→128 k5 + BN + ReLU + MaxPool2; AdaptiveAvgPool1d(1); Linear128→64 + ReLU + Dropout0.3 + Linear64→4. Wejście `[B,12,1000]`, projektowane wyjście `[B,4]` surowych logitów. Brak Softmax w modelu i brak wag pretrained.

**Obliczona liczba parametrów trenowalnych: 74116.** Potwierdzona arytmetyką konfiguracji i testem niezależnym od PyTorch, ale nie runtime nn.Module w tym środowisku. Konstruktor modelu ma sprawdzenie zgodności faktycznego licznika po instalacji. Lokalne cechy przed poolingiem mają pole recepcyjne ~0,54 s; globalne uśrednianie agreguje 10 s, tracąc część kolejności zdarzeń. Nie wykazano dotąd skuteczności architektury w klasyfikacji rytmów.

Model, Dataset i protokół nie zmieniają filtracji/normalizacji. Korzystają z istniejącego per-record/per-lead Z-score po filtrach 0,5–45 Hz, bez notch przy fs=100 Hz. Nie przeliczono sygnałów i nie dopasowano globalnego scalera.

## Środowisko i blokady

Interpreter `/home/madzia/Pulpit/ECG-APP/.venv/bin/python`, Python 3.12.3, Linux x86_64/glibc 2.39. PyTorch brak. Instalacja tylko w .venv, CPU, z `https://download.pytorch.org/whl/cpu`, zakończyła się bez pobrania listy wersji; curl potwierdził `Could not resolve host: download.pytorch.org`. Zatrzymano próby instalacji. Nie ma potwierdzonego kompatybilnego GPU; nvidia-smi i /dev/nvidia* nie były dostępne. Nie zakładano CUDA.

`smoke_result.json`: **blocked_missing_pytorch; smoke_executed=false; optimizer_updates=0**. Nie potwierdzono rzeczywistych tensorów, logitów, loss, gradientów ani zmiany parametrów. Skrypt po instalacji ma wykonać dokładnie jeden krok CPU na małej partii treningowej, sprawdzić skończoność i kształty oraz niczego nie zapisywać jako trained checkpoint. Wymaga osobnego nowego katalogu wyników.

Git początkowo był czysty na main. Próba `git switch -c feature/ptbxl-1dcnn` została zablokowana zapisem do `.git` (cannot lock ref / unable to create directory; metadane Git tylko do odczytu w środowisku agenta). Gałąź nie została utworzona; prace pozostają niezatwierdzone na main. Nie resetowano istniejącej pracy i nie wykonywano commit/push. Terminal VS Code z prawem zapisu do Git może utworzyć gałąź, zachowując wszystkie te zmiany.

## Testy i ochrona danych

Pełna sesja `MPLCONFIGDIR=/tmp/ecg_phase16_mpl .venv/bin/python -m pytest -ra`: **505 passed, 1 skipped, 229 warnings**, 201,85 s. 478 dotychczasowych testów (w tym 13 wizualnych) + 27 nowych kontroli protokołu. Pytest pomija jeden **cały moduł** `tests/test_ptbxl_torch.py` na etapie kolekcji z powodu braku torch; zawiera on pięć planowanych przypadków tensorów/forward/backward. Nie zaliczono ich jako passed. Ostrzeżenia są deprecacjami WFDB/pandas, nie zostały wyciszone. Zmienna MPLCONFIGDIR kieruje cache do dozwolonego /tmp, nie zmienia obliczeń ani ustawień warnings.

Zaliczone kontrole: format NPZ, leniwy odczyt NumPy, kształt/dtype, cztery etykiety, ID i foldy, źródłowe metadane rzeczywistej kohorty, deterministyczne indeksy, separacja pacjentów, błędy brakujących/uszkodzonych plików, walidacja NaN, training-only wagi i odrzucenie leakage. `compileall` sprawdził składnię nowych modułów, a wersjonowany JSON konfiguracji jest zgodny z domyślną konfiguracją Python. Forward/backward/optimizer są przygotowane, ale nadal wymagają wykonania PyTorch.

Porównano SHA256 wcześniejszego kodu, raportów i manifestów z checkpointem tej fazy; poza celowo rozszerzonym `.gitignore` bez zmian. Zaobserwowano zero zmian w manifestach MIT-BIH i PTB-XL. Nowy kod ma wyłącznie read-only dostęp do eksportów. Sprawdzone cztery realne NPZ mają SHA256 zgodne z manifestem. Nie twierdzimy, że w tej fazie ponownie zweryfikowano sumy wszystkich 19605 plików — Dataset weryfikuje je na żądanie, a pełny audyt należy do fazy 15.

`.gitignore` wyklucza checkpoints/, runs/, results/training/, results/model_setup/, cache i pliki *.pt/*.pth/*.ckpt. `git check-ignore` potwierdził reguły; brak śledzonych NPZ/DAT/checkpointów. Nie zmieniono wcześniejszych reguł ani plików requirements preprocessingu.

## Nowe pliki

- `src/ptbxl_protocol.py`: manifest, indeks, kontrola metadata/pacjentów, oficjalne partycje i training-only counts/weights.
- `src/ptbxl_torch.py`: Dataset z X/y i ID, bez cache całego zbioru.
- `src/ptbxl_model_config.py`, `src/models/__init__.py`, `src/models/ptbxl_cnn.py`: baseline i arytmetyka konfiguracji.
- `configs/ptbxl_cnn_v1.json`, `requirements_model_cpu.txt`.
- `examples/phase16_model_smoke.py`: audyt i jeden opcjonalny optimizer step; brak pętli epok.
- `tests/test_ptbxl_protocol.py`, `tests/test_ptbxl_torch.py`.
- `PTBXL_MODEL_ARCHITECTURE.md`, `PTBXL_TRAINING_SETUP.md`, niniejszy raport.
- Ignorowane artefakty: `results/model_setup/phase16/` (logi instalacji/DNS, protocol, smoke_result, logi testów, preservation i final_verification).

## Dokładne polecenia po przywróceniu sieci

```bash
cd ~/Pulpit/ECG-APP
# Tylko raz, jeśli gałąź jeszcze nie istnieje:
git switch -c feature/ptbxl-1dcnn
.venv/bin/python -m pip install -r requirements_model_cpu.txt
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.rand(2, 3)); print(torch.cuda.is_available())"
.venv/bin/python -m examples.phase16_model_smoke --output-dir results/model_setup/phase16_verified
.venv/bin/python -m pytest -ra
```

Procedura CPU odpowiada [oficjalnej instrukcji PyTorch dla Linux bez zweryfikowanego GPU](https://pytorch.org/get-started/locally/). Nie używać sudo ani systemowego pip. Jeśli katalog phase16_verified już istnieje, podać kolejny nowy katalog. Prawidłowy wynik smoke musi mieć status passed, optimizer_updates=1 i kształt logits [4,4]. Dopiero po zaliczeniu tych kontroli można rozważyć osobno zatwierdzoną fazę 17. Teraz gotowy jest kod i protokół, a blokadą wykonania pozostaje środowisko PyTorch/Git. Nie rozpoczęto treningu, ewaluacji finalnego testu, MIT-BIH CNN ani Streamlit.
