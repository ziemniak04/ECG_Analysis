# PTB-XL — środowisko i protokół treningu, faza 16

**Stan: kod przygotowany; wykonanie modelu zablokowane przez brak PyTorch. Pełny trening nie został rozpoczęty.** W tej fazie wolno wykonać tylko jeden techniczny krok optymalizatora, bez epok, doboru hiperparametrów ani pomiaru końcowego testu.

## Dane i istniejący eksport

Katalog `data/processed/ptbxl_rhythm_v1/`. Ukończony manifest: `runs/90b449bf54b649509519747f417b9e5e/PTBXL_DOWNLOAD_MANIFEST.json`. Starszy run `4760b72c29934d1ab406cf7b9ded933a` jest historycznie niepełny i nie nadaje się do użycia. Gotowy eksport: 19605 rekordów, 17223 pacjentów, klasy SR/AFIB/STACH/SBRAD (0/1/2/3). Jeden plik `recordings/<ecg_id:05d>_rhythm.npz` na ECG, X float32 `[12,1000]`, y int64 skalar, 100 Hz.

`PTBXLExportIndex` korzysta z faktycznego manifestu i istniejącego `load_pilot_record`. Sprawdza kompletność i kolejność, ID i ścieżki, obecność wszystkich NPZ, brak duplikatów, regułę kohorty i zgodność pacjentów/foldów/etykiet z istniejącym audytem oryginalnych CSV. Wymaga niezmienionej dataset identity i konfiguracji fazy 13. Przy każdym odczycie weryfikuje SHA256, kształt, dtype, skończoność oraz metadane rekordu i zapisane parametry preprocessingu. Błąd ma kontekst ecg_id i ścieżki; rekord nie jest po cichu usuwany lub naprawiany.

W RAM są metadane/indeksy, nie wszystkie sygnały. `PTBXLRhythmDataset` ma zerowy cache waveformów i ładuje jeden NPZ per __getitem__. Wynik jest słownikiem X/y/ecg_id/patient_id/strat_fold. Domyślny DataLoader collate tworzy X `[B,12,1000]`, y torch.long `[B]` i tablice ID. `num_workers=0` jest bezpiecznym początkiem na CPU; większa liczba workerów oznacza dodatkowy koszt pamięci metadanych i I/O.

## Instalacja wyłącznie w .venv

Rozpoznane środowisko: Linux x86_64, Python 3.12.3, glibc 2.39. PyTorch brak. Próba pip CPU zakończyła się bez pobrania; curl potwierdził `Could not resolve host: download.pytorch.org`. Nie wykonywano kolejnych prób instalacji ani zmian systemowego Python. GPU nie został zweryfikowany. Zgodnie z [oficjalną instrukcją PyTorch](https://pytorch.org/get-started/locally/) dla Linux bez potwierdzonego GPU wybieramy CPU.

W terminalu VS Code z dostępem do sieci:

```bash
cd ~/Pulpit/ECG-APP
# Zachowuje wszystkie obecne, niezatwierdzone zmiany; nie ma reset/force-push.
git switch -c feature/ptbxl-1dcnn
.venv/bin/python -m pip install -r requirements_model_cpu.txt
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.rand(2, 3)); print(torch.cuda.is_available())"
.venv/bin/python -m examples.phase16_model_smoke --output-dir results/model_setup/phase16_verified
.venv/bin/python -m pytest -ra
```

Gałęzi nie udało się utworzyć w środowisku agenta: metadane `.git` są dostępne tylko do odczytu, `git switch -c` zgłosił cannot lock ref / unable to create directory. Początkowy main był czysty; istniejące zmiany nie zostały odrzucone. Polecenie branch należy wykonać raz w terminalu z prawem zapisu do Git, o ile gałąź jeszcze nie istnieje. Prace pozostają widoczne jako zmiany robocze na main; nie wykonano commita ani push.

Plik requirements_model_cpu.txt wybiera oficjalny CPU index i zakres torch>=2.7,<3. To zakres instalacyjny, nie potwierdzona wersja runtime. Po udanej instalacji wynik smoke zapisze faktyczną wersję; przed fazą 17 należy zachować dokładne wersje środowiska. Nie potrzebujemy torchvision/torchaudio. Nie aktualizować pakietów preprocessingu bez osobnej walidacji. Nie instalować PyTorch globalnie ani przez sudo.

Wyniki bieżącej próby: `results/model_setup/phase16/partition_protocol.json`, `smoke_result.json`, logi pip, DNS, pytest. Bieżący wynik smoke ma **blocked_missing_pytorch, optimizer_updates=0**, a nie passed. Nowy katalog phase16_verified zachowuje wcześniejszy wynik i zapobiega nadpisywaniu raportów.

## Oficjalne partycje — już zweryfikowane metadane

| Partycja | Oryginalne foldy | ECG | Pacjenci |
|---|---|---:|---:|
| Train | 1–8 | 15689 | 13749 |
| Validation | 9 | 1958 | 1763 |
| Final test | 10 | 1958 | 1711 |

Brak przecinania pacjentów i indeksów. Każdy pacjent ma dokładnie jeden oryginalny strat_fold. Indeksy i ECG ID są zapisane deterministycznie, rosnąco po ecg_id, w partition_protocol.json wraz z SHA256 manifestu. Nie użyto losowego splitu. Etykiety validation/test nie służą obliczaniu wag, doborowi architektury, progu ani hiperparametrów. Smoke czyta tylko training folds; final test nie jest oceniany.

## Minimalny Dataset i DataLoader po instalacji

```python
from torch.utils.data import DataLoader
from src.ptbxl_protocol import PTBXLExportIndex
from src.ptbxl_torch import PTBXLRhythmDataset
from src.models.ptbxl_cnn import PTBXLRhythmCNN

index = PTBXLExportIndex()
train = PTBXLRhythmDataset(index, partition="train")
loader = DataLoader(train, batch_size=4, shuffle=False, num_workers=0)
batch = next(iter(loader))
model = PTBXLRhythmCNN()
logits = model(batch["X"])  # oczekiwane [4,4]; nie są to prawdopodobieństwa
# batch["y"]: torch.long [4]
# batch["ecg_id"], batch["patient_id"], batch["strat_fold"] pozostają dostępne.
```

Nie używać Dataset(partition=None) jako przypadkowego treningu na całej kohorcie. Parametr None daje dostęp do kompletnego indeksu do inspekcji, a nie automatyczny trening. Dataset(partition="test") tworzy dostęp zgodny z protokołem, ale nie jest używany przez smoke.

## Niezbalansowanie — tylko training folds

| Klasa | Training ECG | Opcjonalna waga balanced |
|---|---:|---:|
| SR | 13354 | 0.29371348 |
| AFIB | 1189 | 3.29878044 |
| STACH | 643 | 6.09992218 |
| SBRAD | 503 | 7.79771376 |

Wzór: n_train/(4*n_class). `training_class_weights(index, index.partitions.train)` wymaga jawnych unikalnych indeksów należących wyłącznie do foldów 1–8. Odrzuca walidację/test, pełny zakres datasetu, duplikaty, puste dane i brak klasy. Zwraca NumPy float32; opcjonalna konwersja dla loss:

```python
import torch
from src.ptbxl_protocol import training_class_weights
weights = torch.from_numpy(training_class_weights(index, index.partitions.train))
criterion = torch.nn.CrossEntropyLoss(weight=weights)  # na tym samym device co logits
```

Domyślny smoke używa loss bez wag; --class-weights włącza wyłącznie powyższe wagi treningowe. Nie wyznaczano wag z globalnych liczebności i nie stosowano oversamplingu/undersamplingu. Wagi są propozycją opcji, nie zweryfikowanym wynikiem poprawy jakości.

## Smoke i ograniczenia normalizacji

Po instalacji skrypt ma wykonać jedną partię CPU (B=4, seed=42, bez shuffle), model.train, CrossEntropyLoss, backward, kontrolę skończonych gradientów i dokładnie jeden AdamW.step (lr=1e-3, weight_decay=1e-4). Nie ma pętli epok. Nie zapisuje wag ani checkpointu. Loss smoke jest wyłącznie kontrolą numeryczną, nie wynikiem jakości. Testy modelu mają forward/backward, bez dodatkowego optimizer.step. Ten etap nie został wykonany w bieżącym środowisku z powodu braku PyTorch.

Dane pozostają per-record/per-lead Z-score po filtracji 0,5–45 Hz bez notch. Nie dopasowujemy żadnego nowego scalera. Normalizacja korzysta z całego okna i usuwa bezwzględną skalę; pozostaje procedurą offline. Nie przeliczać danych w Dataset i nie mieszać PTB-XL rhythm z beat-level MIT-BIH/AAMI.

## Proponowana faza 17 — plan, bez wykonania

Najpierw zainstalować PyTorch i zaliczyć wszystkie pominięte testy tensorowe oraz realny smoke. Następnie osobno zatwierdzić fazę 17: zamrozić wersje, seed, konfigurację i SHA256 manifestu; ustalić limit epok i early stopping wyłącznie na foldzie 9; trenować tylko na 1–8. Walidację prowadzić z model.eval() i torch.no_grad(), aby nie aktualizować statystyk BatchNorm ani nie pozostawiać aktywnego Dropout. Wagi klas, jeśli użyte, liczyć tylko z treningu. Zapisywać konfigurację/stan optymalizatora/seedy i przebieg treningu w ignorowanych katalogach. Dobór architektury, epoki i ewentualnej kalibracji prowadzić na walidacji. Fold 10 zachować do jednorazowej końcowej oceny po zamrożeniu wszystkich decyzji, w osobnej uzgodnionej fazie. Streamlit i integracja wyników należą przede wszystkim do Anny. Nie uruchomiono fazy 17 ani modeli MIT-BIH.
