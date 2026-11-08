# Phase 13 — adapter PTB-XL i mały pilot

Status: **adapter zaimplementowany; pilot na rzeczywistych sygnałach zablokowany przez DNS**.
Próby pobrania z oficjalnego PhysioNet zakończyły się `Name or service not known`.
Wybrano 8 zapisów na podstawie rzeczywistych metadanych: pobrano 0 kompletnych zapisów (0 plików), odczytano 0, przetworzono 0, wygenerowano 0 wykresów rzeczywistych EKG. Nie istnieją jeszcze pilotowe tablice NPZ. Katalog `data/processed/ptbxl_rhythm_pilot/` zawiera manifest nieudanej próby, a nie gotowy zbiór danych.

## Dobór zapisów

Ponownie wykorzystano moduł audytu fazy 12. Rekord musi mieć dokładnie jeden kod rytmu spośród **wszystkich 12** rozpoznanych kodów, należący do SR/AFIB/STACH/SBRAD. Brak kodu nie oznacza SR; nakładające się kody nie są automatycznie rozstrzygane. Zachowano pozostałe SCP i ich wartości, w tym zera oznaczające możliwą nieznaną pewność. Wybrano dwa najmniejsze `ecg_id` na klasę, w kolejności SR, AFIB, STACH, SBRAD. Jest to ograniczony test adaptera, bez podziału pacjentów lub balansowania zbioru treningowego. Maksymalnie 8 zapisów; brak automatycznego zastępowania uszkodzonego zapisu innym.

| Kod | ecg_id | patient_id | strat_fold | filename_lr |
|---|---:|---:|---:|---|
| SR | 1 | 15709 | 3 | `records100/00000/00001_lr` |
| SR | 3 | 20372 | 5 | `records100/00000/00003_lr` |
| AFIB | 152 | 9796 | 9 | `records100/00000/00152_lr` |
| AFIB | 282 | 437 | 5 | `records100/00000/00282_lr` |
| STACH | 103 | 13722 | 4 | `records100/00000/00103_lr` |
| STACH | 108 | 11810 | 8 | `records100/00000/00108_lr` |
| SBRAD | 2 | 13243 | 2 | `records100/00000/00002_lr` |
| SBRAD | 12 | 11031 | 8 | `records100/00000/00012_lr` |

Wszystkie cztery klasy występują w **wybranych metadanych**. Żadna nie została jeszcze potwierdzona przez pomyślny odczyt przebiegu.

## Adapter i kontrole

`src/ptbxl_rhythm.py` korzysta z istniejących `ptbxl_metadata`, `filter_ecg` i `ECGNormalizer`. Nie zmienia modułów MIT-BIH.
Odczyt przez WFDB używa dokładnie `filename_lr`. Sprawdzane są: tożsamość numeru pliku i nagłówka, 100 Hz, 1000 próbek, 12 kanałów, jednostki mV, oryginalna kolejność `I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6`, kalibracja, rozmiar i sumy kontrolne danych oraz skończoność wartości. Przyjęty ścisły format pilota: jeden wspólny plik format-16, 24000 bajtów. Inne formaty lub kolejności są zgłaszane, nigdy nie są potajemnie konwertowane.
Pacjent i fold pochodzą z niezmienionego wiersza metadanych; nagłówek nie zapewnia niezależnego potwierdzenia tożsamości pacjenta. Kontrola zgodności plików nie jest weryfikacją biometryczną.

## Konfiguracja przetwarzania

- Butterworth SOS: high-pass 0,5 Hz i low-pass 45 Hz, rząd 4 każdego filtru; filtrowanie w przód i wstecz o zerowej fazie, wyłącznie offline.
- Przy 100 Hz Nyquist wynosi 50 Hz. Cutoff 45 Hz jest poprawny matematycznie, ale blisko Nyquista; nie odzyska informacji utraconej przy próbkowaniu. Nie dodano notch: 50 Hz leży na Nyquiście, 60 Hz poza nim. Adapter jawnie odrzuca włączenie notch w tym pilocie.
- Z-score po filtrowaniu: osobne średnie i odchylenia dla każdego odprowadzenia każdego 10-sekundowego zapisu, `ddof=0`, epsilon `1e-8`. Dokładne statystyki trafiają do eksportu. Nie dopasowuje się wspólnych parametrów na pacjentach z różnych foldów.
- Przyszłe przetwarzanie nowych zapisów musi używać identycznego okna, częstotliwości, kolejności operacji i normalizacji. Normalizacja usuwa bezwzględną skalę amplitudy; zależy od całego okna i artefaktów. Filtrowanie ma efekty brzegowe. Wizualna kontrola pozostaje konieczna.

W przeciwieństwie do MIT-BIH: całe 10 sekund zamiast segmentów 252-próbkowych wokół uderzeń, 12 zamiast 2 kanałów, 100 zamiast 360 Hz, normalizacja pojedynczego krótkiego zapisu zamiast długiego nagrania. Etykiety dotyczą rytmu całego zapisu; nie są mapowaniem AAMI uderzeń.

## Format i pliki

Po udanym odczycie każdy zapis trafi do `recordings/<ecg_id:05d>_rhythm.npz` w osobnym katalogu pilota. `X`: float32 `[12,1000]`; `y`: skalar int64, SR=0, AFIB=1, STACH=2, SBRAD=3. Eksport zachowuje `ecg_id`, `patient_id`, `strat_fold`, kod rytmu, wszystkie pierwotne SCP i wartości, surową komórkę SCP, metadane jakości, nazwy kanałów, 100 Hz, konfigurację ze statystykami i SHA256 źródeł. Odczyt `np.load(..., allow_pickle=False)` lub `load_pilot_record(path)`; pola JSON są tekstem Unicode, bez pickle.

Nowe pliki implementacji: `src/ptbxl_rhythm.py`, `examples/prepare_ptbxl_pilot.py`, `tests/test_ptbxl_rhythm.py`.
Dokumentacja i manifest: `PTBXL_PREPROCESSING_CONFIG.json`, `PTBXL_PILOT_MANIFEST.json`, niniejszy raport i `PHASE14_FULL_PTBXL_EXPORT_PLAN.md`.
Artefakty wykonania: `results/preprocessing/phase13_ptbxl_pilot/`; manifest wyjściowy: `data/processed/ptbxl_rhythm_pilot/pilot_manifest.json`. Docelowe wykresy: podkatalog `plots/` artefaktów, po jednym na klasę, 300 dpi, odprowadzenia I, II, V1 i V5, zsynchronizowane osie czasu 0–10 s, kolumny raw mV / filtered mV / Z-score bez jednostki. Obecnie brak takich wykresów. Nie dokonano oceny morfologii ani interpretacji diagnostycznej.

## Brakujące pliki i wznowienie

Każdy poniższy plik ma URL `https://physionet.org/files/ptb-xl/1.0.3/` + ścieżka i lokalizację `data/ptbxl/` + ścieżka:

- `records100/00000/00001_lr.hea`
- `records100/00000/00001_lr.dat`
- `records100/00000/00003_lr.hea`
- `records100/00000/00003_lr.dat`
- `records100/00000/00152_lr.hea`
- `records100/00000/00152_lr.dat`
- `records100/00000/00282_lr.hea`
- `records100/00000/00282_lr.dat`
- `records100/00000/00103_lr.hea`
- `records100/00000/00103_lr.dat`
- `records100/00000/00108_lr.hea`
- `records100/00000/00108_lr.dat`
- `records100/00000/00002_lr.hea`
- `records100/00000/00002_lr.dat`
- `records100/00000/00012_lr.hea`
- `records100/00000/00012_lr.dat`

Gotowy skrypt zawiera dokładne polecenia curl tylko dla tych 16 plików i pomija pliki już istniejące. W terminalu VS Code z dostępem do sieci:

```bash
cd ~/Pulpit/ECG-APP
bash results/preprocessing/phase13_ptbxl_pilot/download_selected_pilot.sh
.venv/bin/python -m examples.prepare_ptbxl_pilot --no-download \
  --output-dir data/processed/ptbxl_rhythm_pilot_verified \
  --artifact-dir results/preprocessing/phase13_ptbxl_pilot_verified \
  --report-dir results/preprocessing/phase13_ptbxl_pilot_verified
.venv/bin/python -m pytest
```

Nowe ścieżki zachowują wynik pierwotnej próby. Kolejne uruchomienie wymaga kolejnych wolnych ścieżek. Uszkodzony istniejący plik wymaga świadomego przeglądu; program go nie nadpisuje. Nie pobiera całej bazy. W środowisku bez sieci kod nie zastępuje danych sygnałami syntetycznymi.

## Weryfikacja

Testy adaptera wykorzystują jawnie syntetyczne tymczasowe rekordy WFDB wyłącznie do sprawdzenia algorytmów, nie jako wyniki pilota. Sprawdzają selekcję i wykluczanie wieloetykietowych rytmów, zgodność metadanych, format i kolejność kanałów, checksum, braki/uszkodzenia, 100 Hz i Nyquista, skończoność, deterministyczność, niezmienność wejścia, eksport/odczyt i osie wykresów. Integracja z rzeczywistymi przebiegami jest pomijana, dopóki brakuje plików. Wynik pełnej sesji i kontrola zachowania wcześniejszych plików: `results/preprocessing/phase13_ptbxl_pilot/final_verification.json`.

## Źródła i ograniczenia

[Oficjalny PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), [katalog plików 1.0.3](https://physionet.org/files/ptb-xl/1.0.3/), [przykładowy oficjalny nagłówek](https://physionet.org/content/ptb-xl/1.0.3/records100/00000/00001_lr.hea). Lokalne metadane i wyniki fazy 12 są źródłem selekcji; ich SHA256 zapisano w manifeście. Nagłówek przykładowego rekordu wspiera założenia formatu, ale nie dowodzi poprawności wszystkich ośmiu zapisów. Pilot ośmiu ECG nie reprezentuje całej populacji ani jakości bazy. Do zakończenia części empirycznej fazy 13 potrzebny jest rzeczywisty odczyt wszystkich wybranych plików i przegląd wykresów. Pełnego eksportu i prac Anny nie rozpoczęto.

Pełna sesja `.venv/bin/python -m pytest`: **437 passed, 1 skipped**, 438 testów, 81,22 s; 174 ostrzeżenia deprecacyjne zależności WFDB/pandas. Test rzeczywistej integracji pominięto z powodu brakujących waveformów. Kontrola SHA256: 121 wcześniejszych plików bez zmian; 676 plików źródłowych MIT-BIH zgodnych z manifestem fazy 10. Szczegóły w `final_verification.json`.
