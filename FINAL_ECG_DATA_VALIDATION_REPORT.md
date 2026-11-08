# Finalna walidacja danych ECG — faza 15

**Wynik: oba eksporty kompletne i zgodne z manifestami. Dane gotowe do przekazania Annie jako dwa oddzielne zadania.** Zakres: wyłącznie odczyt, weryfikacja i dokumentacja. Nie pobierano danych, nie wykonywano ponownego preprocessingu produkcyjnego ani eksportu i nie zmieniano danych ani istniejących modułów. Audyt odczytywał gotowe tablice; dotychczasowe testy nadal sprawdzały istniejące funkcje preprocessingu w swoim zakresie.

## Potwierdzone dane

| Właściwość | MIT-BIH heartbeat | PTB-XL rhythm |
|---|---|---|
| Przykłady | 109452 uderzenia | 19605 ECG |
| Nagrania | 48 | 19605 |
| Osoby/grupy | 47 technicznych grup | 17223 patient_id |
| Logiczny X | float32 `[109452,2,252]` | float32 `[19605,12,1000]` |
| y | int64 0–4 | int64 0–3 |
| fs | 360 Hz | 100 Hz |
| Kontrola eksportu SHA256 | 48/48 shardów | 19605/19605 rekordów |
| Kontrola plików źródłowych SHA256 | 676/676 | 39210/39210 |

Ukończony run PTB-XL: `data/processed/ptbxl_rhythm_v1/runs/90b449bf54b649509519747f417b9e5e/PTBXL_DOWNLOAD_MANIFEST.json`; SHA256 manifestu: `ec81bfdfcaf8037591f65fa8999ae1f11a4c4d2930b4346fbf9a0bd1e3c6e9e1`. Status complete, selected=successful=19605, failed=0, not_attempted=0. Starsza niepełna próba pozostaje historyczna; nie jest źródłem niniejszych wyników.

MIT-BIH: N=90594, S=2781, V=7235, F=802, Q=8040 — razem 109452, zgodnie z zapisanym manifestem i fazą 9/10. Odczytano wszystkie wyrównane pola per uderzenie. Nie ma powtórzonych indeksów adnotacji w obrębie nagrania; kolejność adnotacji zachowana. Symbole i pozycje porównano z oryginalnymi .atr, kanały/fs z nagłówkami, metadane grup z patient_groups.json. Potwierdzono wspólną grupę 201/202 istniejącym walidatorem; wszystkie źródła i artefakty są zgodne z SHA256 zapisanymi przy eksporcie.

PTB-XL — rozkład końcowy zgodny z niezależnie odtworzoną kohortą fazy 12:

| Indeks | Rytm | Zapisy | Pacjenci per klasa | Udział zapisów |
|---|---|---:|---:|---:|
| 0 | SR | 16687 | 14986 | 85.116% |
| 1 | AFIB | 1484 | 1220 | 7.569% |
| 2 | STACH | 805 | 763 | 4.106% |
| 3 | SBRAD | 629 | 608 | 3.208% |

Nie sumować pacjentów per klasa jako liczby osób całego zbioru. Oficjalne foldy są zachowane; konfliktów pacjent–fold: 0. CSV rozkładu klas i foldów ukończonego runu porównano z policzonym eksportem. Nie tworzono nowych splitów.

## Integralność PTB-XL

Każdy NPZ: istniejący `load_pilot_record`, float32 `[12,1000]`, skalar int64 y, skończone X, oryginalny ecg_id/patient_id/strat_fold i wszystkie SCP dokładnie zgodne z lokalnymi CSV. Zbiór ID jest równy kohorcie: bez braków, nadmiarowych NPZ, podwójnych ID i powtórzonych ścieżek. Kolejność manifestu rosnąca po ecg_id. Nagłówki źródeł potwierdzają fs, 1000 próbek, 12 kanałów, nazwy, ich kolejność i mV. SHA256 źródeł zgadzają się z manifestem i NPZ. Filtry/normalizacja każdego pliku są identyczne z fazą 13; statystyki należą do danego ecg_id i scope recording. Nie przeliczano waveformów.

Nie wykryto brakujących eksportów, niepoprawnych etykiet ani niespójnych metadanych. SHA256 potwierdza zgodność z zapisanym eksportem, nie kliniczną prawdziwość rozpoznania. Nie przeprowadzano niezależnej oceny diagnostycznej ani benchmarku modeli. Audyt offline potwierdza zgodność źródeł z zapisanymi hashami, ale nie sprawdzał zdalnych podpisów ani autentyczności serwera.

## Audyt 222 ostrzeżeń pytest

Kategoria wszystkich ostrzeżeń: **Pandas4Warning**, pochodzące z WFDB 4.3.1 przy pandas 3.1.0. Przyczyna: użycie deprecated argumentu `inplace=True` w API DataFrame; komunikat zapowiada usunięcie w przyszłej wersji pandas. Audyt oparto na pełnym logu pytest i odczycie wskazanych miejsc kodu zależności.

| Miejsce w WFDB | Operacja / przyczyna | Liczba |
|---|---|---:|
| annotation.py:3062 | set_index(inplace=True), import tabeli klas adnotacji | 1 |
| annotation.py:3232 | set_index(inplace=True), import tabeli etykiet adnotacji | 1 |
| annotation.py:1472 | set_index(inplace=True), konwersja adnotacji w testach | 130 |
| _header.py:172 | drop(inplace=True), zapis nagłówków fixture'ów WFDB | 90 |
| Razem | Jedna rodzina deprecacji, cztery miejsca | 222 |

Ostatnie 90 = 48 ostrzeżeń testów bulk + 42 testów adaptera; poprzednie 130 pochodzi z testów odczytu/segmentacji/eksportu MIT-BIH i powiązanych modułów. Dwa ostrzeżenia importu występują raz każde. Brak w logu ostrzeżeń numerycznych NaN/overflow lub ostrzeżeń o niepoprawnych etykietach. Testy dodatkowo jawnie sprawdzają wartości skończone.

**Wymagane działanie utrzymaniowe:** monitorować zgodność WFDB z przyszłym pandas, zachować wersje zweryfikowanego środowiska i przetestować aktualizacje przed ich użyciem. Obecnie nie ma podstaw do korekty danych ani zmiany preprocessingu z powodu tych komunikatów. Nie edytowano site-packages, nie usuwano ostrzeżeń, nie dodawano filterwarnings ani globalnego wyciszania. Nie twierdzimy, że ostrzeżenia można ignorować przy przyszłej zmianie zależności. Szczegóły maszynowe: warnings_audit.json i warnings_audit.csv.

## Loadery, testy i zachowanie wcześniejszej pracy

Wykonano `load_record_shard` dla wszystkich shardów MIT-BIH oraz `load_pilot_record` dla wszystkich plików PTB-XL. Przykłady NumPy w `loading_examples.py` przeszły; pokazują `[2271,2,252]` i `[1,12,1000]` po dodaniu batch. PyTorch nie jest zainstalowany: przykład standardowego `torch.from_numpy` znajduje się w handoff, ale konwersji nie wykonano i nie instalowano zależności.

Pełna sesja `.venv/bin/python -m pytest -ra`: **465 passed, 222 warnings**, 0 failed, 0 skipped; 135.78 s. Testy na kontrolowanych fixture'ach weryfikują mechanikę kodu; osobny audyt w tej fazie weryfikuje wszystkie rzeczywiste eksporty, nie zastępuje ich fixture'ami.

Przed/po audycie porównano size/mtime_ns **98812 istniejących plików** (eksporty, źródła, metadane, kod i wcześniejsze raporty): bez zmian. Dodatkowo każdy wyeksportowany NPZ i źródła porównano z SHA256 zapisanymi w oryginalnych manifestach. Wszystkie zapisane hashe kodu odpowiadają obecnym modułom.

## Nowe dokumenty i artefakty

- ECG_DATA_HANDOFF_FOR_ANNA.md — wspólny kontrakt, przykłady, grupy pacjentów, foldy, normalizacja i ograniczenia.
- FINAL_ECG_DATA_VALIDATION_REPORT.md — niniejszy raport.
- results/preprocessing/phase15_final_validation/: verify_readonly.py, dataset_validation.json, verified_export_checksums.csv (19653 eksporty), preservation_snapshot.json, validation.log, pytest.log, warnings_audit.json/csv, ptbxl_class_distribution.json, ptbxl_patient_fold_distribution.json, loading_examples.py/log, final_summary.json.

Dokumenty fazy 15 wskazują ukończony run i zastępują historyczny stan opisany w raportach checkpoint A/B; samych wcześniejszych raportów ani plików danych nie nadpisano. Dalsze obowiązki Anny: architektury, dobór patient-separated validation, trening, ewaluacja i Streamlit. Nie rozpoczęto tych prac.
