# Aktualizacja checkpoint B — dataset jeszcze niekompletny

**Pełny zbiór nie jest jeszcze gotowy do pracy Anny.** Zatwierdzone pobieranie zostało przerwane przez błąd DNS S3. W `data/processed/ptbxl_rhythm_v1/recordings/` są obecnie trzy rzeczywiste, zweryfikowane NPZ: ECG 1, 2 i 3. Częściowe liczby: SR=2, AFIB=0, STACH=0, SBRAD=1; 3 pacjentów. Nie traktować tej części jako docelowej kohorty ani próbki do projektowania splitów.

Bieżący manifest: `data/processed/ptbxl_rhythm_v1/runs/4760b72c29934d1ab406cf7b9ded933a/PTBXL_DOWNLOAD_MANIFEST.json`. Ma status incomplete, 3 exported, 1 failed (ECG 4: DNS) i 19601 not_attempted. To awaria dostępu, nie dowód uszkodzenia lub braku rekordu na S3. Docelowa kohorta pozostaje 19605 ECG / 17223 pacjentów; oczekiwane klasy: SR 16687, AFIB 1484, STACH 805, SBRAD 629. Żaden rekord nie został arbitralnie wykluczony ani zastąpiony.

Parametry, mapowanie, wszystkie ID, oryginalne SCP i oficjalne foldy zachowano. SHA256, metadata alignment, CSV i powtórne przeliczenie X zweryfikowano dla trzech eksportów. Dotychczasowy pilot i MIT-BIH są nietknięte. Pełny zestaw: **465 testów zaliczonych**. Testy syntetyczne służą wyłącznie kontroli kodu, nie walidują niepobranych danych.

Wznowienie po odzyskaniu sieci, z katalogu projektu:

```bash
.venv/bin/python -m src.ptbxl_bulk --execute --approve-bulk
```

Polecenie nie nadpisuje zweryfikowanych eksportów, ponownie sprawdza ich zgodność i tworzy osobny run. Przed użyciem danych należy wskazać **nowy ukończony manifest**, sprawdzić status complete oraz końcowe liczby i checksum. Rootowy manifest checkpoint A nie jest aktualnym wynikiem eksportu. Przykład odczytu poniżej celowo wymaga complete; nie należy usuwać tego warunku, aby ukryć brak danych.

Szczegóły awarii i postępu: `PTBXL_FULL_EXPORT_REPORT.md`. Kontrakt poniżej pozostaje bez zmian; wcześniejszy opis stanu checkpoint A jest historyczny.

---

# PTB-XL rhythm dataset — interfejs dla Anny

**Checkpoint A: implementacja gotowa do kontrolowanego wykonania; pełny dataset nie został jeszcze pobrany ani wyeksportowany.** Istniejący pilot i MIT-BIH są niezależnymi, niezmienionymi produktami danych. Przykład poniżej zadziała po zakończeniu zatwierdzonego eksportu fazy 14.

## Kontrakt danych

Docelowy katalog: `data/processed/ptbxl_rhythm_v1/`. Jeden plik `recordings/<ecg_id:05d>_rhythm.npz` na poprawny zapis, a nie jedna wielka tablica RAM. Logiczny X float32 `[N,12,1000]`: 12 standardowych odprowadzeń, pełne 10 sekund, 100 Hz. Pojedynczy shard ma X `[12,1000]` i y jako skalar int64. Mapowanie: **SR=0, AFIB=1, STACH=2, SBRAD=3**. SR oznacza opisany rytm zatokowy, nie zdrowe ECG ani NORM. STACH/SBRAD oznaczają rytmy zatokowe, nie wszystkie tachykardie/bradykardie.

Kolejność kanałów: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6. Adapter wymaga jej zgodności z nagłówkiem i jednostek mV; nie zmienia nazw/kolejności ani jednostek po cichu. Wyjściowy Z-score jest bezwymiarowy.

Eksport zachowuje `ecg_id`, `patient_id`, `strat_fold`, `rhythm_code`, `original_scp_codes` (JSON), `original_metadata` (JSON z surową komórką SCP, wszystkimi kodami, likelihood i jakością), `channel_names`, `class_names`, `sampling_frequency`, `preprocessing_configuration` i `source_hashes`. JSON przechowywany jest jako Unicode, bez pickle. patient_id pochodzi z CSV, a nie niezależnej identyfikacji pacjenta przez nagłówek.

Kohorta: dokładnie jeden kod spośród wszystkich 12 rozpoznanych kodów rytmu i musi należeć do czterech celów. Brak kodu nie oznacza SR. Nie wybieramy etykiety priorytetem ani największą likelihood; zero może oznaczać nieznaną pewność. Pozostałe diagnozy nadal mogą współistnieć. Wyłączenie nakładających się rytmów ogranicza reprezentatywność.

## Konfiguracja i ograniczenia

Ponownie używane są walidowane funkcje fazy 13: Butterworth high-pass 0,5 Hz / low-pass 45 Hz, rząd 4, SOS zero-phase offline. Notch wyłączony: 50/60 Hz jest na/ponad Nyquistem przy fs=100 Hz. Z-score osobno dla każdego odprowadzenia każdego pełnego rekordu, po filtrowaniu, ddof=0, epsilon=1e-8; zapisujemy dokładne dopasowane statystyki. Nie dopasowujemy parametrów globalnych do połączonych foldów. Nowe zapisy wymagają zgodnego przetwarzania całego okna. To procedura offline, nie przyczynowe przetwarzanie strumienia. Normalizacja usuwa bezwzględną amplitudę i zależy od artefaktów; efekty brzegowe i bliskość cutoff do Nyquista pozostają ograniczeniami.

Dataset identity zapisuje niezmienioną konfigurację pilota jako wersję przetwarzania oraz osobno pełną regułę kohorty. Pole `selection_rule` wewnątrz oryginalnej konfiguracji opisuje pochodzenie pilota; pełna selekcja jest zdefiniowana w `cohort_selection_rule` manifestu, a nie regułą „dwa na klasę”.

## Odczyt po zakończeniu eksportu

Każde uruchomienie tworzy osobny `runs/<run_id>/PTBXL_DOWNLOAD_MANIFEST.json`, CSV i raport. Wskaż konkretny zweryfikowany run; nie wybieraj przypadkowo niepełnej próby. Nie ładuj wszystkich rekordów do RAM i nie zakładaj N=19605, jeśli manifest raportuje błędy.

```python
import json
from pathlib import Path
from src.ptbxl_rhythm import load_pilot_record

root = Path('data/processed/ptbxl_rhythm_v1')
run = root / 'runs' / '<verified_run_id>'
manifest = json.loads((run / 'PTBXL_DOWNLOAD_MANIFEST.json').read_text())
assert manifest['status'] == 'complete'  # review failures before accepting a partial export
for row in manifest['records']:  # deterministic ascending ecg_id
    if row['status'] not in ('exported', 'validated_existing_export'):
        continue
    record = load_pilot_record(root / row['file'])
    X, y = record['X'], record['y']  # float32 [12,1000], scalar int64
    patient_id = int(record['patient_id'])
    fold = int(record['strat_fold'])
    original_scp = json.loads(str(record['original_scp_codes']))
    # Anna can convert each contiguous NumPy array with torch.from_numpy(X).
    # A batch of B such arrays has [B,12,1000]; no model is implemented here.
    break
```

`load_pilot_record` jest ponownie użytym walidatorem formatu pojedynczego zapisu, używanym również w pełnym eksporcie; nazwa funkcji nie oznacza, że wczytuje jedynie osiem pierwszych ECG.

## Pacjenci, foldy i rozkład klas

Zachowujemy oficjalne strat_fold. Sprawdzono, że pacjent nie występuje w kilku foldach. Oficjalne zalecenie 1–8 trening, 9 walidacja, 10 test jest informacją dla Anny; ten moduł nie tworzy splitów. Należy utrzymać separację pacjentów przy wyborze rekordów. Balansowanie, wagi i statystyki globalne wolno wyznaczać tylko z treningu po ustaleniu podziału; nie z całej tabeli rozkładów.

Wybrana kohorta: SR 16687 / 14986 pacjentów; AFIB 1484 / 1220; STACH 805 / 763; SBRAD 629 / 608. Łącznie 19605 ECG / 17223 pacjentów. Suma pacjentów per klasa nie jest liczbą unikalnych osób: ta sama osoba może mieć różne rytmy w różnych ECG. SR dominuje, rzadkie klasy mają mniej różnorodnych pacjentów. Pliki CSV wyraźnie oddzielają selected od exported; na checkpoint A exported=0, ponieważ pilot nie jest pełnym eksportem. Po wykonaniu użyć CSV konkretnego run, uwzględniających rzeczywiste niepowodzenia.

MIT-BIH klasyfikuje pojedyncze uderzenia AAMI, 2 kanały / 252 próbki / 360 Hz. PTB-XL klasyfikuje nazwany rytm całego krótkiego zapisu, 12 kanałów / 1000 próbek / 100 Hz. Nie łączyć X ani y tych dwóch zadań w jeden zbiór. Nie wykonano modelowania, ewaluacji ani Streamlit.

Źródła: istniejące raporty faz 12–13 i lokalne CSV PTB-XL 1.0.3. [Oficjalna dokumentacja](https://physionet.org/content/ptb-xl/1.0.3/). Publiczny endpoint pobierania wskazany i użyty w pilocie: https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/ . Dostępność sieci nie jest sprawdzana przez dry-run.
