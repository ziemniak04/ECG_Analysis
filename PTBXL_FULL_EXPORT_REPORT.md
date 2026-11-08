# Phase 14 — checkpoint B: wykonanie zatwierdzone, eksport niepełny

**Aktualny stan: przerwanie pobierania przez DNS; pełny dataset nie jest gotowy dla Anny.** Zgoda na checkpoint B została udzielona. Uruchomiono rzeczywisty downloader S3; trzy kolejne próby pobrania `records100/00000/00004_lr.hea` zakończyły się `URLError: [Errno -2] Name or service not known`. Dodatkowa próba curl zgłosiła `Could not resolve host: physionet-open.s3.amazonaws.com`. Nie jest to dowód braku pliku na serwerze ani uszkodzenia ECG.

- Nowe kompletne zapisy pobrane w tej próbie: **0**. Osiem zapisów pilota pozostaje lokalnie; pipeline w porządku rosnącym ecg_id dotarł do trzech z nich przed pierwszą próbą pobrania nowego.
- Poprawnie przetworzone i wyeksportowane rzeczywiste zapisy: **3**, ecg_id **1, 2, 3**, od **3 pacjentów**.
- Błędy próby pobrania: **1** (ecg_id 4, DNS). Jeszcze niepróbowane: **19601**. Do dokończenia eksportu pozostaje łącznie **19602** ECG.
- Nie odrzucono ani nie zastąpiono żadnego wybranego rekordu. Nie zmieniono kohorty: **19605 ECG / 17223 pacjentów**. Nie wykryto klinicznej ani strukturalnej niespójności w trzech odczytanych zapisach; dostępność/ważność pozostałych waveformów nie została potwierdzona.

| Rytm | Rzeczywisty częściowy eksport | Wybrana kohorta docelowa |
|---|---:|---:|
| SR | 2 | 16687 |
| AFIB | 0 | 1484 |
| STACH | 0 | 805 |
| SBRAD | 1 | 629 |

Te liczby częściowe nie są końcowym rozkładem pełnego datasetu i nie uzasadniają treningu.

## Zachowany postęp i integralność

Wyjście: `data/processed/ptbxl_rhythm_v1/recordings/00001_rhythm.npz`, `00002_rhythm.npz`, `00003_rhythm.npz`. Każdy ma float32 X `[12,1000]`, skalarny y int64, oryginalne ecg_id/patient_id/strat_fold, SCP, nazwy odprowadzeń, 100 Hz oraz konfigurację. Wariant logiczny aktualnego częściowego X: `[3,12,1000]`.

Aktywny raport tej próby i CSV: `data/processed/ptbxl_rhythm_v1/runs/4760b72c29934d1ab406cf7b9ded933a`. Manifest: `data/processed/ptbxl_rhythm_v1/runs/4760b72c29934d1ab406cf7b9ded933a/PTBXL_DOWNLOAD_MANIFEST.json`. Wszystkie **19605** wybrane ID są w nim jawnie oznaczone jako exported / failed / not_attempted, bez pominięcia ogona po przerwaniu. Rootowe `PTBXL_DOWNLOAD_MANIFEST.json` i CSV pozostają historycznym checkpoint A; do oceny aktualnego postępu używać plików z powyższego run.

Niezależna kontrola SHA256 wszystkich trzech eksportów i źródeł, pełnej zgodności wierszy metadata, etykiet i foldów oraz ponownego przeliczenia X zakończyła się pomyślnie. Zestawiono też tabelę klas i foldów z rzeczywistymi NPZ. Nie ma wspólnych globalnych statystyk normalizacji: nadal Z-score per recording/per lead po filtrach Butterwortha 0,5/45 Hz, order 4, zero-phase, bez notch. Oryginalna konfiguracja fazy 13 pozostała identyczna. Tożsamość datasetu wiąże metadane, kohortę i hashe kodu.

Kod fazy 14 otrzymał kontrolę zatrzymania CLI po pierwszym wyczerpanym błędzie oraz jawne rozróżnienie failed/not_attempted; nie zmieniono numerycznego preprocessing ani modułów faz 10–13. Test zatrzymania sprawdza zapis postępu i zwolnienie locków. Wcześniejsze dane MIT-BIH, pilot, CSV źródłowe i raporty faz 10–13 są zachowane. Sprawdzono SHA256 151 wcześniejszych plików (zmieniono wyłącznie dozwolone pliki fazy 14 i dwa wskazane raporty) oraz zgodność 676 plików źródłowych MIT-BIH z fazą 10. Kopie obu raportów sprzed aktualizacji zachowano w katalogu checkpoint B.

## Testy i odtworzenie

`.venv/bin/python -m pytest`: **465 passed**, 0 failed, 0 skipped, 110,10 s; 222 ostrzeżenia deprecacyjne WFDB/pandas. Wynik testów nie oznacza walidacji niepobranych waveformów. Logi i niezależne kontrole: `results/preprocessing/phase14_checkpoint_b/` (`execution.log`, `pytest.log`, `export_integrity.json`, `verified_exports.csv`, `final_verification.json`).

Po przywróceniu dostępu DNS/HTTPS do S3, w terminalu VS Code z katalogu projektu:

```bash
cd ~/Pulpit/ECG-APP
.venv/bin/python -m src.ptbxl_bulk --execute --approve-bulk
.venv/bin/python -m pytest
```

Zgoda użytkowniczki na to wykonanie została już udzielona. Powtórzenie polecenia zachowuje ten sam katalog, weryfikuje gotowe NPZ bez nadpisania i wznawia pliki tymczasowe. Nie należy usuwać poprawnych eksportów ani zmieniać konfiguracji. Locki z tej próby zostały prawidłowo zwolnione. Każda kolejna próba dostanie osobny katalog run; po wznowieniu należy wskazać nowy manifest i sprawdzić `status == "complete"`, 19605 poprawnych zapisów, 17223 pacjentów, cztery docelowe liczby klas i checksum wszystkich eksportów. Poważny błąd walidacji zatrzyma CLI bez zmiany zasad selekcji.

Nie trenowano modeli, nie tworzono splitów, nie balansowano danych, nie wdrażano Streamlit ani fazy 15. Obecnie blokadą jest rozwiązywanie nazwy S3 w środowisku wykonawczym.

---

## Zachowany historyczny raport checkpoint A

# Phase 14 — checkpoint A: pełny pipeline PTB-XL przygotowany, bez bulk download

Status: **oczekuje na wyraźną zgodę Magdaleny na checkpoint B**. Nie rozpoczęto pobierania zbiorczego ani pełnego eksportu. Nie utworzono docelowego `data/processed/ptbxl_rhythm_v1/`.

## Zweryfikowane liczby

Niezależna rekonstrukcja z rzeczywistych lokalnych CSV i porównanie z istniejącym audytem: **19605 ECG / 17223 pacjentów**, 12 rozpoznanych kodów rytmu. SR=16687, AFIB=1484, STACH=805, SBRAD=629. Brak rozbieżności względem fazy 12. Brak progów likelihood/jakości, brak zamiany braków na SR. Zachowano oryginalne SCP, identyfikatory i foldy.

Wymagane pliki: **39210** (.hea + .dat), wyłącznie records100 ze ścieżek filename_lr. Lokalnie ponownie odczytano WFDB i zweryfikowano osiem par pilota (**16 plików**), bez nadpisywania. Pozostało **39194 plików** poza zweryfikowanymi parami. Nowo pobrane w fazie 14: **0**. Pełny eksport: **0**. To nie jest raport ukończonego datasetu.

Wszystkie 18869 pacjentów źródła mają spójny pojedynczy strat_fold. Kohorta zachowuje istniejące foldy, nie tworzy podziałów. Rozkłady wybranych rekordów i pacjentów per klasa/fold: wymagane CSV w katalogu projektu; exported=0 oznacza brak wykonania pełnego eksportu, nie brak poprawnych zapisów pilota.

## Dysk i transfer

- Dane .dat pełnej kohorty: 470520000 bajtów, wynik kontraktu 1000×12×2 bajty (potwierdzonego dla lokalnego pilota).
- Część .dat poza zweryfikowanymi parami: **470328000 bajtów** (około 470,33 MB). Inny format pliku zostanie zgłoszony jako błąd, nie zostanie przycięty.
- Dokładny łączny transfer wraz z nagłówkami: **nieznany**. Nie wykonywano 39210 żądań HEAD ani zdalnego listowania.
- Nieskompresowane X: 941040000 bajtów, y: 156840 bajtów. Stopień kompresji, narzut JSON/NPZ, bloki filesystemu i miejsce stagingu nie są dokładnie znane. Znane składniki budżetu: 1411524840 bajtów (~1,412 GB); to nie całkowite wymaganie ani górna granica.
- Dostępne miejsce podczas dry-run: **379015442432 bajtów** (~379,02 GB / 352.99 GiB). Nie wykryto lokalnej blokady pojemności dla znanych składników. Stan dysku należy sprawdzić ponownie przed checkpoint B.

## Implementacja i odtwarzanie

Nowy niezależny moduł `src/ptbxl_bulk.py`, testy `tests/test_ptbxl_bulk.py`. Moduł reuse'uje istniejący odczyt WFDB, filtrację, normalizację i format pojedynczego NPZ. Nie modyfikuje wcześniejszych modułów ani konfiguracji. Wersja datasetu ptbxl-rhythm-v1 ma osobną regułę pełnej kohorty; wersja numerycznego przetwarzania pozostaje identyczna z fazą 13.

Dry-run bez sieci i bez efektów zapisu poza jawnie wybranym nowym raportem:

```bash
cd ~/Pulpit/ECG-APP
.venv/bin/python -m src.ptbxl_bulk
# Opcjonalnie raport pod NOWĄ ścieżką:
.venv/bin/python -m src.ptbxl_bulk --dry-run-report results/preprocessing/phase14_checkpoint_a/dry_run_second.json
.venv/bin/python -m pytest
```

**Wyłącznie po wyraźnej zgodzie użytkowniczki** (nie wykonano na checkpoint A):

```bash
.venv/bin/python -m src.ptbxl_bulk --execute --approve-bulk
```

To samo polecenie wznawia pełną pracę. Flaga jest jawnym potwierdzeniem operatora; kod nie interpretuje samego uruchomienia jako zgody. Brak flagi blokuje wykonanie. Endpoint https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/ został wskazany przez użytkowniczkę po udanym pilocie; jego bieżąca dostępność nie była badana na checkpoint A.

## Pobieranie, awarie i wznowienie

Pobierane są tylko pary wybranej kohorty. Oryginalne pliki nigdy nie są nadpisywane. Każda istniejąca para musi przejść walidację WFDB przed pominięciem pobierania. Wznowienie używa plików `.part` w `data/ptbxl/.phase14_staging/`, HTTP Range i kontroli Content-Range; jeśli serwer ignoruje Range, tylko plik tymczasowy zaczyna się od nowa. Content-Length, oczekiwany rozmiar danych oraz ograniczenie wielkości nagłówka są sprawdzane. Próby: 3, timeout 30 s na operację socketu, exponential backoff 1/2 s (ogólny limit 30 s). Nie wymaga AWS CLI ani poświadczeń.

Kompletna para stagingu musi przejść walidację nagłówka, ecg_id, 12 kanałów, 100 Hz, 1000 próbek, kolejności, mV, kalibracji, checksum i wartości skończonych. Dopiero wtedy następuje atomowa instalacja bez nadpisania (`os.link`) na tym samym filesystemie. `.part` do ukończonego pliku stagingu przechodzi atomowo (`os.replace`). To operacje wyłącznie na plikach zarządzanych przez nowy downloader. Jeśli przerwanie nastąpi między instalacją nagłówka i danych, następny run kończy parę po ponownej walidacji.

Nie ma serwerowych SHA256 pobranych z zaufanej listy; zapisujemy lokalne SHA256 i sprawdzamy sumy kanałów WFDB (niekryptograficzne). HTTP Range nie zapewnia sam w sobie niezmienności zdalnego obiektu; pełna para jest sprawdzana przed publikacją. Nie twierdzimy, że te kontrole dowodzą poprawności klinicznej etykiety.

Awaria sieci: pozostawić staging i ponownie uruchomić po przywróceniu połączenia. Plik o błędnej checksum, niepoprawnym formacie lub istniejący uszkodzony oryginał: sprawdzić wskazaną ścieżkę i po świadomej decyzji operatora przenieść do kwarantanny; program nie usuwa ani nie zastępuje go automatycznie. Nierozwiązany błąd pozostaje jawny w manifeście. Lock zapobiega równoczesnym zapisom; po twardym przerwaniu operator musi najpierw upewnić się, że proces nie działa, przed usunięciem stale `.run.lock` lub `.download.lock`. Nie uruchamiać równolegle niezależnych downloaderów w tych samych katalogach.

## Eksport i manifesty

Każdy poprawny rekord: float32 `[12,1000]` plus int64 y i wszystkie metadane. Przetwarzanie ma stały niski koszt pamięci na sygnał. Nowy run zapisuje własny `runs/<run_id>/events.jsonl` po każdym rekordzie, a po zakończeniu `PTBXL_DOWNLOAD_MANIFEST.json`, oba CSV i `PTBXL_FULL_EXPORT_REPORT.md`. Historia wcześniejszych runów pozostaje zachowana. W razie przerwania przed raportem końcowym dostępny jest dziennik JSONL; nowy run ponownie sprawdza cały zakres.

`dataset_identity.json` wiąże kohortę, metadane, dokładną konfigurację i hashe kodu. Zmiana tożsamości wymaga nowego katalogu wersji. Gotowe NPZ są niezmienne. Wznowienie odczytuje źródło, przelicza rekord i porównuje **wszystkie pola** z istniejącym NPZ; nie polega jedynie na obecności nazwy pliku. Round-trip oraz SHA256 są zapisywane w manifeście. Kolejność zawsze rosnąca po ecg_id; końcowe N wynika z faktycznie poprawnych rekordów, nigdy z wymuszania oczekiwanej liczby. Pliki tymczasowe NPZ po twardym przerwaniu mogą pozostać w `.pending`; nie są publikowanymi shardami i wymagają ręcznego przeglądu.

Rootowe wymagane raporty/CSV/manifest obecnie dokumentują checkpoint A. Checkpoint B zapisuje nowe raporty we własnym katalogu run, nie nadpisuje checkpoint A. Instrukcje interfejsu: `PTBXL_DATASET_FOR_ANNA.md`. Duże przetworzone pliki pozostają objęte istniejącym `.gitignore`; bazy i stagingu nie dodawać do Git.

## Kontrole i granice fazy

Rzeczywistą walidację wykonano dla ośmiu istniejących waveformów pilota, nie dla pozostałych zapisów. Kontrolowane syntetyczne fixture'y testują HTTP, awarie, resume i eksport; nie są dowodem poprawności niepobranych danych klinicznych. Nie wprowadzono splitów, balansowania, modeli, treningu, ewaluacji ani Streamlit. Pełne wyniki testów i zachowanie wcześniejszych plików będą zapisane w `results/preprocessing/phase14_checkpoint_a/final_verification.json`.

Źródła: rzeczywiste lokalne CSV i raporty faz 12–13; [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/); publiczny S3 użyty w pilocie. Wcześniejszy plan fazy 14 zachowuje historyczną informację o blokadzie DNS; został celowo nietknięty. Nowy raport opiera się na istniejącym udanym manifeście `phase13_ptbxl_pilot_verified`.

Końcowa sesja testów: **464 passed**, 0 failed, 0 skipped, 103,48 s; 219 ostrzeżeń deprecacyjnych zależności WFDB/pandas. Nowe testy fazy 14: 26. Kontrola SHA256: 143 wcześniejszych plików bez zmian; 676 plików źródłowych MIT-BIH zgodnych z fazą 10. Nie utworzono katalogu pełnego eksportu ani stagingu pobierania. Nowy `data/ptbxl/.gitignore` wyklucza waveformy i staging, pozostawiając metadane możliwe do śledzenia.
