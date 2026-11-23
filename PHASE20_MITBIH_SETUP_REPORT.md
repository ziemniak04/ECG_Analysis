# Faza 20 — MIT-BIH heartbeat CNN: konfiguracja i smoke test

Status: ukończono przygotowanie osobnego modelu uderzeń i jednorazowy test techniczny; **nie uruchomiono pełnego treningu ani oceny walidacyjnej/testowej**. Zakończone zadanie rytmów PTB-XL pozostaje bez zmian.

## Dane, identyfikacja i audyt

Wykorzystano istniejący eksport `data/processed/ecg_dataset_v1/`, bez odtwarzania ECG i bez pobierania danych. Potwierdzono wszystkie 48 sum SHA256 shardów, etykiety, oryginalne symbole, record IDs i grupy: **109452 uderzenia, 48 nagrań, 47 technicznych grup**. Kształt logiczny `[109452,2,252]`, X float32, y int64. Klasy N=0, S=1, V=2, F=3, Q=4; liczebności 90594/2781/7235/802/8040.

Manifest SHA256: `1121d2ccc825624cbb5e91906347ae6551e0feb63df88c6cc185adeffb5105c4`.

Audyt przed podziałem korzystał z rzeczywistych etykiet/metadanych NPZ, bez odczytu tablic X. Nie używał wyników modelu. Zachowano istniejący filtr HP 0.5 Hz, LP 45 Hz, brak notch, zero-phase oraz Z-score per nagranie i kanał. Okno `[p-90,p+162)` ma 252 próbki przy 360 Hz i nie przesuwa adnotacji eksperta. Różnice odprowadzeń są zachowane, nie wyrównane.

## Zamrożony podział pacjentów

`configs/mitbih_patient_split_v1.json`, SHA256 `8c5cc28d8dcae170fd222432d9f247fd19207ba01be60763241da4b7cae61831`. Wszystkie grupy są rozłączne i pokrywają pełny eksport. **201 oraz 202 mają wspólną grupę `mitdb_group_201_202`, razem w walidacji.** Nie są traktowane jako dwie osoby.

| Część | Grupy | Nagrania | Uderzenia | N | S | V | F | Q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Trening | 29 | 29 | 62926 | 52254 | 1715 | 4389 | 401 | 4167 |
| Walidacja | 9 | 10 | 23134 | 19094 | 491 | 1444 | 38 | 2067 |
| Końcowy test | 9 | 9 | 23392 | 19246 | 575 | 1402 | 363 | 1806 |

Wszystkie klasy są obecne w każdej części. Liczby grup z F: 10/4/2, z Q: 5/2/2. Dokładne nagrania, wszystkie grupy i algorytm ustalenia podziału opisuje `MITBIH_TRAINING_SETUP.md`; CSV pokazują każdy rekord i grupę.

Przed wyborem protokołu ustalono nominalny cel 60/20/20 i 29/9/9 grup, jawne kotwice rzadkich klas, minimalne wsparcie grup oraz funkcję odchylenia liczebności. Wykonano deterministyczne przeszukanie 20000 kandydatów z seed 42, 5197 spełniało warunki. Rzeczywiste udziały uderzeń to 57.49/21.14/21.37%. To własny podział, nie oficjalny benchmark. Nie wolno zmieniać go po obejrzeniu wyników modelu.

### Nieuniknione ograniczenia reprezentacji

F ma tylko 802 przykłady; 734 (91.52%) pochodzą z dwóch grup. Trzy niezależne części nie mogą równocześnie uzyskać proporcjonalnego udziału tych dwóch osób. Walidacja ma 38 F; w końcowym teście 362 z 363 F pochodzi z jednego nagrania 213. Q: 8025 z 8040 (99.81%) pochodzi z czterech grup; niemal cała walidacyjna Q pochodzi z 104, a testowa z 217. Duża liczba uderzeń nie zastępuje liczby niezależnych pacjentów. Wyniki przyszłego modelu dla F/Q będą szczególnie niepewne i podatne na specyfikę osoby/odprowadzenia. Uwzględniamy wszystkie 48 nagrań, także stymulatorowe, więc nie jest to wariant 44-nagraniowy.

## Loader i model

Loader jest leniwy, deterministyczny według record ID i kolejności uderzeń; domyślny cache LRU mieści dwa shardy na proces. Nie ładuje całego zbioru do RAM. X kopiowane per próbka chroni źródło przed operacjami in-place. Zachowuje symbole, record IDs, grupy, pozycje i indeksy adnotacji, nazwy kanałów, częstotliwość i pozycję w shardzie. Błędne lub brakujące dane powodują jawny błąd. Finalny test jest zablokowany domyślnie przed odczytem waveformu i wymaga przyszłego jawnego opt-in po osobnej autoryzacji.

Osobny `MITBIHBeatCNN`: trzy bloki Conv1d/BatchNorm/ReLU/MaxPool, kanały 16/32/64, jądra 15/9/5, globalne uśrednianie i głowica 64→32→5 z dropout 0.3. **17797 parametrów**. Wejście `[B,2,252]`, wyjście `[B,5]` surowych logitów. Dokumentacja: `MITBIH_MODEL_ARCHITECTURE.md`.

Opcjonalne wagi CE obliczono wyłącznie z 62926 treningowych przykładów: N 0.240847, S 7.338309, V 2.867441, F 31.384539, Q 3.020206. Funkcja wymaga jawnej, dokładnej listy treningowych grup i odrzuca inne. Włączenie wag w smoke teście nie wybiera strategii przyszłego pełnego treningu. Nie zastosowano balansowania ani augmentacji.

## Smoke na rzeczywistych danych — dokładnie jedna aktualizacja

- CPU, PyTorch 2.15.0+cpu, Python 3.12.3, seed 42, 2 wątki.
- 10 rzeczywistych przykładów treningowych: po dwa z każdej klasy, z dwóch różnych grup dla każdej klasy.
- Odczytane X: tylko nagrania **100, 101, 102, 108, 109**.
- Wejście `[10,2,252]` float32; etykiety long; wyjście `[10,5]`, skończone logity.
- Ważona CrossEntropyLoss: **1.518456578**, norma gradientów **1.081637979**; wszystkie gradienty skończone, łączna norma niezerowa.
- Adam, lr 0.001, **1 optimizer step**; parametry rzeczywiście się zmieniły i pozostały skończone.
- Zapis/odczyt stanu wag identyczny tensor po tensorze; bez dodatkowego forward po zapisie.
- Czas samej operacji forward/backward/update: 0.154 s; nie jest to benchmark pełnego treningu.
- **Zero forwardów nowego modelu na walidacji i teście. Brak Accuracy/F1 lub innych miar skuteczności.**

`results/model_setup/phase20_mitbih/run_002/smoke_after_one_update.pt` jest jawnie oznaczony jako techniczny zapis po jednej aktualizacji, nie wytrenowany model do oceny/aplikacji. Szczegóły każdej wybranej próbki i jej adnotacji są w `smoke_result.json`.

## Testy i zachowanie wcześniejszej pracy

Nowy moduł testowy: **22 testy**, obejmujące etykiety/metadane, uszkodzenia i braki, leniwy odczyt/cache, zgodność tensorów, split/patient leakage, zamknięcie testu przed odczytem, wagi train-only, deterministyczną propozycję podziału, backward/update, liczbę parametrów i rzeczywistą próbkę wyłącznie treningową.

Pełne uruchomienie `.venv/bin/python -m pytest -ra`: **620 passed, 229 warnings, 366.24 s**. Log: `results/model_setup/phase20_mitbih/run_002/full_pytest.log`. Ostrzeżenia to deprecacje `inplace` w pandas wywoływane przez WFDB (`set_index`, `drop`); nie są błędami nowego modelu. Nie wyciszano ich i nie zmieniano zależności. Historyczne testy preprocessingowe sprawdzają dane/QC, nie oceniają nowego CNN na finalnym zbiorze; testy PTB końcowej oceny wykorzystują kontrolowane fixture, nie uruchamiają ponownie rzeczywistego fold 10.

Końcowa kontrola `final_integrity_check.json` porównuje SHA256 wcześniejszych źródeł, raportów, wyników treningu i ewaluacji PTB oraz rozmiary/czasy modyfikacji i zbiór plików obu eksportów. Dodatkowo wszystkie 48 shardów MIT-BIH ponownie porównano SHA256 z manifestem. Sumy modelu B PTB-XL i jego wcześniejszych wyników pozostają niezmienione; nie wykonano żadnej nowej ewaluacji PTB-XL. Checkpoint smoke i duże dane są objęte istniejącym `.gitignore`; nie zmieniano go ani nie wykonywano commitów. Zastane zmiany Git pozostawiono.

Pierwsza próba audytu `run_001` zakończyła się błędem serializacji metadanych NumPy → JSON, zanim zamrożono split i zanim uruchomiono smoke. Poprawiono konwersję liczników do Python int. Zachowano katalog, a poprawny audyt/freeze/smoke wykonano w nowym `run_002`; nie nadpisano wyników i nie powtórzono aktualizacji rzeczywistego modelu.

## Utworzone pliki

- `src/mitbih_protocol.py`, `src/mitbih_torch.py`, `src/mitbih_model_config.py`, `src/models/mitbih_cnn.py`.
- `configs/mitbih_patient_split_v1.json`.
- `examples/setup_mitbih_model.py`, `tests/test_mitbih_setup.py`.
- `MITBIH_MODEL_ARCHITECTURE.md`, `MITBIH_TRAINING_SETUP.md`, niniejszy `PHASE20_MITBIH_SETUP_REPORT.md`.
- Osobne artefakty w `results/model_setup/phase20_mitbih/run_002/`: audyt per grupa/per rekord, propozycja i identyfikator zamrożonego splitu, wsparcie klas, konfiguracja modelu, smoke i jego techniczny stan wag, logi testów, kontrola integralności i sumy nowych artefaktów.

## Granica tej fazy

Gotowy jest interfejs i technicznie działający osobny CNN. Przed pełnym treningiem potrzebne są osobne zatwierdzenie, benchmark leniwego I/O/cache i RAM, konfiguracja treningu/resume oraz plan wyboru modelu wyłącznie na walidacji. Nie wolno zmieniać podziału dla poprawy wyników ani otwierać testu w trakcie selekcji. Nie rozpoczęto pełnego treningu MIT-BIH, nowej oceny PTB-XL ani Streamlit. **Koniec fazy 20.**
