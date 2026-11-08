# Audyt baz i etykiet rytmu — faza 11

To wyłącznie audyt badawczy. Nie zmieniono eksportu fazy 10, istniejących modułów ani wcześniejszych raportów. Nowy kod czyta nagłówki i adnotacje WFDB oraz opcjonalnie CSV z metadanymi. Nie filtruje sygnałów, nie wycina datasetu rytmów, nie pobiera sygnałów PTB-XL, nie ustala splitów i nie trenuje modeli.

## Źródła i odtwarzalność

- [PTB-XL 1.0.3 — oficjalny opis](https://physionet.org/content/ptb-xl/1.0.3/).
- [Oficjalny słownik SCP w przeglądarce](https://physionet.org/content/ptb-xl/1.0.3/scp_statements.csv).
- [Podgląd metadanych rekordów PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/ptbxl_database.csv) — podgląd jest częściowy, nie pełny CSV.
- [Oficjalne symbole WFDB i znaczniki rytmów](https://archive.physionet.org/physiobank/annotations.shtml).
- [MIT-BIH — zapis, odprowadzenia i artefakty](https://archive.physionet.org/physiobank/database/html/mitdbdir/intro.htm).
- [Przykładowy oryginalny nagłówek PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/records500/00000/00001_hr.hea) — kontrola jednostek, fs i kolejności odprowadzeń bez pobierania waveformu.

Wyniki audytu i ich pochodzenie: `results/preprocessing/phase11_rhythm_audit/`. `src/rhythm_audit.py` i `examples/audit_rhythm_datasets.py` są nowymi plikami; dotychczasowy heartbeat preprocessing pozostaje odrębnym, niezmienionym pipeline’em.

## MIT-BIH: co faktycznie sprawdzono

WFDB rdheader/rdann odczytały wszystkie 48 lokalnych rekordów. Rytm rozpoznajemy wyłącznie z adnotacji symbolu `+` i jej pola aux_note, np. `(AFIB`; nie z symbolu uderzenia A/V/N. Nul na końcu niektórych aux i białe znaki są usuwane przy parsowaniu, ale oryginalny tekst i indeks adnotacji zostają zachowane w audycie.

Reguła przedziałów: półotwarty zakres `[pozycja +, pozycja następnego +)`. Stan ostatniego znacznika przenosimy do końca zapisu, jawnie oznaczając prawostronne cenzorowanie. To interpretacja utrzymywanego stanu adnotacji, nie pewność klinicznego początku/końca arytmii. Pierwsza deklaracja stanu może opisywać rytm już trwający przed zapisem; nie jest dowodem początku nowego epizodu. Liczba „epizodów obserwowanych” oznacza maksymalne ciągłe odcinki tej samej etykiety; powtórny znacznik tego samego stanu nie tworzy kolejnego epizodu. Wszystkie surowe granice znaczników zostają osobno.

Przed pierwszym `+` jest UNKNOWN, a nie N. Uszkodzony lub nieznany znacznik przerywa pewny stan i otwiera UNKNOWN. Sprzeczne znaczniki w tej samej próbce oznaczają przedział niejednoznaczny; nie wybieramy ostatniego. Dopiero kolejny poprawny znacznik przywraca udokumentowaną etykietę. Znaki inne niż `+`, nawet z tekstem przypominającym rytm, nie zmieniają stanu i są raportowane. Jakości `~` nie traktujemy jako początku/końca rytmu; czasy poniżej nie są po przeglądzie jakości sygnałów.

### Zweryfikowany inwentarz lokalny

| Aux code | Nagrania | Grupy osób | Obserwowane epizody | Czas stanu [s] |
|---|---:|---:|---:|---:|
| AB | 1 | 1 | 3 | 88.288889 |
| AFIB | 8 | 7 | 107 | 7956.419444 |
| AFL | 3 | 3 | 45 | 785.180556 |
| B | 12 | 12 | 221 | 2510.930556 |
| BII | 1 | 1 | 5 | 699.802778 |
| IVR | 2 | 2 | 4 | 146.388889 |
| N | 42 | 41 | 530 | 63326.719444 |
| NOD | 3 | 3 | 36 | 279.075000 |
| P | 4 | 4 | 60 | 6611.827778 |
| PREX | 1 | 1 | 103 | 740.605556 |
| SBR | 1 | 1 | 1 | 1805.344444 |
| SVTA | 7 | 7 | 26 | 210.513889 |
| T | 12 | 12 | 83 | 1140.797222 |
| VFL | 1 | 1 | 6 | 143.863889 |
| VT | 13 | 13 | 61 | 214.050000 |

Występują wszystkie 15 kodów ze słownika: AB, AFIB, AFL, B, BII, IVR, N, NOD, P, PREX, SBR, SVTA, T, VFL, VT. Nie znaleziono odrębnej etykiety sinus tachycardia. Nie tworzymy jej z SVTA, VT ani progu HR. Pełne listy ID rekordów per rytm są w `mitdb_rhythm_inventory.csv` i JSON, a grupy osób korzystają z istniejącej mapy z fazy 10 (201/202 razem).

Łącznie znaleziono **1291 znaczników + i 1291 ciągłych obserwowanych odcinków opisanych rytmem** oraz 1243 przejścia pomiędzy różnymi znanymi etykietami. W tych danych nie było nieudokumentowanych, uszkodzonych ani kolidujących znaczników rytmu. UNKNOWN obejmuje **47 początkowych przedziałów o łącznym czasie 6,858333 s**. Wszystkie 48 zapisów mają znaczniki rytmu. Cały czas zapisów to 86666.666667 s; znane stany i UNKNOWN dokładnie pokrywają nagrania. Te wyniki nie dowodzą, że każdy fragment ma prawidłową kliniczną etykietę lub dobrą jakość. Końcowe stany w 48 zapisach są przeniesione do EOF; rzeczywistego końca tych epizodów nie znamy.

Kluczowe rekordy:

- AFIB: 201, 202, 203, 210, 217, 219, 221, 222 — 8 nagrań, ale 7 osób.
- AFL: 202, 203, 222 — 3 osoby.
- SBR: tylko 232 — 1 osoba.
- SVTA: 114, 201, 207, 209, 220, 222, 234 — 7 osób.
- VT: 106, 200, 203, 205, 207, 210, 213, 214, 215, 217, 221, 223, 233 — 13 osób.

### Długość epizodów a wykonalność 10-sekundowych przykładów

`mitdb_theoretical_10s_capacity.csv` zawiera tylko arytmetykę długości stanów: sumę floor(liczba_próbek_epizodu / (10*fs)), bez tworzenia okien ECG i bez marginesów bezpieczeństwa. To górna pojemność dla niepokrywających się okien przy danej regule adnotacji, nie liczba gotowych przykładów i nie liczba niezależnych pacjentów.

- AFIB: 752 potencjalne sloty, 66 epizodów >=10 s, 7 grup z takim epizodem.
- AFL: 60 slotów, 14 epizodów >=10 s, 3 grupy.
- SBR: 180 slotów, ale z jednego epizodu i jednej osoby.
- SVTA: 12 slotów z 5 dłuższych epizodów, tylko 3 grupy.
- VT: 8 slotów z 2 epizodów >=10 s, tylko **1 grupa**, mimo 61 obserwowanych epizodów łącznie.

Wiele krótkich napadów może być klinicznie istotnych; nie nazywamy ich niepoprawnymi. Po prostu nie dostarczają samodzielnych jednorodnych odcinków 10 s. Nie zwiększamy ich długości paddingiem, rozciąganiem ani łączeniem oddzielnych epizodów. Nie ogłaszamy nowego standardu diagnostycznego minimalnej długości napadu.

## PTB-XL: co jest zweryfikowane, a czego nie policzono

Wyszukiwanie w projekcie i katalogu domowym nie znalazło `ptbxl_database.csv` ani `scp_statements.csv`. Próba odczytu surowego małego CSV z terminala zakończyła się błędem DNS. Oficjalny web-viewer udostępnił cały słownik SCP oraz tylko początek tabeli rekordów. Nie ma podstaw do przeliczenia częstości dla pełnej bazy z tego podglądu. Zapisaliśmy osobno jawnie opisaną transkrypcję 12 wierszy rhythm ze słownika — to nie oryginalny pobrany CSV ani lista rekordów.

Oficjalna strona podaje **21799 rekordów i 18869 pacjentów** dla wersji 1.0.3. Są to liczby referencyjne z dokumentacji, nie nasz niezależny wynik liczenia metadanych. Faktyczne lokalnie zweryfikowane częstości per kod i liczby osób per kod pozostają **UNKNOWN**, nie 0. Nie używamy częstości z poprzednich wersji ani cudzych blogów.

### Dokładne rhythm-flagged definicje SCP

| Kod | Pole description w oficjalnym słowniku | Rekordy w pełnej bazie | Osoby w pełnej bazie |
|---|---|---|---|
| SR | sinus rhythm | nieustalone | nieustalone |
| AFIB | atrial fibrillation | nieustalone | nieustalone |
| STACH | sinus tachycardia | nieustalone | nieustalone |
| SARRH | sinus arrhythmia | nieustalone | nieustalone |
| SBRAD | sinus bradycardia | nieustalone | nieustalone |
| PACE | normal functioning artificial pacemaker | nieustalone | nieustalone |
| SVARR | supraventricular arrhythmia | nieustalone | nieustalone |
| BIGU | bigeminal pattern (unknown origin, SV or Ventricular) | nieustalone | nieustalone |
| AFLT | atrial flutter | nieustalone | nieustalone |
| SVTAC | supraventricular tachycardia | nieustalone | nieustalone |
| PSVT | paroxysmal supraventricular tachycardia | nieustalone | nieustalone |
| TRIGU | trigeminal pattern (unknown origin, SV or Ventricular) | nieustalone | nieustalone |

Słownik obejmuje także rozpoznania diagnostyczne (m.in. NORM, 1AVB/2AVB/3AVB, WPW, bloki odnóg, zawały, przerosty i zmiany ST/T) oraz form statements. Oficjalne kategorie diagnostyczne to NORM, MI, STTC, CD i HYP. **NORM jest normal ECG, nie etykietą rytmu SR. PAC/PVC to form statements o pobudzeniach, nie timestampy uderzeń ani definicje sinus tachycardia/VT.** Pełny słownik stanowi źródło, a nie nową mapę AAMI.

Nie ma dedykowanego VT wśród 12 rhythm-flagged kodów tego słownika. To nie jest dowód nieobecności klinicznego VT w jakimkolwiek zapisie lub tekście raportu. STACH i SBRAD mają jawnie zatokowe znaczenie; SVTAC, PSVT i SVARR nie są ich zamiennikami. BIGU/TRIGU nie określają pochodzenia komorowego, więc nie są bezpośrednio tożsame z MIT B/T.

### Granularność, współwystępowanie i certainty

SCP statements są przypisane do całego 10-sekundowego rekordu, bez granic wystąpienia każdego rytmu. Mogą współistnieć rozpoznania diagnostyczne, form i rytmu; możliwe są także mnogie kody rytmu. Nie można z ich obecności dowieść, że dany rytm obejmuje każdą próbkę, ani obliczać dokładnych czasów epizodów jako count*10 s.

W `scp_codes` jest słownik kod → likelihood. Dokumentacja mówi, że **0 oznacza nieznaną likelihood**, nie brak rozpoznania; nie jest skalibrowanym prawdopodobieństwem modelu. Podgląd rekordu 1 zawiera SR:0, rekord 2 SBRAD:0, a rekord 17 współwystępujące AFLT:100 i AFIB:0. To zweryfikowane przykłady struktury, a nie statystyki populacji. Odrzucenie kodów o wartości 0 skasowałoby rzeczywiste wpisy rytmu. Parser audytu liczy obecność klucza niezależnie od wartości, raportując wartości likelihood osobno; nie utożsamia takiego wpisu z pewnym klinicznym ground truth. Nie wybieramy dowolnego priorytetu pomiędzy sprzecznymi/współistniejącymi kodami.

Istnieją `patient_id`, `ecg_id`, pola walidacji ludzkiej i jakości oraz `strat_fold` 1–10. Oficjalne foldy uwzględniają grupy pacjentów; zalecenie autora 1–8/9/10 pozostaje metadanymi, nie utworzonym tu splitem. Bez pełnego CSV nie weryfikowano ani rozkładu etykiet w foldach, ani rzeczywistej zgodności pacjentów między foldami.

## Artefakty i reprodukcja

W katalogu audytu są: mitdb_rhythm_audit.json, inventory CSV, surowe markers CSV, przedziały CSV, scalone episodes CSV, transitions CSV, unknown intervals CSV i theoretical_10s_capacity CSV. PTB-XL: ptbxl_metadata_audit.json (jawne braki), ptbxl_source_observations.json i ptbxl_rhythm_definitions_verified.csv (transkrypcja z pochodzeniem). `preservation_before.json` zawiera SHA-256 82 istniejących plików źródłowych, testów, dawnych raportów i eksportu.

```bash
cd ~/Pulpit/ECG-APP
.venv/bin/python -m examples.audit_rhythm_datasets \
  --output-dir results/preprocessing/phase11_rhythm_audit_reproduced
.venv/bin/python -m pytest
```

Przy dostępnych pełnych CSV można dodać `--ptb-database <plik>` i `--scp-statements <oryginalny_plik>`, zawsze do nowego katalogu wyników. Parser wyłącznie analizuje metadane; nie pobiera sygnałów, nie tworzy splitów ani etykiet treningowych. Wymagane jest udokumentowanie wersji, kompletności i hashy CSV. Dane z testowych fixture'ów nie są przypisywane do prawdziwej bazy.

Główne wnioski projektowe i plan kolejnego etapu są w RHYTHM_CLASS_FEASIBILITY.md i PROPOSED_RHYTHM_PREPROCESSING_PLAN.md. Jawna kompatybilność pojęć, bez arbitralnego łączenia rozpoznań, jest w RHYTHM_LABEL_COMPATIBILITY.csv. Dotychczasowe raporty, README i dataset fazy 10 nie zostały nadpisane.


## Weryfikacja końcowa i zachowanie fazy 10

Całe `.venv/bin/python -m pytest`: **373 passed, 132 warnings in 57.65s** — 342 dotychczasowe testy i 31 nowych. Ostrzeżenia dotyczą pandas wewnątrz WFDB. Nowe testy obejmują wyłącznie nowy kod audytu: padding aux, nieznane/uszkodzone tagi, prefix UNKNOWN, utrzymywanie stanów, powtórzenia i konflikty tej samej próbki, cenzorowanie, granice i sumy przedziałów, bezpieczne parsowanie SCP oraz score 0, co-labels i metadane pacjent/fold w małych, jawnie testowych fixture'ach, a także samą arytmetykę pojemności czasu. Nie są to wyniki trenowania modelu.

Kontrola składni nowych modułów przeszła. Porównano SHA-256 **82 istniejących plików** (wcześniejsze moduły, testy, skrypty, raporty, mapa grup i pełny eksport) z baseline przed audytem: brak zmian. Ponownie porównano wszystkie **676 plików źródłowych** z manifestem fazy 10: brak zmian. Testy eksportu nadal potwierdzają 109452 segmenty i 47 grup. Kontrola sum czasów znanych stanów i UNKNOWN zgadza się z długościami nagrań. Wynik zapisano w `final_verification.json` w katalogu audytu.

Utworzone raporty: RHYTHM_DATASET_AUDIT.md, RHYTHM_LABEL_COMPATIBILITY.csv, RHYTHM_CLASS_FEASIBILITY.md, PROPOSED_RHYTHM_PREPROCESSING_PLAN.md. Nowy kod: src/rhythm_audit.py, examples/audit_rhythm_datasets.py i tests/test_rhythm_audit.py. Wcześniejszego README i raportów nie modyfikowano. Nie pobrano waveformów PTB-XL, nie utworzono datasetu rytmowego ani modelu. Audyt zamknięto z jawną niewiadomą: pełne per-klasowe częstości i liczby pacjentów PTB-XL wymagają kompletnego CSV wersji 1.0.3.
