# Propozycja etykiet rytmu PTB-XL po audycie rzeczywistych metadanych

## Rekomendowany cel początkowy

Dla wykonalnego projektu uczelnianego proponuję cztery nazwane etykiety rekordu:

- **SR — sinus rhythm**: klasa porównawcza jawnego rytmu zatokowego, nie „zdrowy pacjent” ani NORM.
- **AFIB — atrial fibrillation**: migotanie przedsionków.
- **STACH — sinus tachycardia**: tachykardia zatokowa, bez utożsamienia z SVTAC/PSVT/SVARR lub VT.
- **SBRAD — sinus bradycardia**: bradykardia zatokowa, bez utożsamienia z blokiem AV lub innym wolnym rytmem.

To rekomendacja inżynierii danych, nie wynik oceny modelu ani gwarancja klinicznej trafności. Metadane mają wystarczającą różnorodność pacjentów i niezerowe pokrycie wszystkich foldów, aby rozważyć uczenie tych klas. Nie stosowano uniwersalnego, arbitralnego progu liczby pacjentów. Reguły i scope ustalamy przed modelowaniem, bez wyników Accuracy/F1 lub doboru hiperparametrów na teście.

## Multi-label czy mutually exclusive multi-class?

**Całe oryginalne PTB-XL należy traktować jako wieloetykietowe.** W 143 ECG jest wiele kodów rhythm, w tym 17 wspólnych AFIB/AFLT. Nie ma podstaw do globalnego przypisania każdego ECG do jednej klasy ani do wyboru największej likelihood. Score 0 oznacza nieznaną certainty i jest standardowo obecny w wielu wpisach rytmu.

**Dla pierwszego, jasno ograniczonego zadania rekomenduję mutually exclusive multi-class na jawnie audytowanej kohorcie jednoznacznych kodów.** Reguła propozycji: dokładnie jeden kod spośród WSZYSTKICH 12 rhythm-flagged SCP, należący do czterech celów. Nie wystarczy dokładnie jeden kod spośród czterech celów, jeśli rekord zawiera dodatkowo PACE, AFLT lub inny rytm. Diagnostic/form i jakość są zachowane, nie są podstawą wykluczenia w tej propozycji.

Alternatywa multi-label na pełnym zakresie jest naukowo zgodna z oryginalną strukturą, lecz wymaga osobnej polityki negatywów, braków i znaczenia etykiet dla skróconego zakresu celów oraz większej uwagi w treningu/ocenie. Nie jest konieczna, aby ukończyć podstawowy projekt z nazwanymi zaburzeniami. Decyzję o kontrakcie należy uzgodnić z Anną/opiekunem; nie wdrożono architektury, loss ani splitu.

„Jednoznaczność” oznacza tu pojedynczy zapisany kod, nie potwierdzenie jednolitego rytmu w każdej próbce ani klinicznie bezbłędnej adnotacji. Wiele rozpoznań serca nadal może współistnieć w wybranych rekordach. Nie zmieniamy wszystkich rekordów pacjenta na jeden rodzaj rytmu.

## Dokładne liczby przed/po propozycji

| Kod | Rekordy przed | Osoby przed | Rekordy po | Osoby po | Osoby po: fold 9 / 10 |
|---|---:|---:|---:|---:|---:|
| SR | 16748 | 15035 | 16687 | 14986 | 1532 / 1500 |
| AFIB | 1514 | 1245 | 1484 | 1220 | 119 / 128 |
| STACH | 826 | 782 | 805 | 763 | 79 / 77 |
| SBRAD | 637 | 616 | 629 | 608 | 59 / 60 |

Po propozycji: **19605 rekordów od 17223 osób**. Sumy liczb osób per klasa mogą być większe niż liczba różnych osób w kohorcie: ta sama osoba może mieć inny rytm w innych własnych ECG. Wykryto 346 takich osób dla czterech wybranych etykiet. Nie są to współwystępujące etykiety w jednym wybranym rekordzie ani przeciek między foldami.

Powody niekwalifikowania do propozycji są rozłączne i zachowane dla wszystkich 21799 rekordów:

| Powód | Rekordy |
|---|---:|
| Brak dowolnego kodu rhythm | 769 |
| Więcej niż jeden kod rhythm | 143 |
| Pojedynczy kod rhythm spoza proponowanych czterech | 1282 |
| Proponowane kwalifikujące | 19605 |

Nie wykonano faktycznego usunięcia rekordów z metadanych, przepisania ich etykiet ani budowy finalnego datasetu. `proposed_cohort_audit.csv` zachowuje każdy rekord, patient_id, oryginalne kody i proponowany status. Nie zastosowano progów likelihood, quality ani filtrów demograficznych. Nie uznano braku kodu za potwierdzony normalny rytm. Oryginalne wpisy i wartości są również w `original_statements_and_quality.jsonl`.

## AFLT — ważny cel, ale nie w pierwszej obowiązkowej wersji

AFLT występuje w **73 rekordach od 56 pacjentów**. W oryginalnych foldach 9 i 10 są odpowiednio tylko 3 i 6 różnych osób. Przy takiej samej regule jednoznaczności co dla rdzenia pozostałoby **36 rekordów od 30 osób**, zaledwie **2 osoby w foldzie 9 i 2 w foldzie 10**. To konkretny problem małej różnorodności, a nie odwołanie do arbitralnego progu „min. 100 pacjentów”. Nie obiecujemy wiarygodnej oceny generalizacji tej klasy na podstawie kilku osób.

Wariant `core_plus_aflt` ma 19641 rekordów i 17243 różnych osób; liczby pozostałych klas pozostają jak wyżej. Pokazano go w JSON jako kontrfaktyczne zestawienie, bez generowania treningu. AFLT można pozostawić jako osobny przyszły cel eksploracyjny, zebrać dodatkowe zgodne dane lub rozważyć inny schemat badania z Anną/opiekunem. **Nie scalać AFLT z AFIB**, nie syntetyzować nowych pacjentów i nie mylić liczby powielonych okien z liczbą osób.

## Pozostałe kody

- **SARRH:** 772 rekordy / 757 osób; po jednoznacznej regule 765 / 750. Wariant z tą klasą ma 20370 rekordów i 17828 osób, z 75/76 pacjentami SARRH w foldach 9/10. Dane pozwalają rozważyć piątą klasę, ale arytmia zatokowa może być fizjologiczna, więc nie należy automatycznie traktować jej jako „choroby serca”. Rekomenduję jako opcjonalne rozszerzenie po czteroklasowym celu.
- **PACE:** 294 / 237 — stan prawidłowo działającego stymulatora, poza początkowym celem AF/tachy/brady; nie jest brakującą klasą normalną.
- **SVARR:** 157 / 148 — broad supraventricular arrhythmia, zbyt niespecyficzna, aby nazywać ją konkretną tachykardią zatokową lub PSVT.
- **BIGU:** 82 / 79 — bigeminia o nieokreślonym pochodzeniu; nie wolno przypisać jej automatycznie komorowego znaczenia.
- **SVTAC:** 27 / 27, **PSVT:** 24 / 24, **TRIGU:** 20 / 20 — mało osób i zwykle 2–3 w foldach 9/10; nie proponuję ich jako obowiązkowych odrębnych klas w pierwszej wersji. Ich odroczenie nie oznacza nieistotności klinicznej ani braku takich stanów.

Nie wybieramy klas wyłącznie na podstawie liczby rekordów. Definicje, możliwa heterogeniczność oraz liczby pacjentów i pokrycie foldów są jawne w głównych CSV. Wszystkie oryginalne kody pozostają dostępne do dalszych decyzji.

## Certainty i jakość

Wartość 0 występuje przy wszystkich 16748 SR, 637 SBRAD i 772 SARRH, a także 1466 z 1514 AFIB. Filtr score>0 usunąłby istotne etykiety i nie jest zastosowany. PACE i AFLT mają inny rozkład scores, więc nie należy traktować jednego progu jako porównywalnego kryterium między rytmami. Likelihood nie jest tu prawdopodobieństwem wyjścia modelu ani zweryfikowaną miarą certainty wszystkich kodów.

W fazie 13 warto obejrzeć niewielki pilotaż sygnałów z zachowaniem validation/quality fields i ocenić clipping, kanały i artefakty. Na tym etapie nie odrzucono rekordów na podstawie jakości i nie przepisano diagnoz z tekstu raportu. Nie wolno usuwać lub poprawiać przypadków na podstawie późniejszych błędów testowych modelu bez jawnego, odrębnego protokołu danych.

## Foldy i odpowiedzialność Anny

Dla wszystkich 18869 pacjentów potwierdzono jedną wartość strat_fold na osobę. Zachować tę informację. Oficjalne zalecenie 1–8 trening / 9 walidacja / 10 test opisujemy, ale nie tworzymy tablic/list splitu. Dobór modelu, treningu i ewaluacji należy do Anny. Balansowanie i ewentualne wagi wolno później ustalać tylko z treningu; globalna tabela z tej fazy nie jest gotowym słownikiem wag.

## Źródła i artefakty

- [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) i [SCP definitions](https://physionet.org/content/ptb-xl/1.0.3/scp_statements.csv).
- Rzeczywiste lokalne CSV; hashe i walidacja w PTBXL_METADATA_AUDIT.md.
- PTBXL_RHYTHM_CLASS_COUNTS.csv, PTBXL_RHYTHM_PATIENT_COUNTS.csv i PTBXL_LABEL_COOCCURRENCE.csv.
- Pełne wyniki obu wariantów rozszerzenia i foldów w `results/preprocessing/phase12_ptbxl_metadata_audit/`.

Wcześniejsze MIT-BIH AAMI i audyt fazy 11 pozostają niezmienione; nie mieszamy etykiet uderzeń i rytmu.
