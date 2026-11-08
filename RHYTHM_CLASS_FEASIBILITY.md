# Wykonalność klasyfikacji rytmu — decyzja po audycie fazy 11

## Rekomendacja

Wybrać **wariant B: odrębny klasyfikator rytmu i istniejący klasyfikator uderzeń w jednym frameworku ECG-APP**. To rekomendacja projektowa oparta na audycie danych, nie wynik eksperymentu porównującego modele. Główne zadanie rytmowe rozwijać dalej na PTB-XL dopiero po policzeniu pełnych metadanych; zachować ukończony MIT-BIH heartbeat dataset jako osobny, niezmieniony produkt. Audyt rytmów MIT-BIH można wykorzystać do ograniczonego pilota AF lub późniejszej analizy zewnętrznej, bez obietnicy wystarczającej liczby pacjentów dla wszystkich klas.

Nie można obecnie zatwierdzić klas AF/sinus tachycardia/sinus bradycardia/normal jako wystarczająco licznych w PTB-XL: ich definicje potwierdzono, ale pełne częstości i liczby osób są nieznane. W MIT-BIH sinus tachycardia nie ma własnego zweryfikowanego tagu, SBR ma jednego pacjenta, a przy 10 s tylko jedna grupa dostarcza wystarczająco długich fragmentów VT. Tych ograniczeń nie usuwa oversampling.

## Porównanie trzech opcji

| Opcja | Zalety | Bariery stwierdzone w audycie | Ocena dla projektu dwuosobowego |
|---|---|---|---|
| A. Jeden model rytmu z połączonych okien obu baz | Potencjalnie więcej danych dla rzeczywiście wspólnych pojęć | Niepełna zgodność kodów, inne kanały, długości kontekstu i warunki rejestracji, adnotacje przedziałowe vs rekordowe, nieznane częstości PTB, skrajnie mało pacjentów MIT w części celów | Nie jako pierwszy etap. Wspólny koncept nie oznacza gotowego wspólnego datasetu |
| B. Osobny rytm i osobne uderzenia w jednym frameworku | Zachowuje ukończony eksport AAMI, oddziela jednostki etykietowania, ogranicza ryzyko błędnej harmonizacji | Trzeba utrzymać dwa jawne kontrakty wejścia/wyjścia i dobierać zbiory do zadania | Najbardziej wykonalne, etapowe rozwiązanie; nie projektujemy tutaj modeli |
| C. PTB-XL pretraining i transfer na MIT-BIH | Może być późniejszym eksperymentem wykorzystującym większy zbiór | Inne cele i rozdzielczość etykiet, 12 vs 2 zmienne odprowadzenia, inne okna i domeny; konieczne adaptacja i ocena należące do Anny | Opcja rozszerzona po wiarygodnym baseline; nieuzasadniona jako obowiązkowy pierwszy krok |

Wariant C nie ma potwierdzonej przewagi i nie rozwiązuje sam z siebie małej liczby osób z SBR lub długim VT. Model wspólny A mógłby wykorzystywać tożsamość bazy zamiast cech rytmu, jeżeli część klas występowałaby tylko w jednej domenie. To ryzyko projektowe, nie stwierdzony wynik modelu.

## Proponowana ontologia — jawne pojęcia, nie nowe targety treningowe

Zachować źródłowy kod, bazę i jednostkę adnotacji jako podstawę. Nie nadpisywać AAMI N/S/V/F/Q. Nazwy poniżej to propozycja wersji `rhythm-ontology-proposed-v1`; nie przydzielono ostatecznych indeksów klas ani nie wygenerowano nowego datasetu.

- `atrial_fibrillation`: MIT aux AFIB / PTB SCP AFIB — zgodność pojęcia, różna granularność i pewność adnotacji.
- `atrial_flutter`: MIT AFL / PTB AFLT — zgodność pojęcia, nie łączyć z AF; MIT tylko trzy osoby.
- `sinus_bradycardia`: MIT SBR / PTB SBRAD — zgodność pojęcia, ale MIT jednoosobowy; brak wykonalnego niezależnego podziału pacjentów tej klasy w samej bazie.
- `sinus_rhythm_explicit`: MIT N / PTB SR — warunkowa zgodność przy zachowaniu znaczenia jawnego rytmu zatokowego. Nie jest to zdrowy pacjent ani normal ECG. Nie przenosić STACH/SBRAD/SARRH do tej kategorii bez jawnej reguły hierarchicznej.
- `sinus_tachycardia`: bezpośrednio PTB STACH; brak potwierdzonego odpowiednika tagu MIT. SVTA i VT nie mogą go zastąpić.
- `sinus_arrhythmia`: bezpośrednio PTB SARRH; nie wyprowadzać automatycznie z MIT N lub zmiennych RR.

Dalsze kody źródłowe trzymać osobno, np. `mitdb:VT`, `mitdb:SVTA`, `mitdb:VFL`, `ptbxl:SVTAC`, `ptbxl:PSVT`, `ptbxl:SVARR`. Nie utożsamiać ogólnej arytmii nadkomorowej z nadkomorową tachykardią ani z tachykardią zatokową. BIGU i TRIGU w PTB mają nieokreślone pochodzenie i nie są automatycznie komorowymi B/T z MIT. PACE podkreśla prawidłowe funkcjonowanie stymulatora, MIT P tylko rytm stymulowany — ich scalenie wymagałoby świadomego szerszego celu „pacing present”, z utratą szczegółu. Nie wdrożono takiego scalenia.

BII i 2AVB mogą służyć osobnemu zadaniu bloku AV, ale nie są etykietami zatokowej bradykardii. WPW/PREX to odrębna oś preekscytacji. Form statements PAC/PVC w PTB opisują obecność pobudzeń w rekordzie, a symbole A/V w MIT wskazują pojedyncze próbki — nie są gotowym wspólnym targetem rytmu.

### Jawna niepewność i współwystępowanie

Dla każdej przyszłej obserwacji zachować `label_source`, `source_code`, `granularity`, oryginalny score, status walidacji i `label_status`. UNKNOWN jest statusem braku wiedzy, nie klasą normalną i nie ujemnym AF. Stany nierozstrzygnięte i granice przejść pozostają oznaczone.

PTB SCP może być wieloetykietowe; obecne audytowane przykłady zawierają nawet wspólnie AFIB/AFLT. Nie wymuszać softmaxowej wzajemnej wyłączności przez arbitralny wybór pierwszej etykiety. Po uzyskaniu pełnego CSV należy zdecydować: zachować multi-label, czy jawnie utworzyć podzbiór z jednoznacznym pojedynczym rytmem i raportować odrzucenia. To decyzja o definicji celu, wymagająca uzasadnienia z Anną i opiekunem. Brak klucza w niepełnym raporcie nie jest automatycznie potwierdzonym negatywem. Score 0 jest nieznaną likelihood, nie zerowym prawdopodobieństwem wystąpienia kodu.

## Co można harmonizować rzeczywiście?

AFIB, AFL/AFLT i SBR/SBRAD mają bezpośrednią zgodność opisu pojęcia. N/SR jest możliwe tylko przy jednoznacznej definicji jawnego rytmu zatokowego i bez utożsamienia z normalnym zdrowym ECG. Żadna z tych zgodności nie potwierdza jednorodności całego 10 s PTB ani nie gwarantuje dostatecznej liczby różnych osób. Dla STACH nie istnieje potwierdzony tag MIT. Dla VT nie ma dedykowanego kodu wśród 12 rhythm-flagged SCP; nie oznacza to zerowej klinicznej częstości w całym PTB.

Dlatego odpowiedź brzmi: **możliwy wspólny słownik części pojęć, ale brak podstaw do automatycznego połączenia obu baz w jeden kompletny zbiór proponowanych klas**. Macierz dokładnych zgodności i braków jest w `RHYTHM_LABEL_COMPATIBILITY.csv`.

## Najbliższy praktyczny krok — dopiero po zatwierdzeniu

1. Uzyskać tylko pełne `ptbxl_database.csv` i `scp_statements.csv` wersji 1.0.3 oraz zapisać ich pochodzenie/hashe.
2. Przeliczyć obecność kodów, distinct patient_id, ich współwystępowanie, rozkład likelihood, walidację ludzką i pokrycie foldów. Nie rozdzielać danych ani nie wybierać modelu.
3. Potwierdzić wykonalność AFIB/STACH/SBRAD/SR na poziomie różnych pacjentów, nie tylko liczby rekordów. Uzgodnić multi-label lub jednoznaczny podzbiór.
4. W osobnej zatwierdzonej fazie przygotować mały audyt sygnałów PTB-XL i wersjonowany pipeline rytmu; dopiero potem pełne preprocessing/export. Model i wszystkie podziały nadal należą do Anny.

W tym etapie zatrzymujemy się po audycie, bez wykonywania powyższych rozszerzeń.

## Źródła

- https://physionet.org/content/ptb-xl/1.0.3/
- https://physionet.org/content/ptb-xl/1.0.3/scp_statements.csv
- https://physionet.org/content/ptb-xl/1.0.3/ptbxl_database.csv — tylko częściowy podgląd, bez pełnego liczenia.
- https://archive.physionet.org/physiobank/annotations.shtml
- Wyniki lokalnego WFDB w `results/preprocessing/phase11_rhythm_audit/`.
