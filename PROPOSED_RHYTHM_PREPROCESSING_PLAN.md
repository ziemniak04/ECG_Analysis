# Proponowany preprocessing rytmu — plan, bez implementacji sygnałowej

## Niezmienny punkt odniesienia

Dotychczasowy `data/processed/ecg_dataset_v1/` pozostaje datasetem uderzeń MIT-BIH: 109452 okna `[2,252]`, 360 Hz, per-recording Z-score i pięć grup AAMI. Nie wolno zmienić semantyki etykiet tych plików na AF/tachykardię/bradykardię. Nie modyfikowano istniejących modułów, mapy AAMI ani wcześniejszych raportów. Plan poniżej nie jest wykonanym preprocessingiem, nowym datasetem ani modelem.

## Audyt zgodności technicznej

| Właściwość | MIT-BIH lokalnie | PTB-XL 1.0.3 — dokumentacja / nagłówek | Konsekwencja |
|---|---|---|---|
| fs | 360 Hz | 500 Hz oraz wersja 100 Hz | Wspólny sampler wymaga jawnego resamplingu; nie wolno traktować tego samego indeksu jako tego samego czasu |
| Długość | 650000 próbek na rekord, ok. 30 min | 10 s na rekord | MIT ma długi kontekst, PTB krótki rekord; nie przenosić statystyk i filtracji bez kontroli efektów brzegowych |
| Kanały | 2, zmienne konfiguracje MLII/V1/V2/V4/V5, odwrócona kolejność w 114 | 12 standardowych odprowadzeń, nagłówek przykładu I, II, III, AVR, AVL, AVF, V1–V6 | MLII nie jest fizycznie identyczne ze standardowym II; nie upodabniać semantyki samym przemianowaniem |
| Jednostki | WFDB fizyczne mV z nagłówków | Nagłówek 00001_hr: gain 1000/mV; dokumentacja rozdzielczość 1 µV/LSB | Czytać skalę z nagłówka, nie mnożyć arbitralnie; amplitudy po Z-score są bezwymiarowe |
| Adnotacje | + aux z granicami stanów; osobno uderzenia z próbkami | SCP na cały rekord, wieloetykietowe, z likelihood/validation fields | Nie kopiować rekordowej etykiety jako dowodu dokładnych granic epizodu |
| Osoby | 47 grup, jawnie 201/202 razem | patient_id i oficjalne strat_fold | Zachować bazę jako namespace ID; brak potwierdzonych relacji osób między bazami |
| Warunki | Długie ambulatoryjne zapisy, taśma, artefakty i możliwy skew kanałów | Krótkie kliniczne standardowe 12-lead ECG | Istotny domain shift, którego resampling/normalizacja nie usunie |

Nie pobrano ani nie audytowano waveformów PTB-XL. Wartości z tej kolumny są zweryfikowane w dokumentacji i jednym nagłówku, a nie z pomiarów wszystkich sygnałów. Odprowadzenia każdego przyszłego pliku trzeba sprawdzić, nie zakładać kolejności na podstawie samej tabeli.

## Czy wspólny moduł jest możliwy?

**Tak, jako współdzielone czyste operacje numeryczne i osobne adaptery baz; nie jako automatyczne scalenie etykiet.** Istniejące filtry i ECGNormalizer przyjmują fs i tablice `[samples, channels]`, więc ich matematykę można wykorzystać także dla innych kanałów po świadomym doborze konfiguracji. Obecny loader jest specyficzny dla lokalnych trzycyfrowych rekordów MIT-BIH; PTB potrzebuje własnego adaptera ścieżek i metadanych. Nie implementowano go tutaj.

Oddzielić przyszłe konfiguracje `heartbeat-mitdb-v1` i `rhythm-proposed-v1`; zachować source recording, bazę, oryginalne kanały, fizyczne jednostki, oryginalne czasy, provenance etykiet i jawne typy missing/uncertain. Nie tworzyć domyślnych klasyfikacyjnych nazw z samych progów tętna.

## Proponowana kolejność przyszłej fazy danych

1. Pełne CSV PTB-XL: kontrola wersji i kompletności, SCP, patient_id, recommended folds, jakości/validation i co-labels. Najpierw wykonalność klas, potem sygnały.
2. Uzgodnienie z Anną jednostki obserwacji i polityki etykiet. Dla PTB naturalnym punktem wyjścia jest pełny rekord 10 s z zachowaniem wielu wpisów. Ostateczna lista klas i ich indeksy wymaga pełnych metadanych; teraz pozostaje propozycją.
3. Dopiero po zatwierdzeniu mały pilotaż sygnałów: prawidłowość skali WFDB, kanałów i jakości, brak clippingu, długości i NaN/Inf. Bez pełnego downloadu w fazie 11.
4. Osobna implementacja adaptera rytmu, testy alignment i przyszły eksport. Nie zmieniać pipeline’u heartbeat. Anna ustala później model i podział pacjentów.

## Okna i granice rytmów — propozycja

MIT: rozważyć 10-sekundowe fragmenty całkowicie wewnątrz jawnie opisanych stanów rytmu. Nie przekraczać UNKNOWN ani przejść; propozycja wymaga jeszcze wyboru marginesu bezpieczeństwa wokół granic. Krótsze epizody zachować w audycie i ewentualnie osobnym celu krótkich napadów, nie dopełniać ich do 10 s etykietą z innego rytmu. Finalny stan przy EOF jest cenzorowany. Powtarzające się/overlapping okna tej samej osoby nie są niezależnymi osobami. Nie generowano tych okien w tym etapie.

PTB: pojedynczy pełny rekord 10 s nie ma granic rytmu. Zachować etykietę jako statement-level/weak temporal label. Nie udawać pewności, że każda próbka zawiera ten sam rytm. Rekordy z mnogimi rytmami wymagają jawnej polityki multi-label albo przeglądu i udokumentowanego wykluczenia w przyszłym, osobnym wariancie. Nie wybierać etykiety wyłącznie na podstawie maksymalnej likelihood, gdy 0 oznacza unknown.

## Fs, kanały i filtry — parametry do zatwierdzenia, nie zmiana defaultów

- Dla prostego przyszłego baseline PTB można rozważyć udostępnioną wersję 100 Hz i zachowanie wszystkich 12 odprowadzeń w tym zadaniu. Wersja 500 Hz jest alternatywą, jeżeli potrzebna będzie większa szczegółowość; to nie zostało rozstrzygnięte eksperymentem.
- Dla ewentualnego wspólnego zadania z MIT trzeba jawnie ustalić docelowe fs i resampling antyaliasingowy oraz przeliczenie współrzędnych czasu. Nie wystarczy skrócić wektor ani powielić próbki. Żadnego resamplingu teraz nie implementowano.
- Nie redukować 12 kanałów do dwóch po samej pozycji indeksu. Dobór odprowadzeń i ograniczenia MLII vs II należy omówić, nie maskować ich nazwą. Utrzymywać dwa kontrakty wejścia jest prostsze niż wymuszać jeden model.
- Dotychczasowe 0,5–45 Hz jest zweryfikowane technicznie dla heartbeat MIT, nie jako optymalny zestaw dla wszystkich zadań PTB. Dla rytmu trzeba obejrzeć zachowanie P, QRS i czasów RR oraz artefakty na 10-sekundowym rekordzie. Filtr może tłumić ważne szczegóły morfologii; to wymaga kontroli, bez deklaracji klinicznej poprawności.
- Przy 100 Hz Nyquist to 50 Hz: nie dokładać notch 50/60 Hz po downsamplingu, używać kontroli widma i wiedzy o aliasingu. Istniejący filtr odrzuca notch równy/wyższy od Nyquista. Nie wymuszać historycznych ustawień sygnału 360/500 Hz na 100 Hz.
- Zero-phase nadal jest offline; krótki zapis ma inne problemy brzegowe niż długi Holter. Dla MIT można filtrować pełny dopuszczony kontekst przed wyświetleniem/wycięciem fragmentu; dla PTB dysponujemy tylko 10 s. Nie kopiować kontekstu przez granice niezależnych zapisów lub przyszłych podziałów.

## Normalizacja

Obecny heartbeat dataset pozostaje bez zmian: osobne mean/std pełnego ok. 30-minutowego rekordu i kanału. Proste zastosowanie identycznej funkcji do PTB da statystyki z 10 s, a nie z 30 min — formalnie ta sama transformacja nie oznacza tej samej domeny.

Dla przyszłego, odrębnego rytmowego kontraktu można rozważyć Z-score per 10-sekundowa obserwacja/per kanał jako jawny baseline offline. Jest to propozycja, nie wykonana zmiana ustawień. Usuwa ona skalę fizyczną i może wzmacniać szum w kanałach o niskiej amplitudzie; wymaga obecnej polityki kanałów stałych i przeglądu jakości. Alternatywą są zamrożone parametry global training-only; te wymagają wcześniejszego splitu Anny, więc nie są tutaj dopasowane. Nie wyliczać wspólnych parametrów na całej połączonej bazie lub na walidacji/test.

Każdy przyszły eksport musi jednoznacznie opisywać fs, jednostki, kanały, zakres statystyk, kontekst filtracji, granice okna i semantykę etykiety. W ten sposób nowa procedura może być konsekwentnie odtworzona bez niejawnego wykorzystywania przyszłych próbek niedostępnych w docelowym użyciu.

## Bramka danych przed następną fazą

Dla każdej proponowanej klasy potrzebne są rzeczywiste częstości i liczby pacjentów PTB-XL, jakość labeli, współwystępowanie oraz długość/zakres sygnału. Brakujących rytmów i osób nie tworzymy przez padding, oversampling ani progi HR. Nie wolno nazywać nieopisanych odcinków „normal”. Brak niezależnych pacjentów dla SBR/10s-VT w MIT oznacza ograniczenie wykonalności, a nie błąd do naprawienia losowym splitem.

Najbliższy etap powinien pozostać metadanymi PTB-XL; pełne preprocessing i model wymagają osobnej zgody. Po uzyskaniu dostępu można pobrać tylko dwa pliki CSV do nowego katalogu, bez sygnałów:

```bash
cd ~/Pulpit/ECG-APP
.venv/bin/python - <<'PY'
from pathlib import Path
from urllib.request import urlopen
folder = Path('results/preprocessing/phase11_ptb_metadata_download')
folder.mkdir(parents=True, exist_ok=False)
for name in ('ptbxl_database.csv', 'scp_statements.csv'):
    with urlopen('https://physionet.org/files/ptb-xl/1.0.3/' + name, timeout=60) as response:
        content = response.read(10 * 1024 * 1024 + 1)
    if len(content) > 10 * 1024 * 1024:
        raise ValueError('Unexpected metadata size; refusing to download more')
    with (folder / name).open('xb') as output:
        output.write(content)
PY
.venv/bin/python -m examples.audit_rhythm_datasets \
  --ptb-database results/preprocessing/phase11_ptb_metadata_download/ptbxl_database.csv \
  --scp-statements results/preprocessing/phase11_ptb_metadata_download/scp_statements.csv \
  --output-dir results/preprocessing/phase11_rhythm_audit_with_metadata
```

To instrukcja na przyszłość; pobranie w bieżącym środowisku nie powiodło się z powodu DNS. Sam audyt CSV nie trenuje i nie przetwarza waveformów. Liczby i hashe plików należy potem jawnie dołączyć do nowej wersji audytu, zachowując tę wersję jako dokumentację obecnego stanu wiedzy.

## Źródła

- https://physionet.org/content/ptb-xl/1.0.3/
- https://physionet.org/content/ptb-xl/1.0.3/scp_statements.csv
- https://physionet.org/content/ptb-xl/1.0.3/records500/00000/00001_hr.hea
- https://archive.physionet.org/physiobank/database/html/mitdbdir/intro.htm
- https://archive.physionet.org/physiobank/annotations.shtml
- Lokalny audyt adnotacji WFDB w `results/preprocessing/phase11_rhythm_audit/`.
