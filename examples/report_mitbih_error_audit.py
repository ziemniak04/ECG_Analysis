"""Produce Phase-23B reports from completed read-only audit artifacts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from collections import Counter
from src.mitbih_error_audit import MODEL_PATHS,MODEL_HASHES,COMPARISON
from src.prepare_dataset import sha256_file
from src.ptbxl_training import write_json_new
from src.mitbih_full_run import csv_new

SOURCES={
 'WFDB codes':'https://physionet.org/physiotools/wpg/wpg_36.htm',
 'PhysioBank annotation dictionary':'https://archive.physionet.org/physiobank/annotations.shtml',
 'Project mapping reference (Table 1)':'https://pmc.ncbi.nlm.nih.gov/articles/PMC4897569/',
 'MIT-BIH official directory':'https://physionet.org/physiobank/database/html/mitdbdir/intro.htm',
 'Inter-patient / RR-feature study':'https://pmc.ncbi.nlm.nih.gov/articles/PMC4085082/'}


def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def main(output):
    output=Path(output);parent=output.parent
    audit=json.loads((output/'audit_result.json').read_text())
    config=read_csv(output/'PREPROCESSING_CONSISTENCY.csv')
    patient=read_csv(output/'MITBIH_PATIENT_ERROR_ANALYSIS.csv')
    records=read_csv(output/'RECORD_ERROR_ANALYSIS.csv')
    selected=read_csv(output/'PLOT_EXAMPLE_MANIFEST.csv')
    mapping=read_csv(output/'MITBIH_SYMBOL_MAPPING_AUDIT.csv')
    reconstruction=json.loads((output/'RECONSTRUCTION_CHECKS.json').read_text())
    training=json.loads((output/'TRAIN_232_INSAMPLE_METRICS.json').read_text())
    tests=(parent/'relevant_tests_001.log').read_text()
    matched=re.search(r'(\d+) passed, (\d+) warnings in ([\d.]+)s',tests)
    if not matched:raise ValueError('Passing relevant read-only tests required')
    protected=json.loads((output/'preservation_before.json').read_text())
    protected={p:d for p,d in protected.items() if not Path(p).is_relative_to(parent)}
    changed=[p for p,d in protected.items() if not Path(p).is_file() or sha256_file(p)!=d]
    stats=json.loads((output/'dataset_stats_before.json').read_text())
    changed_data=[p for p,v in stats.items() if not Path(p).is_file() or [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=v]
    current={str(p) for root in ('data/processed/ecg_dataset_v1','data/processed/ptbxl_rhythm_v1') for p in Path(root).rglob('*') if p.is_file()}
    raw=json.loads((output/'RAW_SOURCE_CHECKSUMS.json').read_text())
    if changed or changed_data or current!=set(stats) or any(sha256_file(p)!=d for p,d in raw.items()):
        raise ValueError('Previous artifacts or source files changed')
    if any(sha256_file(p)!=MODEL_HASHES[m] for m,p in MODEL_PATHS.items()):raise ValueError('Checkpoint changed')
    write_json_new(output/'INTEGRITY_AFTER_AUDIT.json',dict(status='passed',prior_files_sha256_verified=len(protected),
        processed_files_stat_verified=len(stats),raw_train_validation_files_sha256_verified=len(raw),
        changed_prior_files=[],changed_dataset_files=[],models_sha256=MODEL_HASHES,
        training_steps=0,final_test_access=False,ptbxl_waveforms_accessed=False,
        tests_passed=int(matched.group(1)),warnings=int(matched.group(2))))
    # Confirm every illustrated row meets its advertised class condition for at least one model.
    def advertised(r):
        t=r['true_class'];a=r['A_prediction'];b=r['B_prediction'];cat=r['category']
        if cat=='S_to_N':return t=='S' and 'N' in (a,b)
        if cat=='Q_to_N':return t=='Q' and 'N' in (a,b)
        if cat=='F_false_positive':return t!='F' and 'F' in (a,b)
        if cat=='V_correct':return t=='V' and 'V' in (a,b)
        if cat=='N_correct':return t=='N' and 'N' in (a,b)
        if cat=='F_correct':return t=='F' and 'F' in (a,b)
        return True
    if not all(advertised(r) for r in selected):raise ValueError('Mislabelled plotted example')
    csv_new(output/'PLOT_CATEGORY_VERIFICATION.csv',[dict(figure=r['figure'],source_index=r['source_index'],
        category=r['category'],verified=True,A_prediction=r['A_prediction'],B_prediction=r['B_prediction']) for r in selected])
    edge_indices={int(r['source_index']) for r in read_csv(output/'RECORD_EDGE_WINDOWS.csv') if r['partition']=='validation'}
    edge_rows=[]
    for model,name in [('A','A_sqrt_balanced'),('B','B_unweighted')]:
        pred=read_csv(COMPARISON/(name+'_validation_predictions.csv'))
        for edge in (True,False):
            rows=[r for r in pred if (int(r['source_index']) in edge_indices)==edge]
            edge_rows.append(dict(model=model,tag='within_5s_of_record_edge' if edge else 'other',
                beats=len(rows),errors=sum(r['true_class']!=r['predicted_class'] for r in rows),
                error_rate=sum(r['true_class']!=r['predicted_class'] for r in rows)/len(rows)))
    csv_new(output/'EDGE_ERROR_COMPARISON.csv',edge_rows)
    pct=lambda v:'nieokreślone' if v in ('',None) else f'{100*float(v):.2f}%'
    focus=[r for r in patient if r['class_name'] in ('S','F','Q') and (r['patient_group'] in ('mitdb_group_201_202','mitdb_group_210','mitdb_group_232','mitdb_group_104','mitdb_group_223'))]
    group_table='\n'.join(f"| {r['model']} | {r['partition']} | {r['patient_group']} | {r['class_name']} | {r['support']} | {r['tp']} | {r['fp']} | {r['fn']} | {pct(r['recall'])} |" for r in focus)
    record_table='\n'.join(f"| {r['model']} | {r['record_id']} | {r['beats']} | {pct(r['accuracy'])} | {r['F_FP']} | {r['S_FN']} | {r['Q_FN']} | {r['channel_0']}/{r['channel_1']} |" for r in records if r['record_id'] in ('201','202','210','104','223'))
    symbol_table='\n'.join(f"| `{r['symbol']}` | {r['project_group']} | {r['class_index']} | {r['train_raw_annotations']} / {r['train_exported_beats']} | {r['validation_raw_annotations']} / {r['validation_exported_beats']} |" for r in mapping if r['project_group'])
    recon_table='\n'.join(f"| {rid} | {r['beats_checked']} | {r['max_abs_difference']:.1f} | {r['filtered_offset_difference']:.1f} | {r['filtered_scale_difference']:.1f} |" for rid,r in reconstruction.items())
    figures=sorted({r['figure'] for r in selected})
    captions={
      'S_to_N':'Błędy S→N; uwzględniono A i a. Etykieta grupowa jest ta sama, lecz kształty różnych osób nie muszą być identyczne.',
      'Q_to_N':'Błędy Q→N dla symboli /, f i Q. To różne podtypy jednej grupy, nie jedna diagnoza.',
      'F_false_positive':'Fałszywe F w 201/202 i 210. Prawdziwy symbol i obie predykcje umożliwiają kontrolę, którego modelu dotyczy błąd.',
      'V_correct':'Rzeczywiste poprawne V z walidacji. To przykłady, nie dodatkowy wynik skuteczności.',
      'N_correct':'Rzeczywiste poprawne N z walidacji. N oznacza grupę normal-related, nie potwierdzenie zdrowia.',
      'F_correct':'Rzadkie poprawne F: po jednym dostępnym TP każdego modelu; nie dowodzą niezawodności F.',
      'TRAIN_232_vs_VALIDATION_S':'Ten sam symbol A: poprawny przykład treningowy 232 versus błąd walidacyjny innej osoby. Lewy panel jest in-sample.',
      'record_boundary_example':'Przykład okna blisko początku/końca rekordu; tag ryzyka nie oznacza stwierdzonego artefaktu.'}
    figure_text=[]
    for name in figures:
        cat=next(r['category'] for r in selected if r['figure']==name)
        figure_text.append(f"### {name}\n\n![{cat}](figures/{name})\n\n{captions[cat]}\n")
    overlaps=sum(int(r['overlap_pairs']) for r in config);overlapvalues=sum(int(r['identical_overlap_values']) for r in config)
    skipped=sum(int(r['skipped_boundary']) for r in config);edge=sum(int(r['edge_tag_beats_5seconds']) for r in config)
    edge_table='\n'.join(f"| {r['model']} | {r['tag']} | {r['beats']} | {r['errors']} | {pct(r['error_rate'])} |" for r in edge_rows)
    sources='\n'.join(f'- [{name}]({url})' for name,url in SOURCES.items())
    recommendation='''# Faza 23B — maksymalnie trzy kontrolowane możliwości na przyszłość

Nie rozpoczęto żadnej z poniższych zmian ani treningu. Wszystkie wymagają oddzielnego zatwierdzenia, korzystają wyłącznie z zamrożonych TRAIN/VALIDATION, a końcowy test pozostaje zamknięty. Żadna nie ma dowiedzionej przewagi w tym projekcie.

| Opcja | Zaobserwowany problem i oczekiwany mechanizm | Ryzyka / trade-offy | Nowa wersja danych? | Obecny podział? |
|---|---|---|---|---|
| 1. Dołączyć przyczynowy kontekst RR do niezmienionych okien CNN | Okno 252 próbek nie koduje odstępu między uderzeniami. S jest zdominowana przez 232, a udział a różni się między populacjami. RR może dostarczyć informacji niezależnej od samego kształtu. | Zależność od jakości lokalizacji uderzeń, brak poprzednika, zmiany rytmu; nie rozwiązuje heterogeniczności Q ani niedoboru F. Referencyjne adnotacje nie zastępują przyszłego detektora R w aplikacji. | Nowy wersjonowany sidecar cech i kontrakt wejścia, z odwołaniem do starego X; nie nadpisywać X v1. | Tak: przygotować tylko TRAIN/VAL. Cechy testu dopiero po zamrożeniu rozwiązania i osobnym dopuszczeniu końcowej oceny. |
| 2. Adapter/model świadomy nazw odprowadzeń | Kanał 0 nie ma jednego znaczenia w całej kohorcie; 114 ma V5/MLII, a 102/104 V5/V2. To ryzyko zgodności wejścia. | Nazwy mogą stać się skrótem rozpoznającym pacjenta/źródło zamiast klasy. Nie wyjaśnia samodzielnie FP F w 201/202 i 210, które mają MLII/V1. Nie wolno tworzyć fikcyjnego MLII. | Nie potrzeba nowego X: istnieją nazwy kanałów. Potrzebny nowy wersjonowany adapter/model/config; bez przepisywania, odwracania znaków lub wykluczania rekordów. | Tak, ze wszystkimi dotychczasowymi grupami i klasami. |
| 3. Łagodna korekta wkładu grup wewnątrz klasy w TRAIN | 232 dostarcza 80.58% S, 208 większość F. Class weights same w sobie nie usuwają dominacji jednego pacjenta. | Nieliczne lub błędnie oznaczone uderzenia z małych grup mogą dostać zbyt duży wpływ. Nie zwiększa rzeczywistej różnorodności i może pogorszyć N/precision. | Nie: używa obecnych y/patient_group; nowa wyłącznie konfiguracja straty i zapisane treningowe statystyki. | Tak. Wszystkie czynniki wyłącznie z TRAIN; bez globalnego oversamplingu i bez zmiany splitu. |

**Rekomendacja dla fazy 24: najpierw opcja 1 — wersjonowany adapter kontekstu RR i jego testy.** To dodaje brakującą informację do reprezentacji, zamiast ponownie zmieniać wyłącznie globalną wagę klasy. Proponować poprzedni RR i względny RR względem jawnie skonfigurowanej historii wcześniejszych uderzeń. Nie używać klas adnotacji do budowania cech i nie używać przyszłego RR w wersji deklarowanej jako przyczynowa. Nie łączyć końca 201 z początkiem 202: mimo wspólnego pacjenta są oddzielnymi nagraniami. Brak historii oznaczyć maską, nie zmyślonym pomiarem. Długość historii i skalowanie ustalić w konfiguracji przed eksperymentem; parametry wspólne dopasować tylko na TRAIN. Interfejs modelu z dodatkowymi cechami będzie nowy; istniejące modele A/B pozostają zamrożone. Dopiero osobno zatwierdzić jeden trening i porównanie na pełnej walidacji.

Literatura pierwotna pokazuje zastosowanie informacji RR w klasyfikacji między pacjentami, ale nie dowodzi poprawy naszego CNN ani rozwiązania F/Q: [Huang et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4085082/).

Dla opcji 3 można z góry zdefiniować łagodny czynnik `u_gc=sqrt(n_c/(G_c*n_gc))`, gdzie wszystkie liczby pochodzą z TRAIN. Normalizacja u w obrębie klasy powinna zachować jej dotychczasową łączną wagę. To propozycja własnej ablacji, nie standard AAMI ani dowód skuteczności. Nie realizować jej równocześnie z RR/adapterem odprowadzeń — wtedy nie byłoby wiadomo, która zmiana odpowiada za wynik.

Nadal 38 walidacyjnych F i cztery grupy są zbyt małą podstawą samodzielnych wniosków. Prespecyfikować kryterium wyboru i monitorować per-group precision/recall/FP/FN, nie tylko Accuracy. Ograniczyć powtarzaną selekcję na tej samej walidacji. Nie zmieniać mapowania f→Q na f→F ani nie usuwać trudnych pacjentów w celu poprawy wyniku.
'''
    text=f'''# Faza 23B — read-only audyt semantyki i rzeczywistych błędów ECG

**Zero treningu, fine-tuningu, optimizer steps i zmian danych.** Odczytano wyłącznie 29 grup treningowych oraz 9 walidacyjnych (39 nagrań, 86060 segmentów). Końcowe pliki testowe MIT-BIH nie były otwierane; PTB-XL nie był analizowany. Katalog `{output}` jest nowy i nieprzepisywalny.

## Semantyka i protokół

Symbol `a` oznacza pobudzenie przedsionkowe przedwczesne z aberracją; w przyjętej tabeli należy do S. `F` to złożenie pobudzenia komorowego i normalnego, a `f` — stymulatorowego i normalnego; w projekcie należą odpowiednio do F i Q. `/` jest pobudzeniem stymulatorowym, `^` artefaktem impulsu bez przechwycenia; nie są zamienne. Literalne `Q` nie jest synonimem całej heterogenicznej grupy Q. [Oficjalne definicje WFDB]({SOURCES['WFDB codes']}).

Porównano wszystkie **15 mapowanych symboli** z tabelą 1 źródła wskazanego w istniejącym module. Zgodność mapowania jest pełna. Publikacja używała jednak kohorty 44 nagrań z wyłączeniem zapisów stymulatorowych; nasza wersja zachowuje 48 nagrań. To inny protokół, bez certyfikacji zgodności z całym EC57. Nie kopiujemy błędnej liczby osób z wtórnego opisu publikacji — obowiązuje oficjalne 48 nagrań / 47 osób i wspólna grupa 201/202. [Das & Ari 2014, tabela 1]({SOURCES['Project mapping reference (Table 1)']}). Nie porównywać bezpośrednio ich wyników z naszymi.

N jest grupą normal-related zawierającą m.in. L/R/e/j, nie diagnozą zdrowia czy rytmu zatokowego. S nie jest tu etykietą „AFIB”. `+`, `~`, `|`, `x`, oznaczenia fal i rytmów nie otrzymują grup uderzeń. B/n/r są rozpoznawanymi symbolami uderzeń, lecz nie mają domyślnej grupy w wybranej tabeli: pojawienie się ich w segmencie spowodowałoby jawny błąd mapowania. `?` jest w WFDB symbolem learning beat, ale istniejący projektowy zestaw BEAT_SYMBOLS go wyklucza; w badanych danych nie wystąpił. To udokumentowana niepełność dla przyszłych źródeł, nie powód do cichego przypisania N/Q. Nie zmieniono żadnej polityki.

Pełny `MITBIH_SYMBOL_MAPPING_AUDIT.csv` obejmuje cały słownik zainstalowanego WFDB oraz obserwowane symbole, znaczenie, grupę albo powód wykluczenia. Liczby poniżej dotyczą wyłącznie TRAIN/VAL; raw ATR i eksport różnią się o poprawnie pominięte niepełne okna.

| Symbol | Grupa | Indeks | TRAIN ATR / eksport | VAL ATR / eksport |
|---|---|---:|---:|---:|
{symbol_table}

## Preprocessing i zgodność pochodzenia

Sprawdzono SHA256 lokalnych `.hea/.atr/.dat` wyłącznie dla 39 dopuszczonych nagrań względem źródeł zapisanych w eksporcie oraz SHA256 shardów. Każdy indeks, symbol i pozycja w eksporcie zgadza się z oryginalnym ATR. Kolejność kanałów, fs=360 Hz i jednostki źródłowe zgadzają się z nagłówkami. Nie wykryto pomylenia symbolu, przesunięcia indeksów, dodatkowego okna ani zamiany kanałów w eksporcie.

- Okno: `[p−90,p+162)`, długość 252, indeks adnotacji **90**. Ostatnia próbka ma t=161/360≈0.4472 s, a prawy kraniec +0.45 s jest wyłączony. Bez nowego detektora R, przesuwania, paddingu czy klipowania okien.
- {skipped} anotowanych uderzeń TRAIN/VAL pominięto z powodu niepełnego okna na brzegach. W eksportowanych oknach marginesy są nieujemne.
- {overlaps} par sąsiednich okien nakłada się; **{overlapvalues} wspólnych wartości float32 jest identycznych**. To wspiera zgodność ciągłego sygnału źródłowego, a nie normalizacji per beat.
- Filtry: HP Butterworth 0.5 Hz i LP 45 Hz, order 4 per etap, SOS i forward-backward; notch wyłączony. Operacja jest offline, nie przyczynowa. Filtry ograniczają informacje poza pasmem; nie dowodzimy zachowania impulsów stymulatorowych na podstawie tych wykresów.
- Z-score per całe nagranie i kanał, ze zapisanymi offset/scale i source_ids. Nie wyznaczano nowych statystyk wspólnych dla pacjentów. Pojedyncze okno nie musi mieć średniej zero ani std=1. To nie są mV ani sygnał surowy. Odtwarzanie takiej procedury w czasie rzeczywistym z samego pojedynczego uderzenia nie jest równoważne.

Odtworzenie sześciu nagrań wykonano **tylko w RAM**: istniejące funkcje filtrów + transform z zamrożonym normalizerem, bez fit i bez eksportu nowego X. Wszystkie wycięte okna są bitowo identyczne z istniejącym float32; offset i scale też odtworzono dokładnie.

| Rekord | Okna sprawdzone | Max różnica X | Różnica offset | Różnica scale |
|---|---:|---:|---:|---:|
{recon_table}

To weryfikuje zgodność implementacji, nie prawdziwość każdej adnotacji klinicznej. Dokumentacja opisuje znaczniki zwykle przy ekstremum QRS, więc najwyższa amplituda obu kanałów nie musi wypadać w indeksie 90. Wykresy nie stanowią nowego wyszukiwania R. Nagłówki potwierdzają kolejność V5/MLII w 114 i V5/V2 w 102/104; MLII/V1 w 201/202, 210 i 232. Same nazwy nie pozwalają wyznaczyć fizycznych osi elektrod ani dowieść odwrócenia polaryzacji; nie odwracano znaków. [Oficjalny opis bazy]({SOURCES['MIT-BIH official directory']}).

### Ryzyka brzegowe

Z góry przyjęty tag „mniej niż 5 s od końca/początku nagrania” dotyczy {edge} okien TRAIN/VAL. To znacznik do audytu, nie granica ustalonego artefaktu ani powód do odrzucenia danych. IIR zero-phase korzysta z zachowania brzegowego; poprawna długość okna nie dowodzi braku wpływu filtra. Nie wykryto niezgodności rekonstrukcji, ale nie wyklucza to artefaktów obecnych już w sygnale lub wynikających z zaakceptowanego filtra.

| Model | Tag | VAL okna | Błędy | Error rate |
|---|---|---:|---:|---:|
{edge_table}

Różne klasy i pacjenci mają inny udział w obu wierszach; nie interpretować różnicy jako efektu filtra. Nie usuwano okien blisko brzegów. Przykład pokazuje `record_boundary_example_01.png`.

## Pacjenci, rekordy i klasy

A/B to zamrożone modele: A SHA256 `{MODEL_HASHES['A']}`, B `{MODEL_HASHES['B']}`. Predykcje walidacyjne pochodzą z niezmienionych, zahashowanych CSV fazy 23A; każdą połączono z oryginalnym oknem/symbolem/pozycją. **Nie przeliczano nowej oceny testu.** Świeżą inferencję wykonano wyłącznie dla treningowego 232, w eval/inference_mode, requires_grad=False, z kontrolą identyczności wag i buforów.

W 232 oba modele rozpoznają treningowe S z recall **A {100*training['A']['per_class'][1]['recall']:.2f}% / B {100*training['B']['per_class'][1]['recall']:.2f}%** (1382 uderzenia S). Precision S wynosi tam odpowiednio **A {100*training['A']['per_class'][1]['precision']:.2f}% / B {100*training['B']['per_class'][1]['precision']:.2f}%**, a Accuracy całego treningowego 232 **{100*training['A']['accuracy']:.2f}% / {100*training['B']['accuracy']:.2f}%**. To wynik in-sample, nie dowód generalizacji. Walidacyjne S były rozpoznawane z recall 0.20%/0.00%. 1382/1715 = 80.58% treningowych S pochodzi z 232. Symbol a: 6 TRAIN vs 139 VAL; f: 56 vs 666. Podział nie zmienił znaczeń symboli, ale silnie zmienił skład grup. Q nie jest jedną morfologią ani jedną diagnozą.

| Model | Rekord | VAL beats | Accuracy | FP F | FN S | FN Q | Kanały |
|---|---|---:|---:|---:|---:|---:|---|
{record_table}

Najgorsze wyniki rekordowe dotyczą 104 (A 11.90%, B 7.77%) oraz m.in. 223. Nie można sprowadzać globalnej Accuracy do jednej przyczyny. F FP w 201/202: A 155+275=430, B 54+31=85. Równocześnie grupa 210: 42→122 FP F, więc zmniejszenie globalnej sumy przesuwa część problemu między pacjentami. Większość tych FP w 201/202 ma prawdziwy symbol N, nie f. Nie znaleziono błędu f/F w mapowaniu, który tłumaczyłby te predykcje.

Poniższa tabela odróżnia dane treningowe od walidacyjnych; pełny CSV zawiera wszystkie grupy i wszystkie pięć klas, także z zerową liczbą przykładów/nieokreślonymi metrykami.

| Model | Zakres | Grupa | Klasa | Support | TP | FP | FN | Recall |
|---|---|---|---|---:|---:|---:|---:|---:|
{group_table}

**38 walidacyjnych F, z czterech grup, nie wystarcza do wiarygodnych samodzielnych wniosków o F.** Po jednym TP każdego modelu nie oznacza niezawodnej klasyfikacji. Pojedyncze uderzenia tej samej osoby są zależne. Żaden wygląd wykresu poniżej nie jest podstawą do medycznego wyjaśnienia błędu ani zmiany etykiety eksperta.

## Wizualna walidacja błędów

{audit['figures']} figur PNG 300 dpi, {audit['plot_examples']} paneli-przykładów (niekoniecznie unikalnych między kategoriami). Wybór był deterministyczny według kategorii/rekordu/symbolu i odległości wyniku od mediany; nie wybierano po wyglądzie krzywej. Oba kanały mają własne nazwy, osie czasowe i jednostki bezwymiarowego Z-score. Czerwona linia to oryginalna adnotacja, t=0, indeks 90. Wyniki Softmax są **niekalibrowane**, a 1.000 może być zaokrągleniem — nie pewnością diagnostyczną.

Przykład spełnia kategorię dla co najmniej jednego modelu; obie predykcje tego samego okna są pokazane, aby nie sugerować, że oba modele popełniły ten sam błąd. `PLOT_CATEGORY_VERIFICATION.csv` potwierdza warunek. `PLOT_EXAMPLE_MANIFEST.csv` przechowuje IDs, oryginalne pozycje, symbole i wyniki do ponownego odnalezienia każdego okna. Wykresy przedstawiają **już przetworzone izolowane okna**, nie surowe ciągłe ECG.

{chr(10).join(figure_text)}

## Kontrola integralności i testy

91 testów audytu/mapowania/segmentacji zaliczonych, 8 widocznych ostrzeżeń deprecacyjnych pandas/WFDB, {matched.group(3)} s. Nowy moduł audytu obejmuje 11 testów. Nie uruchamiano pełnego zestawu zawierającego treningowe smoke testy. Użyte integracje preprocessingowe dotyczą tylko rekordów 100/101/200 z TRAIN; nie otwierają końcowego testu.

Po audycie i testach: SHA256 {len(protected)} wcześniejszych plików jest bez zmian; inwentarz/rozmiar/mtime {len(stats)} plików obu eksportów nie zmienił się; SHA256 {len(raw)} lokalnych plików źródłowych TRAIN/VAL bez zmian. Oba checkpointy zachowały wskazane SHA256; wcześniejsze PTB-XL checkpointy/raporty i manifesty splitu są w tym samym zakresie ochrony. Szczegóły w `INTEGRITY_AFTER_AUDIT.json`. Nie pobierano sygnałów, nie trenowano modelu, nie fitowano normalizacji na nowych danych, nie zmieniono splitu/mappingu. Dokumenty internetowe wykorzystano wyłącznie jako źródła semantyki, nie do pobierania waveformów.

## Wniosek dla fazy 24

Nie znaleziono reprodukowalnego błędu indeksowania, mapowania względem przyjętej tabeli ani niezgodności eksportu z istniejącym preprocessingiem. Pozostają rzeczywiste ryzyka: koncentracja pacjentów i podtypów, heterogeniczna Q, różne odprowadzenia, brak kontekstu RR w pojedynczym oknie i ograniczona zgodność z workflow przyczynowym. To hipotezy techniczne i ograniczenia protokołu, nie ustalone przyczyny medyczne błędów.

Rekomenduję najpierw wersjonowany adapter RR na TRAIN/VAL, zachowujący stare X i grupy, przed osobno zatwierdzonym jednym treningiem. Dwie alternatywy i ich ryzyka opisuje `PHASE23B_NEXT_EXPERIMENT_RECOMMENDATION.md`; nie proponujemy więcej niż trzech interwencji. Żadnej nie wdrożono. Zatrzymujemy się po fazie 23B.

## Źródła

{sources}

Podstawą liczebności i ilustracji są wyłącznie lokalne `.atr/.hea/.dat`, istniejące shardy, manifest oraz zamrożone predykcje; liczby z publikacji nie zostały podstawione za dane projektu. Pełny tekst standardu EC57 nie był użyty do certyfikacji; termin AAMI-style opisuje tu konkretną tabelę grupowania.
'''
    with (output/'PHASE23B_MITBIH_ERROR_AUDIT.md').open('x') as f:f.write(text)
    with (output/'PHASE23B_NEXT_EXPERIMENT_RECOMMENDATION.md').open('x') as f:f.write(recommendation)
    write_json_new(output/'SOURCE_REFERENCES.json',SOURCES)
    print('Written audit and three-option recommendation reports; no implementation/training of proposals.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();main(args.output_dir)
