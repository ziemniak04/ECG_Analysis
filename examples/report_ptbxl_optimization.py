"""Write Phase-18C Polish reports only from completed comparisons and pytest logs."""
import argparse
import csv
import json
from pathlib import Path
import re


def reports(root):
    root=Path(root)
    summary=json.loads((root/'evaluation_summary.json').read_text())
    if summary['status']!='passed':raise ValueError('Completed verified comparison required')
    test_log=(root/'full_pytest.log').read_text()
    relevant_log=(root/'relevant_tests.log').read_text()
    # Per-file warning groups precede the total; read the final pytest summary.
    full_summary=next((line for line in reversed(test_log.splitlines()) if re.search(r'\d+ passed',line)), '')
    relevant_summary=next((line for line in reversed(relevant_log.splitlines()) if re.search(r'\d+ passed',line)), '')
    total=re.search(r'(\d+) passed',full_summary)
    relevant=re.search(r'(\d+) passed',relevant_summary)
    warnings=re.search(r'(\d+) warnings',full_summary)
    if not total or not relevant or re.search(r'\d+ failed',test_log+relevant_log):
        raise ValueError('Both test runs must finish successfully')
    count=int(total[1]);warning_count=int(warnings[1]) if warnings else 0
    rows=list(csv.DictReader((root/'PHASE18C_MODEL_COMPARISON.csv').open()))
    classes=list(csv.DictReader((root/'PHASE18C_CLASS_METRICS.csv').open()))
    selection=summary['selection'];chosen=selection['selected']
    models=summary['models'];base=next(r for r in rows if r['model']=='A_balanced' and float(r['threshold'])==0)
    argmax=[r for r in rows if float(r['threshold'])==0]
    best_f1=max(argmax,key=lambda r:float(r['macro_f1_full_cohort']))
    fewest_raw=min(argmax,key=lambda r:int(r['SBRAD_fp']))
    fewest_all=min(rows,key=lambda r:int(r['SBRAD_fp']))
    fewest_raw_names=', '.join(r['model'] for r in argmax if int(r['SBRAD_fp'])==int(fewest_raw['SBRAD_fp']))
    fewest_all_names=', '.join(r['candidate'] for r in rows if int(r['SBRAD_fp'])==int(fewest_all['SBRAD_fp']))
    def value(v):return 'nieokreślone' if v is None or v=='' else f'{float(v):.6f}'
    def table(keys,source):
        return '\n'.join('| '+' | '.join(str(row[key]) if key in ('model','candidate','class','threshold','support','tp','fp','fn','tn','SBRAD_fp','abstained','best_epoch') else value(row[key]) for key in keys)+' |' for row in source)
    raw_table=table(('model','argmax_accuracy_full_cohort','macro_f1_full_cohort','macro_auprc_full_cohort','AFIB_recall','SBRAD_precision','SBRAD_recall','SBRAD_fp'),argmax)
    decision_table=table(('candidate','SBRAD_fp','SBRAD_precision','SBRAD_recall','coverage','abstained','macro_f1_full_cohort','accepted_accuracy_coverage_conditional','accepted_macro_f1_coverage_conditional'),rows)
    full_classes=table(('candidate','class','tp','fp','fn','tn','precision','recall','specificity','f1'),classes)
    duration_table='\n'.join(f"| {name} | {m['epochs']} | {m['best_epoch']} | {m['training_seconds']:.2f} | {m['training_seconds']/60:.2f} | {m['status']} |" for name,m in models.items())
    resource_rows=[]
    for name,m in models.items():
        if name=='A_balanced':continue
        run=Path(m['run'])
        resource=json.loads((run.parent/(run.name+'_audit')/'resource_usage.json').read_text())
        resource_rows.append(f"| {name} | {resource['peak_rss_bytes']/2**20:.2f} | {resource['minimum_available_ram_bytes']/2**30:.2f} | {resource['minimum_disk_free_bytes']/2**30:.2f} | {resource['violation'] or 'brak'} |")
    resource_table='\n'.join(resource_rows)
    weight_lines='\n'.join(f"- {name}: `{m['class_weights']}`." for name,m in models.items())
    epoch_tables=[]
    for name,m in models.items():
        data=list(csv.DictReader((Path(m['run'])/'epoch_metrics.csv').open()))
        body='\n'.join(f"| {r['epoch']} | {float(r['train_loss']):.6f} | {float(r['validation_loss']):.6f} | {float(r['validation_macro_f1']):.6f} | {float(r['validation_macro_auprc']):.6f} |" for r in data)
        epoch_tables.append(f"### {name}\n\n| Epoka | Train loss | Val loss | Val Macro F1 | Val Macro AUPRC |\n|---:|---:|---:|---:|---:|\n{body}\n")
    change_lines=[]
    for r in argmax:
        if r['model']=='A_balanced':continue
        change_lines.append(f"- {r['model']} względem A: ΔMacro F1 **{float(r['macro_f1_full_cohort'])-float(base['macro_f1_full_cohort']):+.6f}**, "
            f"ΔAUPRC **{float(r['macro_auprc_full_cohort'])-float(base['macro_auprc_full_cohort']):+.6f}**, "
            f"Δrecall AFIB **{100*(float(r['AFIB_recall'])-float(base['AFIB_recall'])):+.2f} pp**, "
            f"FP SBRAD **{int(base['SBRAD_fp'])}→{int(r['SBRAD_fp'])}**.")
    change_text='\n'.join(change_lines)
    if chosen:
        decision='argmax' if chosen['threshold']==0 else 'SBRAD >= 0,65, tylko gdy SBRAD wygrywa argmax; inaczej UNCERTAIN'
        coverage_note=('Przy coverage=1 zaakceptowano wszystkie 1958 rekordów, więc te wartości są również '
                       'metrykami całego foldu 9.' if chosen['abstained']==0 else
                       'Obie metryki są warunkowe względem coverage; nie opisują całego foldu 9.')
        recommendation=f"""**Rekomendowany kandydat: {chosen['model']}, reguła: {decision}.**

Checkpoint: `{chosen['checkpoint']}`.
SHA256: `{chosen['checkpoint_sha256']}`.
Pełnowalidacyjne Macro F1 z abstencjami jako pominięciami: **{chosen['macro_f1_full_cohort']:.6f}**,
Macro AUPRC pełnego zbioru: **{chosen['macro_auprc_full_cohort']:.6f}**,
recall AFIB: **{chosen['AFIB_recall']:.6f}**, recall SBRAD: **{chosen['SBRAD_recall']:.6f}**,
precision SBRAD: **{value(chosen['SBRAD_precision'])}**, FP SBRAD: **{chosen['SBRAD_fp']}**.
Coverage: **{chosen['coverage']:.6f}**, abstencje: **{chosen['abstained']}**.
Accuracy na zaakceptowanych: **{chosen['accepted_accuracy_coverage_conditional']:.6f}**,
Macro F1 na zaakceptowanych: **{chosen['accepted_macro_f1_coverage_conditional']:.6f}**.
{coverage_note}

Ścieżki przejścia przez kryterium: `{selection['stages']}`.
Konfigurację kandydata zapisano w `frozen_candidate.json`; nie uruchomiono jego oceny na teście.
"""
    else:
        recommendation='**Brak kandydata spełniającego wszystkie zapisane ograniczenia. Nie zaleca się nowego wdrożenia.**'
    test_section=f"""Po eksperymentach przeszło **{relevant[1]} testów odpowiednich dla modelu/protokołu/treningu/reguł**
oraz **{count} testów całego repozytorium**. Pełny przebieg zgłosił **{warning_count} ostrzeżeń**;
ich dokładne przyczyny i lokalizacje są w `full_pytest.log`. Ostrzeżeń nie tłumiono.
Testy na kontrolowanych fixture sprawdzają mechanikę i odtwarzalność, nie skuteczność kliniczną.
Rzeczywiste dane służą porównaniu pełnego foldu 9; nie wyznaczano testowych metryk.
"""
    report=f"""# Phase 18C — kontrolowana optymalizacja sposobu ważenia straty

Zakończono dwa nowe treningi CPU **B i C**. **A nie trenowano ponownie**.
Baseline A, architektura, preprocessing, etykiety i eksporty ECG pozostały bez zmian.
**Fold 10 nie został odczytany ani oceniony przez eksperymenty lub porównanie modeli.**

## Projekt i kontrola porównania

Wariant A to istniejący model Phase 18, `results/model_training/phase18_full/run_001/best_model.pt`.
B używa nieważonej CrossEntropyLoss. C używa pierwiastka oryginalnych, treningowych
wag zbalansowanych, z normalizacją średniej do 1. Skalowanie wszystkich wag wspólnym
czynnikiem nie zmienia matematycznego ilorazu ważonej straty; obliczenia są float32.
Wszystkie wagi i liczebności wyznaczano wyłącznie z foldów 1–8.

Wagi użyte podczas treningów, w kolejności SR, AFIB, STACH, SBRAD:

{weight_lines}

Wspólna architektura: istniejący 1D-CNN, **74116 parametrów**, wejście `[B,12,1000]`,
cztery logity. Adam, lr **0.001**, batch **16**, CPU **2 wątki**, seed **42**,
maksymalnie **15 epok**, patience **4**, min_delta **0**, wybór checkpointu wyłącznie
przez najwyższe Macro F1 całego foldu 9. Nie stosowano dodatkowych schedulerów,
augmentacji, resamplingu ani zmiany preprocessingów. Softmax nie poprzedza CE.

Trening: **15689 ECG / 13749 pacjentów**, foldy 1–8, klasy [13354,1189,643,503].
Walidacja: **1958 ECG / 1763 pacjentów**, fold 9, klasy [1667,146,82,63].
Populacje, identyfikatory i oficjalne foldy są identyczne we wszystkich wariantach.
Sprawdzono rozłączność pacjentów, zgodność inicjalizacji, pierwszy tasowany batch
oraz całą kolejność identyfikatorów dla wspólnie wykonanych epok.
Początkowy stan kontrolny A odtworzono z jego konfiguracji seed/niezmienionego kodu;
Phase 18 nie przechowywała osobnego checkpointu epoki 0. Hashe początkowych stanów
B i C są identyczne z tym odtworzonym stanem; A nie trenowano ponownie.
Każda pełna epoka ma 981 batchy treningowych i 123 walidacyjne, bez pomijania końcówek.

Nowa nakładka `src/ptbxl_optimization.py` deleguje trening do istniejącej implementacji.
Sposób ważenia i hash nakładki zapisuje w planie checkpointu. Zapobiega to wznowieniu
z innym rodzajem straty. Istniejącego pliku trenera, architektury i loaderów nie zmieniano.

Przed startem zapisano `study_plan.json`, konfiguracje i kryterium wyboru. Sprawdzono
SHA256 i metadane wszystkich 17647 NPZ treningowych/walidacyjnych. Dla obu nowych
strat wykonano osobny smoke test na 1 rzeczywistym batchu treningowym i 4 zapisach
walidacyjnych; wagi C pochodziły już z pełnych foldów treningowych. Pełne treningi
zaczynały się od nowej, tej samej inicjalizacji seed 42, nie od smoke checkpointów.

## Czasy i zasoby

| Wariant | Epoki | Najlepsza epoka | Czas [s] | Czas [min] | Zakończenie |
|---|---:|---:|---:|---:|---|
{duration_table}

Czas A pochodzi z oryginalnego Phase 18; nie był nowym kosztem tego etapu.
Czasy B/C obejmują wywołanie trenera i jego zapis wyników, nie wstępne audyty,
smoke, późniejszą wspólną inferencję i testy. Różnice czasu i liczby epok wynikają
m.in. z early stopping i obciążenia CPU. Dwa treningi wykonywano kolejno.

Zasoby przed startem i jawne limity znajdują się w `study_plan.json`.
Limity: 2 GiB RSS procesu, rezerwa 512 MiB dostępnej RAM i 1 GiB dysku,
czas epoki <=2×najwolniejsza zmierzona epoka Phase 18 (około 1210,54 s).
Pomiary B/C: `<wariant>_audit/resource_usage.json`. Każdy checkpoint zapisuje się
atomowo po ukończonej epoce, wraz z Adam/RNG/historią. Wznowienie wymaga nowego
katalogu, identycznego planu i kodu; niepełna epoka jest odtwarzana.

| Wariant | Szczyt RSS [MiB] | Min. dostępna RAM [GiB] | Min. wolny dysk [GiB] | Naruszenie limitów |
|---|---:|---:|---:|---|
{resource_table}

## Porównanie trzech zamrożonych modeli — pełne coverage, argmax

Wszystkie trzy checkpointy ponownie oceniono na **tych samych 1958 ECG** po zakończeniu
treningów, z tą samą kolejnością, batch size 16, CPU/2 wątki, `eval()` i `inference_mode()`.
Świeże predykcje odtworzyły predykcje wybranych epok. Wagi i bufory BatchNorm nie zmieniły się.

| Model | Accuracy | Macro F1 | Macro AUPRC | AFIB recall | SBRAD precision | SBRAD recall | SBRAD FP |
|---|---:|---:|---:|---:|---:|---:|---:|
{raw_table}

Najwyższe **argmax Macro F1**: **{best_f1['model']} ({float(best_f1['macro_f1_full_cohort']):.6f})**.
Najmniej FP SBRAD dla argmax: **{fewest_raw_names} ({fewest_raw['SBRAD_fp']})**.
Najmniej FP wśród wszystkich sześciu reguł: **{fewest_all_names} ({fewest_all['SBRAD_fp']})**.
Sama najmniejsza liczba FP nie jest wystarczającą podstawą wyboru: należy zachować
recall AFIB/SBRAD i coverage. Wszystkie zmiany poniżej są pomiarami tej samej walidacji:

{change_text}

To porównanie kontroluje architekturę, seed, dane, kolejność i protokół, ale obejmuje
tylko jedno ziarno. Nie oznacza dowodu statystycznej lub klinicznej przewagi.

## Predefiniowany próg SBRAD 0,65 i UNCERTAIN

Nie dopasowywano nowej siatki progów. Dla każdego modelu oceniono wyłącznie argmax
i wcześniejszy próg **0,65**. Jeśli zwycięża SBRAD, wymagane jest p(SBRAD)>=0,65;
w przeciwnym razie wynik to **UNCERTAIN=-1**, bez przypisywania drugiej klasy.
Inni zwycięzcy argmax pozostają bez zmian. To nie kalibracja prawdopodobieństw.
Softmax nie jest uznawany za skalibrowane prawdopodobieństwo kliniczne, a ważenie
straty może zmieniać jego rozkład pomiędzy modelami. Żadna reguła nie została medycznie zwalidowana.

Recall SBRAD zawsze używa mianownika **63**, z odrzuconymi prawdziwymi SBRAD jako FN.
Pełnozbiorowe Macro F1 również liczy abstencje jako pominięcia prawdziwej klasy.
Precision dotyczy zaakceptowanych alarmów. Dwie ostatnie kolumny poniżej są
**coverage-conditional**, obliczone wyłącznie na zaakceptowanych przypadkach.

| Kandydat | SBRAD FP | Precision | Recall /63 | Coverage | UNCERTAIN | Macro F1 cały zbiór | Acc. warunkowa | Macro F1 warunkowe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{decision_table}

Per-class abstencje (według prawdziwej klasy), pełne coverage oraz dwa rodzaje
Macro F1 znajdują się w `PHASE18C_MODEL_COMPARISON.csv` i `PHASE18C_CLASS_METRICS.csv`.
Niepewność nie oznacza poprawnego SR. Zmniejszenie FP przez odrzucenia nie zamienia
tych błędów w poprawne rozpoznania i może zwiększyć koszt późniejszej obsługi wyników.
FP SBRAD obejmują także pomyłki innych arytmii, np. AFIB→SBRAD. Nie oznaczają
automatycznie fałszywych alarmów obecności dowolnej arytmii. Osobna kolumna
`SR_to_any_target_arrhythmia_fp` liczy SR→AFIB/STACH/SBRAD, bez traktowania
UNCERTAIN jako poprawnej diagnozy SR. SR jest etykietą rytmu, nie dowodem zdrowego ECG.
Przy braku alarmów precision jest nieokreślone (puste CSV/JSON null), nie 100%.

## Wybór kandydata według z góry zapisanej reguły

1. AFIB recall >=0,90, STACH recall >=0,75, SBRAD recall >=0,70, coverage >=0,97;
   Macro AUPRC całego zbioru nie niższe od A o więcej niż 0,02.
2. Zachowaj wyniki w paśmie 0,005 od najlepszego Macro F1 całego zbioru.
3. Następnie pasmo 0,005 od najlepszego Macro AUPRC w pozostałej grupie.
4. Następnie liczba FP SBRAD nie większa niż minimum tej grupy +5.
5. Wśród praktycznie zbliżonych wyników preferuj argmax; potem brak wag, pierwiastek
   wag, pełne wagi. Kryterium nie wykorzystuje warunkowego Macro F1 do rankingu.

Tolerancje są inżynierskie, ustalone przed treningiem, nie są testami istotności.
Chronią AFIB, minimalne wykrywanie mniejszości i niski odsetek niepewności, zamiast
optymalizować samą precision SBRAD. Dokładny przebieg selekcji: `candidate_selection.json`.

{recommendation}

## Pełne metryki każdej klasy i reguły

Wszystkie liczby TP/FP/FN/TN odnoszą się do całego foldu 9, one-vs-rest.
Recall=TP/(TP+FN), specificity=TN/(TN+FP), precision=TP/(TP+FP),
F1=2TP/(2TP+FP+FN). ABSTAIN przy prawdziwej klasie liczy się jako jej FN;
brak alarmu innej klasy może liczyć się jako TN tej klasy, ale nie oznacza
poprawnej pełnej diagnozy. Macro AUPRC to średnie nieważone one-vs-rest average
precision, bez interpolacji, z grupowaniem remisów; liczone na pełnym zbiorze,
nie na podzbiorze zaakceptowanych predykcji.

| Kandydat | Klasa | TP | FP | FN | TN | Precision | Recall | Specificity | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{full_classes}

Pomiar `inference_forward_ms_per_record` w CSV to amortyzowany czas samego
forward na CPU w batchach 16, bez NPZ i metryk. `validation_inference_wall_seconds`
obejmuje pełne wywołanie inferencji z loaderem i kontrolą niezmienności modelu.
Nie jest to opóźnienie pojedynczego żądania aplikacji. Pomiarów nie korygowano
pod kątem chwilowego obciążenia komputera.

## Wykresy i historia

![Macierze trzech modeli i dwóch reguł](COMPARATIVE_CONFUSION_MATRICES.png)

![Straty treningowe i walidacyjne](COMPARATIVE_LOSS_CURVES.png)

Wagi funkcji straty są inne, więc bezpośrednie porównywanie wartości CE między
wariantami nie wyznacza najlepszego modelu. Train loss jest online, a validation
loss po epoce w eval; dropout i BatchNorm pracują w innych trybach.

![Macro F1 i AUPRC na pełnej walidacji](VALIDATION_MACRO_F1_AUPRC_CURVES.png)

![Dwa punkty precision–recall SBRAD na model](SBRAD_PRECISION_RECALL_COMPARISON.png)

Wykres przedstawia sześć ustalonych punktów decyzyjnych, nie pełne krzywe PR.
Nie obliczano z niego AUPRC metodą trapezów.

![Coverage i liczba fałszywych SBRAD](COVERAGE_FALSE_POSITIVE_TRADEOFF.png)

{chr(10).join(epoch_tables)}

## Integralność i testy

Baseline A oraz Phase 18B zachowano; SHA256 zapisano przed treningami.
Audyt porównawczy potwierdził: `{summary['preservation']}`.
Każdy odczyt użytych NPZ weryfikował SHA256 i metadane istniejącym loaderem.
Wspólna ocena czytała wyłącznie sygnały foldu 9 (3×1958 rekordów); treningi wyłącznie
foldy 1–8 i 9. Guardy blokują fold 10 przed I/O. Sprawdzenie globalnych metadanych
pacjentów/eksportu nie oznacza odczytu testowych sygnałów ani użycia ich do selekcji.
Nie zmieniono eksportów MIT-BIH lub PTB-XL; kontrola rozmiarów/mtime nie jest
nowym pełnym haszowaniem wszystkich MIT-BIH. Nie utworzono nowych podziałów pacjentów.

{test_section}

## Odtwarzanie, pliki i ograniczenia

W tym katalogu znajdują się konfiguracje, plany, SHA baseline, dwa nowe katalogi
treningowe, ich `best_model.pt`, `last_resumable_checkpoint.pt`, pełna historia,
metryki epok, pomiary zasobów i audyty. `comparison/` zawiera predykcje i macierze
obu reguł każdego modelu; nie kopiuje sygnałów ECG. Pliki główne:

- `PHASE18C_MODEL_OPTIMIZATION_REPORT.md`, `PHASE18C_FINAL_RECOMMENDATION.md`;
- `PHASE18C_MODEL_COMPARISON.csv`, `PHASE18C_CLASS_METRICS.csv`;
- `study_plan.json`, `*_configuration.json`, `*_plan.json`;
- `candidate_selection.json`, `frozen_candidate.json` (jeśli wybrano kandydata);
- `evaluation_summary.json`, `relevant_tests.log`, `full_pytest.log`.

Reprodukcja wymaga **nowego katalogu wersji**, by nie nadpisać wyników:
`python -m examples.optimize_ptbxl_losses prepare --root NOWY_KATALOG`, potem
`smoke --variant B_unweighted` i `smoke --variant C_sqrt_balanced` z tą samą opcją
`--root`, a następnie `python -m examples.train_ptbxl_optimization_pair --root NOWY_KATALOG`.
Wznowienie przerwanego wariantu: `examples.optimize_ptbxl_losses train --variant ...
--root ... --destination NOWA_NAZWA --resume-from ... --prior-audit ...`.
Nie należy wznawiać zakończonego early stopping eksperymentu z pominięciem tego warunku.
Po wznowieniu należy wskazać właściwe ścieżki `--b-run`/`--c-run` porównywarce.
Nie uruchamiano dodatkowych eksperymentów poza zatwierdzonymi B/C i technicznymi smoke.

Powtarzany wybór epoki, progu i modelu na tym samym foldzie 9 może przeuczać decyzje
projektowe do tej walidacji. Wyniki nie stanowią niezależnej estymacji końcowej
skuteczności. Mniejszościowe klasy mają mało pacjentów, a tylko jedno ziarno nie
mierzy zmienności treningu. Żadnej reguły nie uznano za medycznie zwalidowaną.
Po zatwierdzeniu należy zamrozić konkretny checkpoint i regułę, a dopiero w osobno
zatwierdzonym etapie wykonać końcową ocenę na dotąd nieużytym foldzie 10.
**Nie rozpoczęto Phase 19, Streamlit ani treningu MIT-BIH.**
"""
    final=f"""# Phase 18C — końcowa rekomendacja walidacyjna

{recommendation}

| Model, argmax | Accuracy | Macro F1 | Macro AUPRC | AFIB recall | SBRAD precision | SBRAD recall | SBRAD FP |
|---|---:|---:|---:|---:|---:|---:|---:|
{raw_table}

{change_text}

Najwyższe argmax Macro F1: **{best_f1['model']}**. Najmniej argmax FP SBRAD:
**{fewest_raw_names}**; wśród sześciu reguł: **{fewest_all_names}**.
Nie należy wybierać modelu na podstawie samych FP z pominięciem recall AFIB i mniejszości.

{test_section}

Rekomendacja wynika z wcześniej zapisanych ograniczeń i tolerancji w `study_plan.json`.
Ma charakter inżynierski; nie stanowi walidacji klinicznej ani dowodu statystycznej
przewagi. Progi Softmax nie są skalibrowanymi prawdopodobieństwami choroby.
Metryki po abstencjach są warunkowe i nie zastępują metryk pełnego zbioru.
Powtarzana selekcja na foldzie 9 może przeuczać decyzje do walidacji.

Baseline, dane i preprocessing zachowano. **Nie czytano ani nie oceniano foldu 10.**
Po akceptacji tej rekomendacji należy zamrozić wskazany checkpoint i regułę
przed odrębną końcową oceną. Nie uruchomiono kolejnego treningu ani Phase 19.
Pełne wyniki i ograniczenia: `PHASE18C_MODEL_OPTIMIZATION_REPORT.md`.
"""
    for name,text in [('PHASE18C_MODEL_OPTIMIZATION_REPORT.md',report),('PHASE18C_FINAL_RECOMMENDATION.md',final)]:
        with (root/name).open('x') as target:target.write(text)
    print(json.dumps({'reports_created':True,'full_tests_passed':count,'full_test_warnings':warning_count,'selected':None if chosen is None else chosen['candidate']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/model_training/phase18c_optimization/run_001'))
    args=parser.parse_args();reports(args.root)
