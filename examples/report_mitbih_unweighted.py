"""Polish Phase-23A report from completed frozen validation results only."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
from src.mitbih_unweighted import BASELINE
from src.mitbih_full_run import validated_history
from src.prepare_dataset import sha256_file


def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def main(output,audit):
    output,audit=Path(output),Path(audit);comp=output/'comparison'
    metrics={name:json.loads((comp/(name+'_metrics.json')).read_text()) for name in ('A_sqrt_balanced','B_unweighted')}
    a,b=metrics.values();rows=read_csv(comp/'MODEL_COMPARISON.csv')
    result,plan,history,best=validated_history(output)
    execution=json.loads((output/'execution.json').read_text())
    checked=json.loads((comp/'integrity.json').read_text())
    log=(output/'post_training_pytest.log').read_text()
    match=re.search(r'(\d+) passed, (\d+) warnings in ([\d.]+)s',log)
    if not match:raise ValueError('Full successful post-training tests required')
    pct=lambda v:'nieokreślone' if v is None else f'{100*v:.2f}%'
    frozen=json.loads((comp/'FROZEN_TRAIN_LOSS_AUDIT.json').read_text())
    frozen_table='\n'.join(f"| {name} | {frozen[name]['unweighted_ce']['loss']:.6f} | {m['unweighted_validation_ce']['loss']:.6f} | {frozen[name]['sqrt_weighted_ce']['loss']:.6f} | {m['sqrt_weighted_validation_ce']['loss']:.6f} |" for name,m in metrics.items())
    comparison='\n'.join(f"| {name} | {pct(m['accuracy'])} | {m['macro_f1']:.6f} | {m['macro_auprc']:.6f} | {m['classes_with_nonzero_recall']} | {m['unweighted_validation_ce']['loss']:.6f} | {m['sqrt_weighted_validation_ce']['loss']:.6f} | {m['inference_ms_per_beat']:.4f} |" for name,m in metrics.items())
    class_table='\n'.join(f"| {name} | {r['class_name']} | {r['support']} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['specificity'])} | {r['f1']:.6f} | {r['fp']} | {r['fn']} |" for name,m in metrics.items() for r in m['per_class'])
    epoch_table='\n'.join(f"| {r['epoch']} | {r['train']['loss']:.6f} | {r['validation']['loss']:.6f} | {pct(r['validation']['accuracy'])} | {r['validation']['macro_f1']:.6f} | {r['validation']['macro_auprc']:.6f} | {r['seconds']:.2f} |" for r in history)
    symbols=read_csv(comp/'ORIGINAL_SYMBOL_AAMI_AUDIT.csv')
    symbol_table='\n'.join(f"| {r['partition']} | `{r['original_symbol']}` | {r['mapped_class']} | {r['count']} |" for r in symbols)
    leads=read_csv(audit/'record_lead_class_distribution.csv')
    leadcounts=Counter((r['partition'],r['channel_0'],r['channel_1']) for r in leads)
    leadtable='\n'.join(f'| {p} | {c0} | {c1} | {n} |' for (p,c0,c1),n in sorted(leadcounts.items()))
    fprows=[]
    for name in metrics:
        for rid in ('201','202'):
            m=json.loads((comp/f'{name}_record_{rid}_confusion.json').read_text())
            fprows.append(f"| {name} | {rid} | {m['beats']} | {m['F_false_positives']} |")
    groupmetrics=read_csv(comp/'PATIENT_GROUP_COMPARISON.csv')
    fp210={r['model']:int(r['fp']) for r in groupmetrics if r['class_name']=='F' and r['patient_group']=='mitdb_group_210'}
    supportrows=[r for r in groupmetrics if r['class_name'] in ('S','F','Q') and (int(r['support']) or int(r['fp']))]
    grouptable='\n'.join(f"| {r['model']} | {r['patient_group']} | {r['class_name']} | {r['support']} | {r['tp']} | {r['fp']} | {pct(float(r['recall'])) if r['recall'] else 'nieokreślone'} |" for r in supportrows)
    cmtext=[]
    for name,m in metrics.items():
        text='\n'.join('| '+'NSVFQ'[i]+' | '+' | '.join(map(str,row))+' |' for i,row in enumerate(m['confusion_matrix']))
        cmtext.append('### '+name+'\n\n| True / pred | N | S | V | F | Q |\n|---|---:|---:|---:|---:|---:|\n'+text)
    delta_f1=b['macro_f1']-a['macro_f1'];delta_ap=b['macro_auprc']-a['macro_auprc']
    better='B_unweighted' if delta_f1>0 else 'A_sqrt_balanced' if delta_f1<0 else 'remis'
    selection=(f'Według wcześniej używanego kryterium pełnego walidacyjnego Macro F1 lepszy jest **{better}**. '
        f'Zmiana B−A: Macro F1 {delta_f1:+.6f}, Macro AP {delta_ap:+.6f}, Accuracy {100*(b["accuracy"]-a["accuracy"]):+.2f} punktu procentowego. '
        'Nie oznacza to automatycznie poprawy każdej klasy; poniżej podano wszystkie trade-offy. '
        'To jeden seed i jeden podział, bez dowodu przewagi statystycznej ani klinicznej.')
    tradeoffs=[]
    for i,c in enumerate('NSVFQ'):
        ar,br=a['per_class'][i],b['per_class'][i]
        tradeoffs.append(f"{c}: recall {pct(ar['recall'])} → {pct(br['recall'])}, precision {pct(ar['precision'])} → {pct(br['precision'])}, FP {ar['fp']} → {br['fp']}, FN {ar['fn']} → {br['fn']}")
    fp_symbols=read_csv(comp/'F_FALSE_POSITIVE_SYMBOL_AUDIT.csv') if (comp/'F_FALSE_POSITIVE_SYMBOL_AUDIT.csv').exists() else []
    fptext='; '.join(f"{r['model']} / {r['record_id']} / symbol {r['original_symbol']}: {r['F_false_positives']}" for r in fp_symbols if r['record_id'] in ('201','202'))
    text=f'''# Faza 23A — kontrolowany trening unweighted i audyt błędów walidacyjnych MIT-BIH

**Status: ukończono dokładnie jeden nowy eksperyment.** Model B trenowano od zera. Model A fazy 22 nie był ponownie trenowany ani nadpisany. Oba zamrożone checkpointy oceniono na tej samej pełnej walidacji. Końcowy test MIT-BIH nie został otwarty do oceny ani audytu; nie wykonano nowej ewaluacji PTB-XL.

## Identyfikacja i kontrola eksperymentu

A: `{BASELINE/'best_model.pt'}`, SHA256 `{a['checkpoint_sha256']}`. B: `{output/'best_model.pt'}`, SHA256 `{b['checkpoint_sha256']}`.

Jedyna zamierzona zmiana: `sqrt_balanced` → **unweighted CrossEntropyLoss**, `weight=None`. Wszystkie pozostałe pola konfiguracji, kod trainera/modelu/loadera/metryk, wersje torch/numpy, manifest, konfiguracja modelu oraz zamrożony split porównano programowo z planem fazy 22. Wagi fazy 22 nie służyły do inicjalizacji B: istniejący trainer tworzy nowy model po seed_everything(42), bez resume_from. Oba runy mają ten sam shuffle seed+epoch, Adam lr 0.001, batch 32, 2 wątki CPU, max 15 epok, patience 4 i best checkpoint według pełnego walidacyjnego Macro F1. Różne liczby epok/best epoch są dozwolonym skutkiem tej samej reguły early stopping, nie ręczną zmianą budżetu.

Model: niezmieniony CNN 17797 parametrów, X float32 `[B,2,252]`, y long, pięć logitów N/S/V/F/Q. Softmax stosowano dopiero do AP/probability audit, nigdy przed CE. Bez resamplingu, augmentacji, zmiany etykiet, normalizacji, filtrów, odprowadzeń czy inicjalizacji z nauczonego checkpointu.

Dane: 62926 treningowych uderzeń / 29 grup (N52254,S1715,V4389,F401,Q4167), 23134 walidacyjne / 9 grup (N19094,S491,V1444,F38,Q2067). 201 i 202 należą do jednej grupy walidacyjnej. Podział `configs/mitbih_patient_split_v1.json` SHA256 `8c5cc28d8dcae170fd222432d9f247fd19207ba01be60763241da4b7cae61831`; grupy są rozłączne. Liczby końcowego testu pozostają wyłącznie metadanymi zamrożonego manifestu; nie analizowano jego symboli, sygnałów ani błędów.

## Przebieg nowego runu

Katalog `{output}`. Ukończono **{result['completed_epochs']} epok**, best epoch **{result['best_epoch']}**, status **{result['status']}**, czas całego wywołania train_model **{execution['wall_seconds']:.2f} s ({execution['wall_seconds']/60:.2f} min)**. Czas A: {float(rows[0]['training_seconds']):.2f} s. Pomimo tych samych dwóch wątków obciążenie systemu i warunki termiczne nie były kontrolowane; różnicy czasu nie wolno przypisywać samej funkcji straty.

| Epoka B | Online train CE | Validation CE | Val accuracy | Val Macro F1 | Val Macro AP | Czas s |
|---|---:|---:|---:|---:|---:|---:|
{epoch_table}

## Uczciwe porównanie zamrożonych modeli

Każdy wynik pochodzi z pełnych **23134 tych samych uderzeń walidacyjnych**, eval() i inference_mode(), argmax, bez progów/abstencji. Sprawdzono zgodność populacji, unikalność indeksów i odtworzenie metryk wybranej epoki. Dataset cache napełniono przed oboma przebiegami; czas forward nie obejmuje loading/CE/Softmax/metadanych. Czasy są opisowym pomiarem batchy 32 na bieżącym CPU, nie kontrolowanym benchmarkiem mikroarchitektury. Softmax nie jest deklaracją klinicznej kalibracji.

| Model | Accuracy | Macro F1 | Macro AP | Klasy z recall >0 | Wspólna val CE bez wag | Wspólna val CE sqrt | Forward ms/beat |
|---|---:|---:|---:|---:|---:|---:|---:|
{comparison}

{selection}

### Pełne metryki klas

Support/TP/FP/FN są beat-level, nie liczbami niezależnych pacjentów. Macro F1 i Macro AP obejmują wszystkie pięć klas, również z zerowym recall. AP to nieinterpolowana one-vs-rest average precision, nie pole trapezami. Nieokreślona precision przy braku predykcji to null, nie sztuczne zero. Klasy AAMI-style to grupy uderzeń, nie pięć nazwanych diagnoz rytmu.

| Model | Klasa | Support | Precision | Recall | Specificity | F1 | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{class_table}

Zmiany klasowe: {'; '.join(tradeoffs)}. Liczba klas z niezerowym recall jest tylko wskaźnikiem pokrycia — pojedynczy TP nie oznacza użytecznego rozpoznawania klasy. Wyższa Accuracy może wynikać z częstszego N przy spadku recall mniejszości; punkt odniesienia always-N na walidacji to {100*19094/23134:.2f}%.

{chr(10).join(cmtext)}

## Audyt strat — brak mieszania różnych funkcji celu

W obu trybach train/validation istniejący trainer liczy dokładną stratę jako `sum(w_y * CE_i) / sum(w_y)`. Dla B wagi są 1, więc mianownik to liczba uderzeń. Sumowane są liczniki i mianowniki wszystkich batchy, nie nieważona średnia średnich. Te same **treningowe** wagi fazy 22 obowiązywały w jej train i validation. Nie dopasowywano wag do walidacji.

Wspólne CE obu modeli przeliczono z tych samych zamrożonych walidacyjnych logitów: raz bez wag, raz z oryginalnymi treningowymi wagami sqrt. Własna CE każdego modelu odtwarza zapis wybranej epoki z tolerancją numeryczną; nie wykryto błędu agregacji, pominiętego batcha ani innego zbioru. Ręczny test porównuje agregację z bezpośrednią PyTorch CrossEntropyLoss, również dla ostatniego niepełnego batcha.

Dodatkowo wykonano jedynie inferencję loss-only obu zamrożonych modeli na pełnych 62926 treningowych uderzeniach w eval/inference_mode. Nie było backward, optimizer step ani nowej selekcji modelu. Pozwala to porównywać tryb i funkcję celu identycznie z walidacją:

| Model | Frozen train CE bez wag | Frozen val CE bez wag | Frozen train CE sqrt | Frozen val CE sqrt |
|---|---:|---:|---:|---:|
{frozen_table}

Historia online i frozen train dotyczą różnych chwil/trybów; utrzymywanie się luki frozen train–validation wskazuje na brak generalizacji po wyrównaniu trybu, ale nie identyfikuje przyczyny. Sama obecność BatchNorm nie dowodzi jego błędu.

Historyczna train CE A i B ma inne znaczenie i skalę — nie można z samej różnicy wnioskować o poprawie B. Ponadto training loss/metrics są online, przed kolejnymi aktualizacjami z aktywnymi dropout/BatchNorm, a validation jest po epoce w eval/inference_mode. Sama luka strat nie dowodzi błędu obliczeń ani jednej przyczyny słabej generalizacji. Nie wykonano dodatkowego pełnego treningu ani oceny testu, by „naprawić” lukę.

## Symbole źródłowe i przesunięcie składu klas

Mapowanie pozostaje bez zmian: N←N/L/R/e/j, S←A/a/J/S, V←V/E, F←F, Q←`/`, `f`, `Q`. Duże F to inna etykieta niż małe f; markerów rytmu nie potraktowano jako uderzeń. Oryginalne symbole są zachowane w każdej predykcji CSV i audycie.

| Część | Symbol MIT-BIH | Grupa | Liczba |
|---|---|---|---:|
{symbol_table}

Potwierdzone różnice:

- S: 1382 z 1715 treningowych S (80.58%) pochodzi z nagrania/grupy 232; liczba przykładów nie zastępuje różnorodności osób. `a` 6/1715 w treningu (0.35%) versus 139/491 w walidacji (28.31%); `A` 1676 versus 350. Ocena zbiorczej S miesza odmienne symbole i pacjentów.
- Q: `/` 4105, `f` 56, `Q` 6 w treningu; `/` 1378, `f` 666, `Q` 23 w walidacji. Q jest heterogeniczna; udział `f` rośnie z 1.34% do 32.22%. Nie relabelowano ani nie usuwano tych przykładów. Błędy według symbolu są w ORIGINAL_SYMBOL_ERRORS.csv.
- N: L/R występują w treningu (6615/5345), nie w tej walidacji; j/e mają inny udział. To opis składu konkretnego zamrożonego podziału, nie definicja całej bazy.
- F: trening 401 przykładów z 10 grup, ale 372 z 208; walidacja 38 z czterech grup. Q w treningu jest zdominowana przez 102/107, w walidacji przez 104.

Różnice te są zgodne z ryzykiem domain shift i zależności od pacjenta, ale nie identyfikują przyczyny konkretnych błędów bez dalszego audytu. Nie zmieniamy podziału po obejrzeniu wyników i nie analizujemy końcowego testu, by dostrajać dane.

## Odprowadzenia i grupa 201/202

| Część | Kanał 0 | Kanał 1 | Liczba nagrań |
|---|---|---|---:|
{leadtable}

Indeks kanału nie gwarantuje identycznego anatomicznego odprowadzenia; zachowano nazwy i kolejność eksportu, bez harmonizacji. 201/202 mają MLII/V1, tak jak 22 treningowe nagrania. 104 ma V5/V2, reprezentowane też przez treningowe 102. Z samych nazw odprowadzeń nie wynika wyjaśnienie błędów; nie przypisujemy im przyczynowości.

| Model | Rekord | Uderzenia | F false positives |
|---|---|---:|---:|
{chr(10).join(fprows)}

FP F według symbolu w tych dwóch rekordach: {fptext or 'brak'}. W grupie 210 FP F zmieniły się z {fp210['A_sqrt_balanced']} do {fp210['B_unweighted']}; część problemu jest więc silniejsza w innej grupie mimo niższej sumy FP. Pełny audyt obejmuje wszystkie walidacyjne rekordy, nie tylko wybrane przykłady. Symbole i pozycje adnotacji pozwalają później obejrzeć te same rzeczywiste okna bez ich przesuwania czy zmiany etykiet.

### Metryki S/F/Q według grup

| Model | Grupa | Klasa | Support | TP | FP | Recall |
|---|---|---|---:|---:|---:|---:|
{grouptable}

**38 walidacyjnych uderzeń F, skupionych w niewielkiej liczbie pacjentów, nie wystarcza do wiarygodnych samodzielnych wniosków o tej klasie.** Nie traktować skorelowanych uderzeń jako niezależnych pacjentów, nie interpretować skoków jednego TP jako trwałej poprawy. Zachowano obie macierze, także zera, bez ukrywania nieskutecznych klas.

## Integralność, testy i artefakty

Preflight zweryfikował SHA256 i zgodność sygnałów/etykiet wszystkich 39 treningowych/walidacyjnych shardów. Testowe NPZ pozostały zamknięte; inwentarz obu eksportów sprawdzano wyłącznie stat dla ochrony niezmienności. Po pracy: **{checked['prior_files_sha256_verified']} wcześniejszych plików SHA256 bez zmian**, **{checked['dataset_files_stat_verified']} plików eksportów bez zmiany rozmiaru/mtime/inwentarza**. Zachowano wszystkie artefakty PTB-XL i MIT-BIH fazy 22. Oba checkpointy i ich parametry/bufory pozostały identyczne po inferencji. Dodatkowy reload B na małym batchu treningowym daje identyczne logity; nie jest to ocena testowa.

Testy po treningu: **{match.group(1)} passed, {match.group(2)} warnings, {match.group(3)} s**. Uruchomiono nowe testy loss-only config, dokładnej ważonej/nieważonej CE, odmowy dostępu do testu, kompletu walidacji, metadanych i niezmienności modelu, a także cały istniejący zestaw. Deprecacje pandas/WFDB pozostawiono widoczne. Historyczne QC i syntetyczne testy nie oznaczają oceny nowego modelu na końcowym teście.

Pliki:

- `{output/'best_model.pt'}`, `{result['last_checkpoint']}` i ich SHA256; configuration.json, training_plan.json, training.log, epoch metrics, execution.json. Ostatni checkpoint zawiera Adam/RNG i historię; nie przedłużać automatycznie zakończonego runu.
- `analysis/`: historia CSV/JSON, własne krzywe strat/F1/AP B, macierze, metryki klas i grup, kontrola reload.
- `comparison/`: MODEL_COMPARISON.csv, CLASS_COMPARISON.csv, PATIENT_GROUP_COMPARISON.csv, ORIGINAL_SYMBOL_AAMI_AUDIT.csv, ORIGINAL_SYMBOL_ERRORS.csv, F_FALSE_POSITIVE_SYMBOL_AUDIT.csv, osobne predykcje CSV i NPZ obu modeli, macierze rekordów 201/202, integrity.json, porównawcze wykresy confusion_matrices.png i training_comparison.png.
- Preflight: `{audit}`, w tym per-record odprowadzenia/liczebności i oryginalne symbole. Suma kontrolna wszystkich nowych plików jest zapisana osobno po zakończeniu.

## Wniosek i następny krok — bez uruchamiania

{selection}

Następnie warto przejrzeć błędy **według oryginalnego symbolu, grupy i rzeczywistych okien** (zwłaszcza S typu a, Q typu f oraz FP F w 201/202 i 210), a także wyjaśnić pozostałą lukę między zmierzoną frozen train CE i validation CE, uwzględniając koncentrację pacjentów i statystyki BatchNorm. Audyt loss-only już wyrównał tryb eval; obecny eksperyment nadal nie rozstrzyga jednej przyczyny generalizacji. Dopiero po takim audycie i osobnym zatwierdzeniu planować jedną zmianę modelu lub strategii; nie zakładać, że silniejsze wagi rozwiążą koncentrację pacjentów.

Nie uruchomiono dodatkowego wariantu ani nowego podziału. Powtarzana selekcja na tej samej walidacji może ją przeuczyć; ograniczyć liczbę porównań i ustalić kryteria z góry. Nie oceniać końcowego testu przed ostatecznym zamrożeniem rozwiązania i osobną zgodą. **Koniec fazy 23A.**
'''
    with Path('PHASE23A_MITBIH_UNWEIGHTED_REPORT.md').open('x') as f:f.write(text)
    with (output/'PHASE23A_MITBIH_UNWEIGHTED_REPORT.md').open('x') as f:f.write(text)
    print(selection)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--audit-dir',type=Path,required=True)
    args=parser.parse_args();main(args.output_dir,args.audit_dir)
