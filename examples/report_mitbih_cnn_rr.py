"""Seal Phase24B read-only integrity and Polish handoff after frozen comparison."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.mitbih_execution_guard import HoldoutAccessGuard
ROOT=Path('results/model_training/phase24b_mitbih_cnn_rr');OUT=ROOT/'run_001'

def read(p):
    import json
    return json.loads(Path(p).read_text())

def table(rows,columns):
    def cell(x):return 'nieokreślone' if x is None else (f'{x:.6f}' if isinstance(x,float) else str(x))
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(cell(r[c]) for c in columns)+' |' for r in rows)

if __name__=='__main__':
    with HoldoutAccessGuard(ROOT/'runtime_guard_final_integrity_001.json') as guard:
        guard.probe()
        import json,re
        import numpy as np
        from src.prepare_dataset import sha256_file
        from src.ptbxl_training import write_json_new
        from src.mitbih_full_run import validated_history
        result,plan,history,best=validated_history(OUT)
        prior=read(ROOT/'preflight_001/preservation_before.json')
        changed=[p for p,h in prior.items() if not Path(p).is_relative_to(ROOT) and (not Path(p).is_file() or sha256_file(p)!=h)]
        stats=read(ROOT/'preflight_001/dataset_stats_before.json')
        changed_data=[p for p,s in stats.items() if not Path(p).is_file() or [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=s]
        rr=read(ROOT/'preflight_001/rr_preservation_before.json')
        changed_rr=[p for p,h in rr.items() if not Path(p).is_file() or sha256_file(p)!=h]
        if changed or changed_data or changed_rr:raise ValueError(f'Protected artifact changes: {changed}, {changed_data}, {changed_rr}')
        rr_manifest=read('data/processed/mitbih_rr_context_v1/manifest.json')
        raw_checks=rr_manifest['raw_source_sha256']
        if any(sha256_file(p)!=digest for p,digest in raw_checks.items()):
            raise ValueError('Original TRAIN/VALIDATION annotation/header changed')
        if any(sha256_file(r['source_X_file'])!=r['source_X_sha256'] for r in rr_manifest['records']):
            raise ValueError('Original TRAIN/VALIDATION ECG shard changed')
        guards=[read(ROOT/n) for n in ('runtime_guard_training_001.json','runtime_guard_comparison_001.json')]
        if any(g['unexpected_denial_count'] or g['exit_exception'] is not None for g in guards):raise ValueError('Runtime isolation failed')
        comparison=read(OUT/'comparison/integrity.json')
        if comparison['unexpected_guard_denials'] or not comparison['checkpoint_reload_logits_exact']:raise ValueError('Comparison integrity failed')
        testlog=(ROOT/'post_training_tests_001.log').read_text()
        matches=re.findall(r'(\d+) passed(?:, (\d+) warnings)? in',testlog)
        if not matches or 'FAILED ' in testlog:raise ValueError('Tests have not passed')
        tests,warnings=matches[-1];tests=int(tests);warnings=int(warnings or 0)
        pytest_guard=read(ROOT/'post_training_test_guard_001.json')
        if pytest_guard['exit_status']!=0 or pytest_guard['actual_heldout_reads_through_guarded_open']:
            raise ValueError('Pytest hold-out isolation failed')
        if any(a['test']!='tests/test_mitbih_holdout_guard.py::test_real_heldout_open_is_blocked_before_reader (call)' for a in pytest_guard['blocked_file_attempts']):
            raise ValueError('Unexpected test attempted held-out access')
        names=['A_Phase22_CNN','CNN_RR','CNN_RR_neutral_diagnostic'];metrics={n:read(OUT/f'comparison/{n}_metrics.json') for n in names}
        groups={n:read(OUT/f'comparison/{n}_patient_metrics.json') for n in names}
        a=metrics[names[0]];b=metrics[names[1]];neutral=metrics[names[2]]
        execution=read(OUT/'execution.json');seconds=execution['wall_seconds']
        classrows=[dict(model=n,group=r['class_name'],support=r['support'],precision=r['precision'],recall=r['recall'],specificity=r['specificity'],F1=r['f1'],TP=r['tp'],FP=r['fp'],FN=r['fn']) for n in names for r in metrics[n]['per_class']]
        summary=[dict(model=n,accuracy=metrics[n]['accuracy'],Macro_F1=metrics[n]['macro_f1'],Macro_AUPRC=metrics[n]['macro_auprc'],CPU_forward_s=metrics[n]['forward_seconds'],ms_per_beat=metrics[n]['inference_ms_per_beat']) for n in names]
        patients=[]
        for n in names[:2]:
            for group,gm in groups[n].items():
                patients.append(dict(model=n,patient_group=group,beats=gm['beat_count'],errors=gm['beat_count']-int(np.trace(gm['confusion_matrix'])),S_recall=gm['per_class'][1]['recall'],F_support=gm['per_class'][3]['support'],F_TP=gm['per_class'][3]['tp'],F_FP=gm['per_class'][3]['fp'],Q_recall=gm['per_class'][4]['recall']))
        predictions=[dict(model=n,**dict(zip('NSVFQ',map(int,np.asarray(metrics[n]['confusion_matrix']).sum(0))))) for n in names]
        historyrows=[dict(epoch=r['epoch'],training_loss=r['train']['loss'],validation_loss=r['validation']['loss'],validation_Macro_F1=r['validation']['macro_f1'],validation_Macro_AUPRC=r['validation']['macro_auprc'],seconds=r['seconds']) for r in history]
        rrbenefit=b['macro_f1']>a['macro_f1']
        interpretation=('CNN+RR uzyskał wyższe pełnowalidacyjne Macro F1 niż zamrożony CNN. Uzasadnia to zachowanie RR jako kandydata badawczego, przy zachowaniu ograniczeń poniżej.' if rrbenefit else 'CNN+RR nie poprawił pełnowalidacyjnego Macro F1. Wynik nie uzasadnia zastąpienia zamrożonego CNN przez ten wariant; RR pozostaje hipotezą badawczą.')
        pct=lambda x:'nieokreślone' if x is None else f'{x:.4%}'
        minority_changes='\n\n'.join(f"**{c}:** recall {pct(a['per_class'][i]['recall'])} → {pct(b['per_class'][i]['recall'])}, precision {pct(a['per_class'][i]['precision'])} → {pct(b['per_class'][i]['precision'])}; FP {a['per_class'][i]['fp']} → {b['per_class'][i]['fp']}, FN {a['per_class'][i]['fn']} → {b['per_class'][i]['fn']}." for i,c in enumerate('NSVFQ') if c in 'SFQ')
        special_groups='\n\n'.join(f"**{g}:** liczba błędów CNN {groups[names[0]][g]['beat_count']-int(np.trace(groups[names[0]][g]['confusion_matrix']))} → CNN+RR {groups[names[1]][g]['beat_count']-int(np.trace(groups[names[1]][g]['confusion_matrix']))}; F FP {groups[names[0]][g]['per_class'][3]['fp']} → {groups[names[1]][g]['per_class'][3]['fp']}." for g in groups[names[0]] if g in ('mitdb_group_201_202','mitdb_group_210'))
        f_support_groups=sum(g['per_class'][3]['support']>0 for g in groups['CNN_RR'].values())
        report=f'''# Phase 24B — pełny trening MIT-BIH CNN+RR i kontrolowana walidacja

## Zakres i integralność

Uruchomiono dokładnie jeden nowy pełny eksperyment od losowej inicjalizacji. Nie użyto wag Phase 22 ani checkpointu smoke-testu. Zamrożony Phase 22 A stanowi podstawowy komparator; neutralne RR jest ablacją tego samego wytrenowanego CNN+RR, nie osobnym modelem morfologicznym.

Incydent Phase 24A: starszy test QC odczytał dziewięć shardów końcowego testu; nie użyto ich do trenowania, dopasowania skalera, inferencji ani wyboru modelu. Nie twierdzimy, że pliki nigdy nie były odczytane. W Phase 24B blokada audytowa Python działała przed preflightem, treningiem, porównaniem i końcową kontrolą. Sondy potwierdziły odmowę otwarcia 45 chronionych ścieżek przed odczytem. Nie było nieoczekiwanych prób otwarcia ani odczytów chronionych danych w ścieżce wykonania. Blokada jest mechanizmem audytowym Python, nie sandboxem systemu operacyjnego. Nie uruchamiano legacy pełnodatasetowego QC.

Niezmienione: {{len_prior}} historycznych/source'owych plików chronionych SHA256, {{len_data}} plików poprzednich eksportów (rozmiar i mtime, bez otwierania testowych shardów), {{len_rr}} plików adaptera RR (SHA256). Stare checkpointy, split i artefakty PTB-XL zachowano. Szczegóły w FINAL_INTEGRITY.json i logach runtime_guard_*.json.

## Dane i procedura

Trening: 62926 beatów, 29 grup, N/S/V/F/Q = 52254/1715/4389/401/4167. Pełna walidacja: 23134 beaty, 9 grup, 10 rekordów, 19094/491/1444/38/2067. Grupy są rozłączne; 201 i 202 pozostają razem w walidacji, lecz kontekst RR resetuje się dla każdego rekordu. Końcowy test pozostaje niedostępny dla tego eksperymentu; nie utworzono testowego sidecaru RR.

Architektura CNN+RR: 17989 parametrów (CNN:17797; +192 w głowicy). Wejścia float32 [B,2,252] i [B,6], wyjście pięć logitów. Cechy: poprzedni RR, mediana maksymalnie ośmiu wcześniejszych interwałów z wyłączeniem bieżącego poprzedniego RR, iloraz RR/mediana, trzy maski dostępności. Bez next-RR, etykiet i łączenia rekordów. Statystyki skalowania/imputacji niezależnie odtworzono wyłącznie z treningu; dane ECG nie zostały ponownie przefiltrowane ani znormalizowane.

Adam, lr=.001, batch32, seed42, CPU2 wątki, maks15 epok, patience4, brak min_delta. Wagi = sqrt(n_train/(5*n_class)), znormalizowane do średniej1: {plan['class_weights']}. Wagi tylko z treningu. CE dostaje logity bez Softmax. Agregacja strat to suma(w_y*CE)/suma(w_y), zgodna między treningiem a walidacją. Treningowe metryki/loss są online przed aktualizacjami, walidacja po epoce w eval/inference_mode; sama różnica tych strat nie jest dowodem błędu.

## Przebieg

Ukończone epoki: **{result['completed_epochs']}**, najlepsza: **{best['epoch']}**. Status: **{result['status']}**. Czas train_model (ładowanie, metryki i checkpointy, bez preflightu/porównania): **{seconds:.2f} s = {seconds/60:.2f} min**. Wybór checkpointu wyłącznie przez pełnowalidacyjne Macro F1; bez progów/abstencji, argmax wszystkich pięciu klas.

{table(historyrows,['epoch','training_loss','validation_loss','validation_Macro_F1','validation_Macro_AUPRC','seconds'])}

## Porównanie na identycznych 23134 źródłowych beatach

{table(summary,['model','accuracy','Macro_F1','Macro_AUPRC','CPU_forward_s','ms_per_beat'])}

Czas CPU to suma samych forward-passów, bez I/O i metryk; pojedynczy pomiar zależy od cache i obciążenia. Nie jest benchmarkiem klinicznym. AUPRC to średnia pięciu nieinterpolowanych one-vs-rest Average Precision; nie trapezoidalne pole pod PR. Macro F1 obejmuje wszystkie pięć klas. Recall=TP/(TP+FN), precision=TP/(TP+FP), specificity=TN/(TN+FP). W grupach bez danej klasy recall/F1 jest nieokreślony, nie dopisywano zerowego wsparcia jako wyniku.

{table(classrows,['model','group','support','precision','recall','specificity','F1','TP','FP','FN'])}

{minority_changes}

### Rozkład przewidywanych klas

{table(predictions,['model','N','S','V','F','Q'])}

### Pacjenci, rekordy i rzadkie klasy

{table(patients,['model','patient_group','beats','errors','S_recall','F_support','F_TP','F_FP','Q_recall'])}

{special_groups}

Dominujące pomyłki CNN+RR: N→F={b['confusion_matrix'][0][3]} (CNN:{a['confusion_matrix'][0][3]}), S→N={b['confusion_matrix'][1][0]} (CNN:{a['confusion_matrix'][1][0]}), S→V={b['confusion_matrix'][1][2]} (CNN:{a['confusion_matrix'][1][2]}), Q→N={b['confusion_matrix'][4][0]} (CNN:{a['confusion_matrix'][4][0]}). Wzrost minority recall ma więc realny koszt pomyłek N/F oraz S/V i niższej accuracy; nie należy przedstawiać go jako poprawy każdej klasy.

F ma tylko 38 przykładów walidacyjnych w {f_support_groups} grupach; pojedyncze TP ani wysoki recall w jednym rekordzie nie dowodzą niezawodnej generalizacji klasy F. Te grupy nie są 38 niezależnymi pacjentami. Tabele per rekord (osobno201/202) i per grupa są w comparison/record_class_metrics.csv oraz patient_group_class_metrics.csv. Silna koncentracja S w treningowym232 (80.58%) i różny skład oryginalnych symboli (np. a/f) pozostają ryzykiem zmiany rozkładu; nie zmieniono ich etykiet. N/S/V/F/Q to grupy beatów AAMI-style, nie pięć nazwanych diagnoz rytmu.

### Ablacja RR

Wszystkim walidacyjnym beatom podano tę samą reprezentację całkowicie brakującego RR: zapisane mediany treningowe, następnie zapisany Z-score oraz trzy zerowe maski. Wektor: {comparison['neutral_RR_vector']}. Nie są to literalne zera we wszystkich sześciu kolumnach. Macro F1 informative={b['macro_f1']:.6f}, neutral={neutral['macro_f1']:.6f}; Macro AUPRC={b['macro_auprc']:.6f} vs {neutral['macro_auprc']:.6f}. To diagnostyka zależności jednej sieci od RR, bez retreningu lub dostrajania decyzji. Sztuczne usunięcie historii u wszystkich beatów jest rzadkim/odmiennym rozkładem i nie izoluje przyczynowo czystego efektu RR od architektury.

## Wniosek

{interpretation}

Zmiana Macro F1 względem A: {b['macro_f1']-a['macro_f1']:+.6f}; Macro AUPRC: {b['macro_auprc']-a['macro_auprc']:+.6f}. Oprócz informacji RR głowica ma +192 parametrów i inny przebieg inicjalizacji losowej głowicy/dropoutu; jeden seed nie pozwala przypisać całej różnicy wyłącznie informacji czasowej. Powtarzany wybór na tym samym zbiorze walidacyjnym może go przeuczać. Nie oceniano końcowego testu i nie uruchomiono kolejnego wariantu. Wynik badawczy, bez walidacji klinicznej; Softmax nie jest skalibrowanym prawdopodobieństwem medycznym.

## Checkpointy, testy, odtwarzalność

- Best: `{OUT/'best_model.pt'}`; SHA256 `{sha256_file(OUT/'best_model.pt')}`.
- Ostatni resumowalny checkpoint: `{result['last_checkpoint']}` (RNG, Adam, best-state, historia). Ukończonego/early-stopped biegu nie należy wznawiać jako nowego eksperymentu.
- Baseline: `results/model_training/phase22_mitbih_full/run_001/best_model.pt`, SHA256 `{comparison['baseline_sha256']}`.
- Historia JSON/CSV i epoch_*_metrics.json; kompletne konfiguracje ECG/RR/scalera w best checkpoint i training_plan.json. Wersje zależności w ../preflight_001/reproducibility.json.
- Weryfikacja checkpoint-reload dała bitowo identyczne logity na ośmiu przykładach TRAIN; świeża inferencja walidacyjna odtworzyła macierze i metryki obu zapisanych modeli. Wagi nie zmieniły się w czasie inferencji.
- Testy ukierunkowane po treningu z guardem: **{tests} passed, {warnings} warnings**. Ostrzeżenia WFDB dotyczą przestarzałego inplace w pandas; nie są wyciszane. Dodatkowo preflight:27passed,2warnings. Testy wznowienia używały małych syntetycznych fixture'ów, nie dodatkowych pełnych eksperymentów.
- Polecenie testów: `.venv/bin/python -m pytest -p examples.pytest_mitbih_holdout_guard tests/test_mitbih_cnn_rr_training.py tests/test_mitbih_cnn_rr_evaluation.py tests/test_mitbih_rr.py tests/test_mitbih_holdout_guard.py tests/test_mitbih_training.py tests/test_mitbih_unweighted.py tests/test_mitbih_full_run.py tests/test_mitbih_setup.py -ra`. Ustawiano OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS=1 i osobną ścieżkę ECG_HOLDOUT_GUARD_LOG; log i audyt w katalogu nadrzędnym runu. Nie uruchamiano pełnego historycznego QC na wszystkich shardach.
- `ARTIFACT_SHA256.json` obejmuje nowe artefakty runu; istniejące dane i raporty nie zostały nadpisane. Skrypt treningu odmawia nadpisania istniejącego runu. W razie rzeczywistego przerwania API train_model obsługuje resume_from ostatniego poprawnego epoch_XXX.pt do NOWEGO katalogu przy identycznym planie; odrzuca częściową epokę i sprawdza SHA/config/source/RNG. Nie restartować automatycznie ukończonego eksperymentu.

Zatrzymano po Checkpoint B. Ewentualne kolejne eksperymenty lub końcowa ocena testowa wymagają osobnej decyzji.
'''.replace('{len_prior}',str(sum(not Path(p).is_relative_to(ROOT) for p in prior))).replace('{len_data}',str(len(stats))).replace('{len_rr}',str(len(rr)))
        handoff=f'''# Phase24B — kontrakt inferencji CNN+RR (offline)

Checkpoint `{OUT/'best_model.pt'}`, SHA256 `{sha256_file(OUT/'best_model.pt')}`. Wersje: `mitbih-cnn-rr-v1`, `mitbih-causal-rr-v1`, `mitbih-rr-transform-v1`. Nazwy i kolejność cech z rr_configuration.json oraz training_transform.json muszą być identyczne z checkpointem. Nie dopasowywać nowego scalera na uploadzie ani zbiorach walidacyjnych/testowych.

## ECG

Wejście float32 `[B,2,252]`, fs360Hz. Te same dwa kanały w oryginalnej kolejności, z zachowaniem ich nazw (typy leadów różnią się między rekordami). Offline Butterworth HP0.5Hz/LP45Hz order4 SOS zero-phase, bez notch; Z-score osobno dla pełnego rekordu i kanału przed segmentacją. Okno ekspertowego beatu `[annotation-90, annotation+162)`; środek adnotacji ma indeks90. Nie przesuwać szukając nowego piku. To nie kontrakt real-time: zero-phase i per-record mean/std wymagają całego rekordu, okno ma też przyszłe próbki ECG. Nie interpolować samowolnie brakujących kanałów lub innego fs; taka adaptacja wymaga odrębnej walidacji.

## RR

Dla monotonnie rosnących pozycji wszystkich rozpoznanych heartbeat annotations w JEDNYM rekordzie: poprzedniRR=(p[k]-p[k-1])/fs. Baseline to mediana do8 zakończonych wcześniejszych RR, WYŁĄCZAJĄC ten właśnie poprzedni interwał; ratio=previous/baseline. Używane są także poprawne beaty pominięte przy segmentacji na granicy ECG. Nie używać znaku klasy jako liczbowej cechy, nie liczyć RR między rekordami, także201/202.

Pierwszy beat ma trzy braki; drugi ma dostępny previousRR, ale brak baseline i ratio. Braki raw=0 plus bool mask; nigdy nie interpretować0 jako zaobserwowanego interwału. Trzy ciągłe wartości impute TRAIN median, następnie `(value-TRAINmean)/TRAINstd`. Dopisać trzy maski (0/1), bez ich skalowania. Kolejność: previous_rr_seconds, preceding_baseline_rr_seconds, previous_over_baseline, previous_rr_available, baseline_rr_available, ratio_available. Wejście float32 `[B,6]`, finite. Brak nextRR. Nie klipowano wartości fizjologicznie.

Zapisane mediany={plan['rr_training_transform']['imputation_median']}; mean={plan['rr_training_transform']['mean']}; std={plan['rr_training_transform']['scale']}. Pełny stan znajduje się w best checkpoint: preprocessing_configuration.RR_transform; konfiguracja RR w preprocessing_configuration.RR, ECG per rekord w preprocessing_configuration.ECG.

## Warunki wejścia aplikacji

1. Pełny MIT-BIH z ekspertowymi adnotacjami: pozycje mogą bezpośrednio zasilać RR i segmentację według kontraktu.
2. Surowy ciągły upload ECG: adnotacji zwykle nie ma. Konieczny jest osobno zaimplementowany i zwalidowany detektor QRS/beatów, znane fs i kanały. NIE dostarczono i NIE założono takiego detektora. Pominięte/fałszywe detekcje, timing jitter i rozbieżności definicji beatu zaburzają i morfologię, i RR; wymagają testów na niezależnych danych bez dostrajania obecnego testu.
3. Pojedynczy izolowany segment: nie wystarcza do odtworzenia kontekstu RR ani oryginalnego per-record Z-score. Technicznie możliwa jest maska wszystkich braków/mediany, ale ablacją nie udowodniono poprawności takiego zastosowania; nie traktować jej jako zwalidowanego trybu wdrożeniowego.

Wyjście `[B,5]` logitów, standardowy argmax; mapowanie N0 S1 V2 F3 Q4. N nie oznacza potwierdzonej zdrowej osoby ani normalnego rytmu. Q obejmuje /, f i Q; f (paced-normal fusion) różni się od F (ventricular-normal fusion). Brak progów, abstencji i walidacji klinicznej. Nie mieszać z osobnym PTB-XL modelem rytmu.

Model jest prototypem badawczym. Aktualnie oceniono wyłącznie zamrożoną walidację pacjentową; rzadkie F i przesunięcie rozkładu symboli ograniczają generalizację. Adapter nie ma danych końcowego testu. Nie uruchamiać UI/inferencji uploadów z założeniem, że detekcja beatów jest rozwiązana.
'''
        for name,text in [('PHASE24B_CNN_RR_TRAINING_REPORT.md',report),('PHASE24B_DEPLOYMENT_HANDOFF.md',handoff)]:
            with (OUT/name).open('x') as f:f.write(text)
        source_paths=['src/mitbih_cnn_rr_training.py','src/mitbih_cnn_rr_evaluation.py','src/mitbih_execution_guard.py','examples/train_mitbih_cnn_rr.py','examples/compare_mitbih_cnn_rr.py','examples/report_mitbih_cnn_rr.py','tests/test_mitbih_cnn_rr_training.py','tests/test_mitbih_cnn_rr_evaluation.py']
        write_json_new(OUT/'FINAL_INTEGRITY.json',dict(status='passed',historical_files_sha256_checked=sum(not Path(p).is_relative_to(ROOT) for p in prior),historical_changes=changed,dataset_files_stat_checked=len(stats),dataset_changes=changed_data,RR_files_sha256_checked=len(rr),RR_changes=changed_rr,unexpected_final_test_reads=0,historical_24A_QC_incident_acknowledged=True,test_count=tests,test_warnings=warnings,best_checkpoint_sha256=sha256_file(OUT/'best_model.pt'),validation_identity_verified=True,checkpoint_reload_exact=True,final_test_evaluated=False,new_execution_sources_sha256={p:sha256_file(p) for p in source_paths},guard_and_test_artifact_sha256={str(p.relative_to(ROOT)):sha256_file(p) for p in ROOT.glob('*.json')}))
        write_json_new(OUT/'ARTIFACT_SHA256.json',{str(p.relative_to(OUT)):sha256_file(p) for p in OUT.rglob('*') if p.is_file()})
        print(json.dumps(dict(status='complete',epochs=result['completed_epochs'],best_epoch=best['epoch'],seconds=seconds,metrics=b,tests=tests,warnings=warnings),indent=2))
