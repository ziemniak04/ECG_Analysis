"""Read-only Phase25B interpretation, preservation audit and Polish reports."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.mitbih_execution_guard import HoldoutAccessGuard
ROOT=Path('results/model_training/phase25b_reduced_f');OUT=ROOT/'run_001'

def read(path):
    import json
    return json.loads(Path(path).read_text())

def table(rows,keys):
    def cell(v):return 'nieokreślone' if v is None else (f'{v:.6f}' if isinstance(v,float) else str(v))
    return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+'\n'.join('| '+' | '.join(cell(r[k]) for k in keys)+' |' for r in rows)

if __name__=='__main__':
    with HoldoutAccessGuard(ROOT/'runtime_guard_final_integrity_001.json') as guard:
        guard.probe()
        import json,re
        import numpy as np
        from src.mitbih_full_run import validated_history
        from src.prepare_dataset import sha256_file
        from src.ptbxl_training import write_json_new
        result,plan,history,best=validated_history(OUT)
        names=('Phase24B_CNN_RR','Phase25B_Fhalf_CNN_RR')
        metrics={n:read(OUT/f'comparison/{n}_metrics.json') for n in names}
        groups={n:read(OUT/f'comparison/{n}_patient_metrics.json') for n in names}
        a,b=(metrics[n] for n in names);fa=a['per_class'][3];fb=b['per_class'][3]
        comparison=read(OUT/'comparison/integrity.json')
        if not comparison['checkpoint_reload_logits_exact'] or comparison['unexpected_guard_denials']:raise ValueError('Comparison failed')
        tests=(ROOT/'post_training_tests_001.log').read_text();match=re.search(r'(\d+) passed(?:, (\d+) warnings)? in',tests)
        if not match or 'FAILED ' in tests:raise ValueError('Relevant tests not passed')
        test_count=int(match[1]);warning_count=int(match[2] or 0)
        testguard=read(ROOT/'post_training_test_guard_001.json')
        if testguard['exit_status'] or testguard['actual_heldout_reads_through_guarded_open']:raise ValueError('Test guard failed')
        if any(row['test']!='tests/test_mitbih_holdout_guard.py::test_real_heldout_open_is_blocked_before_reader (call)' for row in testguard['blocked_file_attempts']):raise ValueError('Unexpected test probe')
        for filename in ('runtime_guard_training_001.json','runtime_guard_comparison_001.json'):
            g=read(ROOT/filename)
            if g['unexpected_denial_count'] or g['exit_exception'] is not None:raise ValueError('Isolation failed')
        prior=read(ROOT/'preflight_001/preservation_before.json')
        changed=[p for p,h in prior.items() if not Path(p).is_relative_to(ROOT) and (not Path(p).is_file() or sha256_file(p)!=h)]
        stats=read(ROOT/'preflight_001/dataset_stats_before.json')
        changed_data=[p for p,s in stats.items() if not Path(p).is_file() or [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=s]
        rr=read(ROOT/'preflight_001/rr_preservation_before.json')
        changed_rr=[p for p,h in rr.items() if not Path(p).is_file() or sha256_file(p)!=h]
        manifest=read('data/processed/mitbih_rr_context_v1/manifest.json')
        if any(sha256_file(p)!=h for p,h in manifest['raw_source_sha256'].items()):raise ValueError('TRAIN/VAL source annotation changed')
        if any(sha256_file(r['source_X_file'])!=r['source_X_sha256'] for r in manifest['records']):raise ValueError('TRAIN/VAL X changed')
        if changed or changed_data or changed_rr:raise ValueError('Historical artifacts changed')
        duration=read(OUT/'execution.json')['wall_seconds'];weights=plan['class_weights']
        expected=list(plan['original_class_weights']);expected[3]*=.5
        if weights!=expected or plan['loss_weight_adjustment']['renormalized']:raise ValueError('Weights differ')
        collapse=fb['tp']==0 or any(b['per_class'][i]['recall']==0 for i in (1,2,4))
        candidate='Phase25B' if b['macro_f1']>a['macro_f1'] and not collapse else 'Phase24B'
        reason=('Phase25B ma wyższe pełnowalidacyjne Macro F1 i nie wyzerował TP F ani recall S/V/Q.' if candidate=='Phase25B' else 'Phase25B nie spełnia kryterium zastąpienia: wyższe Macro F1 bez wyzerowania TP F lub recall S/V/Q. Sama redukcja FP nie wystarcza.')
        summary=[dict(model=n,accuracy=metrics[n]['accuracy'],Macro_F1=metrics[n]['macro_f1'],Macro_AUPRC=metrics[n]['macro_auprc'],unweighted_CE=metrics[n]['unweighted_validation_ce']['loss'],own_weighted_CE=metrics[n]['experiment_weighted_validation_ce']['loss'],CPU_forward_seconds=metrics[n]['forward_seconds'],CPU_ms_per_beat=metrics[n]['inference_ms_per_beat']) for n in names]
        classrows=[dict(model=n,**r) for n in names for r in metrics[n]['per_class']]
        changes=[]
        for i,c in enumerate('NSVFQ'):
            x,z=a['per_class'][i],b['per_class'][i]
            changes.append(dict(group=c,support=z['support'],P24=x['precision'],P25=z['precision'],R24=x['recall'],R25=z['recall'],F1_24=x['f1'],F1_25=z['f1'],TP24=x['tp'],TP25=z['tp'],FP24=x['fp'],FP25=z['fp'],FN24=x['fn'],FN25=z['fn']))
        patientrows=[]
        for n in names:
            for group,m in groups[n].items():
                patientrows.append(dict(model=n,patient_group=group,beats=m['beat_count'],errors=m['beat_count']-int(np.trace(m['confusion_matrix'])),F_support=m['per_class'][3]['support'],F_TP=m['per_class'][3]['tp'],F_FP=m['per_class'][3]['fp'],S_recall=m['per_class'][1]['recall'],V_recall=m['per_class'][2]['recall'],Q_recall=m['per_class'][4]['recall']))
        special=[]
        for group in ('mitdb_group_201_202','mitdb_group_210'):
            x,z=(groups[n][group] for n in names)
            special.append(f"**{group}:** F FP {x['per_class'][3]['fp']} → {z['per_class'][3]['fp']}; F TP {x['per_class'][3]['tp']} → {z['per_class'][3]['tp']}; wszystkie błędy {x['beat_count']-int(np.trace(x['confusion_matrix']))} → {z['beat_count']-int(np.trace(z['confusion_matrix']))}.")
        concentration={n:sum(groups[n][g]['per_class'][3]['fp'] for g in ('mitdb_group_201_202','mitdb_group_210'))/metrics[n]['per_class'][3]['fp'] if metrics[n]['per_class'][3]['fp'] else None for n in names}
        q104a=groups[names[0]]['mitdb_group_104']['per_class'][4]
        q104b=groups[names[1]]['mitdb_group_104']['per_class'][4]
        class_tradeoffs='\n\n'.join(f"**{c}:** recall {a['per_class'][i]['recall']:.4%} → {b['per_class'][i]['recall']:.4%}; precision {a['per_class'][i]['precision'] if a['per_class'][i]['precision'] is not None else 'nieokreślone'} → {b['per_class'][i]['precision'] if b['per_class'][i]['precision'] is not None else 'nieokreślone'}; F1 {a['per_class'][i]['f1']:.6f} → {b['per_class'][i]['f1']:.6f}." for i,c in enumerate('NSVFQ') if c in 'SVQ')
        historyrows=[dict(epoch=r['epoch'],train_loss=r['train']['loss'],validation_loss=r['validation']['loss'],validation_Macro_F1=r['validation']['macro_f1'],validation_Macro_AUPRC=r['validation']['macro_auprc'],seconds=r['seconds']) for r in history]
        predictions=[dict(model=n,**dict(zip('NSVFQ',map(int,np.asarray(metrics[n]['confusion_matrix']).sum(0))))) for n in names]
        fp_eliminated=fa['fp']-fb['fp']
        recommendation=f'''# Phase25B — rekomendacja następnego kroku

Preferowany kandydat badawczy: **{candidate}**, według pierwotnego kryterium pełnowalidacyjnego Macro F1 z kontrolą minority recall, nie samego FP. {reason}

Macro F1: {a['macro_f1']:.6f} → {b['macro_f1']:.6f}; Macro AUPRC: {a['macro_auprc']:.6f} → {b['macro_auprc']:.6f}. F FP:{fa['fp']} → {fb['fp']} (redukcja ze znakiem:{fp_eliminated}); F TP:{fa['tp']} → {fb['tp']}; FN:{fa['fn']} → {fb['fn']}. Model nie jest uznany za skuteczny tylko dzięki mniejszej liczbie predykcjiF. Ocena wszystkich klas i pacjentów znajduje się w raporcie/CSV; spadki S/V/Q muszą być uwzględnione w decyzji.

W grupie104 jest {q104b['support']} z {b['per_class'][4]['support']} walidacyjnych Q, a Q TP zmieniło się {q104a['tp']} → {q104b['tp']}. Liczba6 trafnych F nie oznacza wykrycia tych samych zdarzeń lub poprawy u tych samych pacjentów: per-grupa wyniki są odmienne. Zmniejszenie wagiF×0.5 w tym jednym biegu nie rozwiązało całościowego problemu generalizacji. Nie dowodzi to, że każda możliwa zmiana wag musi zawieść; dalszych wariantów nie badano.

To jeden seed i ponownie używana walidacja, nie dowód niezawodnej generalizacji. Tylko38F w4 grupach to za mało do samodzielnych silnych wniosków. Preferencja badawcza nie jest decyzją wdrożeniową ani zgodą na ocenę końcowego testu. Zachować oba modele i argmax, bez progów. Nie dopasowywać dalszych wag automatycznie. Ewentualny następny etap wymaga osobnej decyzji i zamrożonego planu; w tej fazie nie otwierano final-test shards ani nie uruchamiano PTB-XL lub Streamlit.
'''
        report=f'''# Phase25B — jeden kontrolowany eksperyment z wagą F ×0.5

## Stan przy wznowieniu sesji

Przed wznowieniem gotowe były preflight,11 testów, konfiguracja, cały trening oraz12 checkpointów epokowych i best_model.pt. Proces treningu zakończył się poprawnie z early stopping: best8, koniec12. Nie restartowano treningu ani nie wznawiano aktualizacji optymalizatora. Wznowiona sesja uzupełniła wyłącznie porównanie zamrożonych modeli, testy, wykresy, raporty i kontrolę integralności. Szczegóły w ../resumption_audit_001.json.

## Integralność i zakres

Referencja: `results/model_training/phase24b_mitbih_cnn_rr/run_001/best_model.pt`, SHA256 `{comparison['baseline_sha256']}`. Nowy model: `{OUT/'best_model.pt'}`, SHA256 `{sha256_file(OUT/'best_model.pt')}`. Oba zamrożone checkpointy ponownie oceniono na identycznych23134 walidacyjnych beatach, sprawdzając źródłowe indeksy/record_id/grupę/symbol/pozycję/kanały i y. Inferencja eval/inference_mode, standardowy argmax, coverage100%, bez progów/abstencji. Wagi nie zmieniły się w czasie inferencji; reload best versus zapisany best-state ostatniego checkpointu dał bitowo identyczne logity na8 przykładach TRAIN.

Nie odczytywano końcowych shardów testowych w25B i nie obliczano metryk testowych. Guard Python działał od preflightu:45 kontrolnych odmów przed odczytem i0 nieoczekiwanych prób. Starszy incydent QC z24A pozostaje jawny; nie twierdzimy, że test nigdy wcześniej nie był czytany. Nie uruchamiano legacyQC. Nie zmieniono modeli PTB-XL, źródeł/preprocessingu/danych/etykiet ani splitu. Integralność historycznych plików potwierdzono SHA256; eksporty sprawdzono także rozmiarem/mtime (bez otwierania testu), adapterRR SHA256. Szczegóły w FINAL_INTEGRITY.json, preflight_001 i logach guardów.

## Jedyna zmiana eksperymentalna

Wagi sqrt-balanced wyliczono istniejącą funkcją wyłącznie z TRAIN62926,29 grup: N52254/S1715/V4389/F401/Q4167. Po dotychczasowej normalizacji wag do średniej1 pomnożono tylko F przez0.5; **bez późniejszej renormalizacji**. Oryginalne wagi={plan['original_class_weights']}; użyte={weights}. Cztery pozostałe wartości są bitowo identyczne z referencją.

Ta sama architektura CNN+RR17989 parametrów, Xfloat32[B,2,252], RRfloat32[B,6], pięć logitów N0/S1/V2/F3/Q4. Ten sam adapterRR, treningowe mediany/mean/std i maski, to samo ECG. Model od świeżej losowej inicjalizacji seed42, nie od24B ani smoke. Adam.001,batch32,CPU2 wątki,max15epok,patience4,min_delta0; dokładnie ta sama kolejność source i shuffle(seed+epoch). Test porównuje tekst pętli train_model z24B: identyczny. Reużyto make_loader,epoch_pass,load_checkpoint,RNG/Adam persistence bez modyfikacji źródeł historycznych. Jedynie plan wag ma rozszerzenie25B.

`experiment_configuration.json` jawnie zapisuje mnożnik F i dokładne wagi. Sam bazowy `configuration.json` nie koduje tej zmiany: używać modułu `src.mitbih_reduced_f_training`, pełnego `training_plan.json` i tego rozszerzenia konfiguracji. Nie uruchamiać starszego trainera z samą konfiguracją bazową, bo odtworzyłby pierwotne wagi.

VAL23134,9 grup/10rekordów, N19094/S491/V1444/F38/Q2067. Grupy rozłączne;201/202 razem wVAL, RR oddzielnie na rekord. CE otrzymuje surowe logity; exact pooled loss=sum(w_y*CE)/sum(w_y). Ponieważ zmieniono wektor wag, own_weighted_CE między modelami nie jest identycznym celem; do bezpośredniego porównania strat służy wspólne unweighted_CE. Treningowe straty są online przed update'ami, walidacyjne po epoce.

## Trening

Ukończono **{result['completed_epochs']} epok**, best **{best['epoch']}**, status **{result['status']}**. Czas train_model **{duration:.2f}s ({duration/60:.2f}min)**, z ładowaniem/metrykami/checkpointami, bez preflightu i porównania. Best wyłącznie przez Macro F1 na CAŁEJ walidacji,5klas. Ostatni checkpoint z RNG/Adam/historią: `{result['last_checkpoint']}`. Bez automatycznego kolejnego eksperymentu.

{table(historyrows,['epoch','train_loss','validation_loss','validation_Macro_F1','validation_Macro_AUPRC','seconds'])}

## Pełna walidacja obu zamrożonych modeli

{table(summary,['model','accuracy','Macro_F1','Macro_AUPRC','unweighted_CE','own_weighted_CE','CPU_forward_seconds','CPU_ms_per_beat'])}

CPU forward to jeden pomiar samej inferencji bez I/O/metryk, przy2wątkach; zależy od cache/obciążenia. Macro F1 uwzględnia5klas. Macro AUPRC to średnia nieinterpolowanego one-vs-rest Average Precision. Precision=TP/(TP+FP), recall=TP/(TP+FN), specificity=TN/(TN+FP); brak zdefiniowanego mianownika→null. W grupach bez danej rzeczywistej klasy nie dopisujemy fikcyjnych recall/F1.

{table(changes,['group','support','P24','P25','R24','R25','F1_24','F1_25','TP24','TP25','FP24','FP25','FN24','FN25'])}

### Kompletne metryki klas

{table(classrows,['model','class_name','precision','recall','specificity','f1','tp','fp','fn','tn'])}

### Rozkład predykcji

{table(predictions,['model','N','S','V','F','Q'])}

### Pacjenci

{table(patientrows,['model','patient_group','beats','errors','F_support','F_TP','F_FP','S_recall','V_recall','Q_recall'])}

{chr(10).join(special)}

Udział dwóch wskazanych grup we wszystkich F FP: {concentration}. Błędy i poprawy nie są równomierne między pacjentami; oddzielne201 i202 są w comparison/record_class_metrics.csv. Nie wnioskujemy o medycznych przyczynach z samych predykcji. Walidacja zawiera zaledwie38F z4grup; kilka TP lub wzrost precision przy nielicznych predykcjach nie dowodzi niezawodnej generalizacjiF. S w treningu jest silnie skoncentrowana w232, a skład oryginalnych symboli train/VAL różni się; etykiet nie zmieniono.

Grupa104 zawiera {q104b['support']} z {b['per_class'][4]['support']} Q walidacji. Liczba Q TP w104 spadła {q104a['tp']} → {q104b['tp']}; to ważny, pacjentowo skoncentrowany koszt, a nie dowód ogólnego działania na wielu niezależnych pacjentach. Trafne F zmieniły rozkład: w210 {groups[names[0]]['mitdb_group_210']['per_class'][3]['tp']} → {groups[names[1]]['mitdb_group_210']['per_class'][3]['tp']}, w223 {groups[names[0]]['mitdb_group_223']['per_class'][3]['tp']} → {groups[names[1]]['mitdb_group_223']['per_class'][3]['tp']}. Ta sama liczba TP nie gwarantuje zachowania tych samych trafień.

## Interpretacja i rekomendacja

Preferencja badawcza: **{candidate}**. {reason}

F FP zmieniło się o **{-fp_eliminated:+d}**, TP {fa['tp']}→{fb['tp']}, FN {fa['fn']}→{fb['fn']}. Precision F:{fa['precision']}→{fb['precision']}; F1 F:{fa['f1']}→{fb['f1']}. Macro F1 zmiana **{b['macro_f1']-a['macro_f1']:+.6f}**, AP **{b['macro_auprc']-a['macro_auprc']:+.6f}**. Zestawienie recall S/V/Q wyżej pokazuje wszystkie koszty i korzyści — mniejszy FP sam nie wystarcza. Nie uznajemy zaprzestania rozpoznawania F za sukces. To badawczy kompromis, nie kliniczna walidacja, a wielokrotna selekcja na tej samej walidacji może ją przeuczać. Nie oceniano nowego niezależnego zbioru/testu. Bez gwarancji, że pojedynczy seed powtórzy przewagę.

{class_tradeoffs}

## Artefakty i testy

PHASE25B_MODEL_COMPARISON.csv i PHASE25B_CLASS_METRICS.csv; training_history.csv/json i epoch_*_metrics.json; best_model.pt oraz epoch_XXX.pt z SHA256; comparison/validation_confusion_matrices.png i training_validation_curves.png; predykcje/probabilities, per-grupa i per-rekordCSV; pełny plan/preprocessing/scaler w checkpointach i training_plan.json; reproducibility.json w preflight_001. Epoch-boundary resume wyłącznie do nowego katalogu przy identycznym planie/źródłach/SHA, z odrzuceniem niedokończonej epoki; nie wznawiać zakończonego biegu jako kolejnego eksperymentu.

Testy po treningu z guardem: **{test_count}passed,{warning_count}warnings**. WFDB/pandas deprecation inplace nie wyciszano. Preflight11passed/2warnings. Testy wag i resume używały małych syntetycznych fixture'ów; to nie dodatkowe eksperymenty pełnego treningu. Pełnego legacyQC nie wykonywano. Zatrzymano po25B, bez progowania lub następnego wariantu.
'''
        for name,text in [('PHASE25B_REDUCED_F_WEIGHT_REPORT.md',report),('PHASE25B_NEXT_STEP_RECOMMENDATION.md',recommendation)]:
            with (OUT/name).open('x') as f:f.write(text)
        write_json_new(OUT/'FINAL_INTEGRITY.json',dict(status='passed',historical_changes=changed,history_sha256_checks=sum(not Path(p).is_relative_to(ROOT) for p in prior),dataset_stat_checks=len(stats),dataset_changes=changed_data,RR_SHA_checks=len(rr),RR_changes=changed_rr,reference_sha256=comparison['baseline_sha256'],candidate_sha256=sha256_file(OUT/'best_model.pt'),test_count=test_count,warnings=warning_count,preferred_research_candidate=candidate,final_test_evaluated=False,unexpected_guard_denials=0,source_sha256={str(p):sha256_file(p) for p in [Path('src/mitbih_reduced_f_training.py'),Path('examples/train_mitbih_reduced_f.py'),Path('examples/compare_mitbih_reduced_f.py'),Path('examples/report_mitbih_reduced_f.py'),Path('tests/test_mitbih_reduced_f_training.py')]}))
        print(json.dumps(dict(candidate=candidate,result=result,seconds=duration,metrics=summary,changes=changes,tests=test_count,warnings=warning_count),indent=2))
    write_json_new(OUT/'ARTIFACT_SHA256.json',{str(p.relative_to(OUT)):sha256_file(p) for p in OUT.rglob('*') if p.is_file()})
