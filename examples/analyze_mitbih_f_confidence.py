"""Phase25A: guarded, frozen, full-validation inference and selective F diagnostics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.mitbih_execution_guard import HoldoutAccessGuard
ROOT=Path('results/model_evaluation/phase25a_f_diagnostics')
if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    number=1
    while (ROOT/f'run_{number:03d}').exists():number+=1
    out=ROOT/f'run_{number:03d}';out.mkdir()
    with HoldoutAccessGuard(out/'runtime_guard.json') as guard:
        probe=guard.probe()
        import json,time
        import numpy as np
        import torch
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from src.mitbih_f_decisions import THRESHOLDS,threshold_metrics,score_summary
        from src.mitbih_cnn_rr_evaluation import evaluate
        from src.mitbih_rr_dataset import MITBIHRRDataset,DEFAULT_RR_ROOT
        from src.mitbih_protocol import MITBIHExportIndex,load_split
        from src.mitbih_readiness import verify_frozen_protocol
        from src.mitbih_training import loss_weights
        from src.mitbih_full_run import snapshot,csv_new
        from src.prepare_dataset import sha256_file
        from src.ptbxl_training import write_json_new
        checkpoint=Path('results/model_training/phase24b_mitbih_cnn_rr/run_001/best_model.pt')
        expected_sha='9440f3494af59d3272d7c8de115c03b592d2738279957d97b5e7a6ac9c235536'
        if sha256_file(checkpoint)!=expected_sha:raise ValueError('Frozen checkpoint identity differs')
        write_json_new(out/'prespecified_analysis.json',dict(thresholds=THRESHOLDS,selection='Exploratory screen only: retain >=5 of baseline6 F TP, reduce F FP >=25%, coverage>=95%; prefer highest coverage; otherwise retain argmax as reference, recommend no threshold deployment',no_new_training=True,UNCERTAIN=-1,threshold_inclusive=True,F_recall_denominator='all actual F including rejected ones',accepted_only_macro='fixed5 classes, null if accepted population lacks a true class'))
        snapshot(out)
        rr_before={str(p):sha256_file(p) for p in DEFAULT_RR_ROOT.rglob('*') if p.is_file()}
        index=MITBIHExportIndex();split=load_split(index);verify_frozen_protocol(index,split)
        dataset=MITBIHRRDataset(index=index,split=split,partition='validation',max_cached_shards=10)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        plan=json.loads((checkpoint.parent/'training_plan.json').read_text())
        if saved['preprocessing_configuration']['RR']!=plan['rr_configuration'] or saved['preprocessing_configuration']['ECG']!=plan['ECG_preprocessing_by_record']:raise ValueError('Preprocessing config differs')
        if saved['preprocessing_configuration']['RR_transform']!=dataset.transform or dataset.transform['fitted_partition']!='train':raise ValueError('Scaler differs')
        torch.set_num_threads(2)
        metrics,groups,metadata,y,p,ce=evaluate(checkpoint,dataset,split,loss_weights(index,split,'sqrt_balanced'))
        if metrics['beat_count']!=23134 or metrics['per_class'][3]['tp']!=6 or metrics['per_class'][3]['fp']!=2612:raise ValueError('Frozen full-validation results not reproduced')
        previous=json.loads((checkpoint.parent/'comparison/CNN_RR_metrics.json').read_text())
        np.testing.assert_array_equal(metrics['confusion_matrix'],previous['confusion_matrix'])
        pred=p.argmax(1);score=p[:,3];margin=score-np.max(p[:,[0,1,2,4]],axis=1)
        conditions={'correct_F':(y==3)&(pred==3),'false_positive_F':(y!=3)&(pred==3),'missed_actual_F':(y==3)&(pred!=3)}
        patients=np.asarray([r['patient_group'] for r in metadata]);records=np.asarray([r['record_id'] for r in metadata])
        write_json_new(out/'frozen_validation_metrics.json',metrics)
        with (out/'validation_scores.npz').open('xb') as f:np.savez_compressed(f,y=y,probabilities=p,patient_groups=patients,record_ids=records,source_indices=np.asarray([r['source_index'] for r in metadata]),F_margin=margin)
        rows=[threshold_metrics(y,p,t) for t in THRESHOLDS]
        csv_new(out/'F_THRESHOLD_COMPARISON.csv',rows)
        summary=[]
        for group in ['ALL']+sorted(set(patients)):
            gm=np.ones(len(y),dtype=bool) if group=='ALL' else patients==group
            for name,mask in conditions.items():
                for variable,values in [('uncalibrated_F_softmax',score),('F_minus_best_nonF_probability_margin',margin)]:
                    summary.append(dict(patient_group=group,category=name,variable=variable,**score_summary(values[gm&mask])))
        csv_new(out/'F_SCORE_SUMMARIES.csv',summary)
        group_thresholds=[]
        for group in sorted(set(patients)):
            mask=patients==group
            group_thresholds.extend(dict(patient_group=group,**threshold_metrics(y[mask],p[mask],t)) for t in THRESHOLDS)
        csv_new(out/'F_PATIENT_THRESHOLD_COMPARISON.csv',group_thresholds)
        csv_new(out/'beat_diagnostics.csv',[dict(**r,actual_class='NSVFQ'[int(y[i])],argmax_class='NSVFQ'[int(pred[i])],uncalibrated_F_softmax=float(score[i]),F_minus_best_nonF_margin=float(margin[i])) for i,r in enumerate(metadata)])
        tp=score[conditions['correct_F']];fp=score[conditions['false_positive_F']]
        ranking=float(np.mean((tp[:,None]>fp[None,:])+.5*(tp[:,None]==fp[None,:])))
        bins=np.linspace(0,1,21);ht=np.histogram(tp,bins)[0]/len(tp);hf=np.histogram(fp,bins)[0]/len(fp)
        overlap=float(np.minimum(ht,hf).sum())
        span_fraction=float(((fp>=tp.min())&(fp<=tp.max())).mean())
        write_json_new(out/'score_overlap.json',dict(argmax_F_TP_vs_FP_pairwise_ranking=ranking,histogram_overlap_20_fixed_bins=overlap,false_positive_fraction_in_TP_minmax_range=span_fraction,interpretation='Descriptive only:6 true detections, correlated patients; bin/range dependent; no calibration claim'))
        plt.rcParams.update({'font.size':10,'savefig.dpi':220})
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        colors=['#228833','#CC3311','#4477AA']
        for (name,mask),color in zip(conditions.items(),colors):
            values=score[mask]
            axes[0,0].hist(values,bins=bins,weights=np.ones(len(values))/len(values),histtype='step',linewidth=2,label=f'{name} (n={len(values)})',color=color)
            values=np.sort(values);axes[0,1].step(values,np.arange(1,len(values)+1)/len(values),where='post',label=name,color=color)
            axes[1,0].hist(margin[mask],bins=np.linspace(-1,1,41),weights=np.ones(mask.sum())/mask.sum(),histtype='step',linewidth=2,label=name,color=color)
        axes[0,0].set(xlabel='Uncalibrated Softmax F score',ylabel='Fraction within category',title='F score distributions; unequal counts')
        axes[0,1].set(xlabel='Uncalibrated Softmax F score',ylabel='Empirical cumulative fraction',title='Score overlap (not calibration)')
        axes[1,0].set(xlabel='P(F) minus highest non-F score',ylabel='Fraction within category',title='Positive margin: argmax F')
        for group,color in zip(('mitdb_group_201_202','mitdb_group_210','OTHER'),colors):
            gm=~np.isin(patients,['mitdb_group_201_202','mitdb_group_210']) if group=='OTHER' else patients==group
            values=np.sort(score[gm&conditions['false_positive_F']])
            if len(values):axes[1,1].step(values,np.arange(1,len(values)+1)/len(values),where='post',label=f'{group} FP n={len(values)}',color=color)
        axes[1,1].set(xlabel='Uncalibrated Softmax F score',ylabel='Empirical cumulative fraction',title='False positives by patient group')
        for ax in axes.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.suptitle('Frozen CNN+RR; full validation only (23134 beats). Actual F=38 across4 groups.')
        fig.savefig(out/'F_CONFIDENCE_DISTRIBUTION.png');plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True)
        ax.plot([r['coverage'] for r in rows],[r['F_fp'] for r in rows],'-o')
        for r in rows:ax.annotate(str(r['threshold']),(r['coverage'],r['F_fp']),fontsize=8)
        ax.set(xlabel='Coverage (all validation beats)',ylabel='F false-positive count',title='Threshold labels; rejected argmax-F → UNCERTAIN')
        ax.grid(alpha=.2);fig.savefig(out/'F_COVERAGE_TRADEOFF.png');plt.close(fig)
        def table(data,keys):
            def cell(v):return 'nieokreślone' if v is None else (f'{v:.6f}' if isinstance(v,float) else str(v))
            return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+'\n'.join('| '+' | '.join(cell(r[k]) for k in keys)+' |' for r in data)
        feasible=[r for r in rows[1:] if r['F_tp']>=5 and r['F_fp']<=.75*2612 and r['coverage']>=.95]
        chosen=max(feasible,key=lambda r:r['coverage']) if feasible else None
        recommendation=('Żaden próg nie spełnił wszystkich uprzednio zapisanych warunków diagnostycznych.' if chosen is None else f"Próg {chosen['threshold']} spełnia operacyjny ekran: TP={chosen['F_tp']}, FP={chosen['F_fp']}, coverage={chosen['coverage']:.4%}. To kandydat do dalszej oceny, nie zatwierdzona reguła wdrożeniowa.")
        compact=[r for r in summary if r['variable']=='uncalibrated_F_softmax' and r['patient_group'] in ('ALL','mitdb_group_201_202','mitdb_group_210')]
        report=f'''# Phase25A — diagnostyka fałszywych F (wyłącznie walidacja)

Checkpoint `{checkpoint}`, SHA256 `{expected_sha}`. Architektura17989 parametrów, wejścia [B,2,252] i [B,6], klasy N0/S1/V2/F3/Q4. Zamrożone konfiguracje ECG i RR oraz TRAIN-only scaler/imputer zweryfikowano względem planu/checkpointu i sidecaru. Nie wykonywano ponownie preprocessingu, fitu skalera, treningu ani aktualizacji wag.

Pełna walidacja:23134 beaty,9 grup,10 rekordów. F:38 beatów w4 grupach; argmax TP6,FP2612,FN32,precision={6/2618:.6f},recall={6/38:.6f}. Odtworzono identyczną macierz Phase24B. Guard przed importami/inferencją odmówił45 kontrolnych otwarć przed odczytem. Nie było nieoczekiwanych prób dostępu do końcowego testu. Incydent QC z24A pozostaje udokumentowany; nie twierdzimy, że testowe pliki nigdy wcześniej nie były czytane. Guard to kontrola Python, nie sandbox OS.

## Ustalony wcześniej plan

Siatka={THRESHOLDS};0 oznacza oryginalny argmax. Tylko gdy argmax=F i P(F)<próg, wynik=UNCERTAIN(-1). Równość progu jest akceptowana. Nie promujemy F, które nie jest argmax, i nie przypisujemy drugiej klasy. Recall F=TP/wszystkie38F, także po odrzuceniach. Precision=TP/(TP+FP), brak przewidywańF→null. Coverage=zaakceptowane/wszystkie. Per-class abstention=odrzucone danej klasy/wszystkie rzeczywiste danej klasy.

Accepted-only Accuracy i Macro F1 dotyczą wyłącznie zaakceptowanych przykładów; nie są pełnopopulacyjną skutecznością. Macro ma ustalone5 klas i jest null, jeśli w zaakceptowanym podzbiorze brakuje rzeczywistej klasy. Wzrost metryk warunkowych przez odrzucanie trudnych przypadków nie dowodzi poprawy klasyfikatora.

Operacyjny ekran ustalony przed inferencją: zachować>=5 z6 TP F, zmniejszyćFP>=25%, coverage>=95%, a spośród spełniających preferować najwyższe coverage. To jawnie umowny filtr diagnostyczny, nie próg medyczny ani gwarancja użytecznego wykrywaniaF; podstawowy recall6/38 jest już niski. Nie optymalizujemy wyłącznie precision.

{table(rows,['threshold','F_tp','F_fp','F_fn','F_precision','F_recall','abstentions','coverage','accepted_only_accuracy','accepted_only_macro_f1'])}

## Rozkłady i marginesy

Softmax to nieskalibrowane wyniki modelu. Margines=P(F)-max(P(N),P(S),P(V),P(Q)); nie jest logitowym ani medycznym prawdopodobieństwem. Dodatni oznacza przewagęF (przy remisach działa deterministyczny argmax).

{table(compact,['patient_group','category','count','minimum','q25','median','q75','maximum'])}

Porównanie punktowe TP-score z FP-score (wygranaTP, remis=0.5):{ranking:.6f}; opisowy overlap histogramów20 stałych binów:{overlap:.6f}; odsetek FP w zakresie min–max sześciu trafnychF:{span_fraction:.4%}. To niepewne deskryptywne miary, zależne od bardzo małych TP i grupowania/binowania, bez niezależności beatów. Nie dostrajano binów ani siatki po obejrzeniu wyników. F_SCORE_SUMMARIES.csv zawiera również marginesy i wszystkie grupy; F_PATIENT_THRESHOLD_COMPARISON.csv zawiera wyniki per grupa i próg. Puste kategorie są oznaczone count0/null, nie wypełniane fikcyjnymi danymi.

## Wniosek

{recommendation}

Nie zalecamy uznania thresholdingu za rozwiązanie problemuF: nawet zachowanie wszystkich6 TP daje tylko15.79% recall, a liczba38 F z4 grup jest zbyt mała do stabilnego wyboru. Poprzednie wielokrotne wybory modeli na tej samej walidacji zwiększają ryzyko przeuczenia decyzji. Wyniki per próg i per grupa mają charakter eksploracyjny. Dalsze zwiększanie progu może usuwać trafneF razem z false alarms. Wykresy nie dowodzą kalibracji ani przyczyn medycznych błędów.

Do rozważenia osobno zatwierdzony eksperyment z mniejszą wagąF w loss, przy identycznej architekturze/split/preprocessingu i pozostałych ustawieniach; nie uruchomiono go. Ryzyko: spadek już słabego recallF lub przesunięcie błędów do innych klas. Nie zmieniono obecnego argmax/checkpointu. Szczegóły w PHASE25A_NEXT_EXPERIMENT_RECOMMENDATION.md.

## Pliki i bezpieczeństwo

F_CONFIDENCE_DISTRIBUTION.png: rozkłady, ECDF, marginesy i różnice201/202–210–pozostałe. F_COVERAGE_TRADEOFF.png: coverage kontraFP. Wszystkie osie opisują wyniki nieskalibrowane. validation_scores.npz i beat_diagnostics.csv pozwalają odtworzyć wyniki bez ponownej inferencji. Oryginalne dane/modele nie zostały nadpisane. Testy i końcowa kontrola integralności są zapisane w osobnych plikach tej wersji.
'''
        with (out/'PHASE25A_F_FALSE_POSITIVE_ANALYSIS.md').open('x') as f:f.write(report)
        with (out/'PHASE25A_NEXT_EXPERIMENT_RECOMMENDATION.md').open('x') as f:f.write(f'''# Następny eksperyment — propozycja, nie uruchomienie

{recommendation}

Nie wdrażać progu jako medycznie uzasadnionego. Zapoznać się z całym tradeoffem FP/TP/coverage i abstencją poszczególnych klas. MałoF, tylko4 grupy oraz ponowne użycie walidacji ograniczają pewność. Nie szukać kolejnych progów na tej samej siatce po fakcie i nie otwierać testu.

Proponowany kontrolowany wariant: mnożnik0.5 wyłącznie dla obecnej treningowej wagiF w sqrt-balanced CE (ew. renormalizacja wszystkich wag do średniej1, która nie zmienia względnego ważenia CE). Zachować pozostałe wagi względne, seed42, Adam.001, batch32, CPU2,15epok/patience4, ten sam split i best-by-full-validation Macro F1. Mnożnik0.5 jest hipotezą techniczną jednego ograniczonego eksperymentu, nie optymalną/klinicznie ustaloną wartością. Architektura17989, ECG/RR/scaler i etykiety pozostają identyczne. Bez nowej wersji danych, ale nowa wersja konfiguracji/modelu i osobna zgoda są konieczne.

Porównać z zamrożonym24B: Macro F1/AP, N/S/V/F/Q precision/recall, F FP/FN, coverage1 przy argmax i błędy per grupa201/202/210. Nie wybierać wyłącznie najniższego FP: wymagane zachowanie minority recall i ogólnej jakości; kryterium ustalić przed uruchomieniem. Możliwa poprawaFP jest hipotezą, a ryzykiem jest całkowite pomijanieF. 38 F nie wystarcza do wiarygodnych samodzielnych wniosków o generalizacji. Potrzebne dalsze dane/ocena patient-independent w osobno uzgodnionym protokole; nie zmieniać obecnego final-test lock.

Nie trenowano ani nie fine-tunowano żadnego modelu w25A. Nie rozpoczęto25B.
''')
        prior=json.loads((out/'preservation_before.json').read_text())
        changed=[p for p,h in prior.items() if not Path(p).is_relative_to(ROOT) and (not Path(p).is_file() or sha256_file(p)!=h)]
        stats=json.loads((out/'dataset_stats_before.json').read_text())
        changed_data=[p for p,s in stats.items() if not Path(p).is_file() or [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=s]
        changed_rr=[p for p,h in rr_before.items() if sha256_file(p)!=h]
        if changed or changed_data or changed_rr or guard.unexpected:raise ValueError('Integrity or isolation failed')
        write_json_new(out/'integrity.json',dict(status='passed',checkpoint_sha256_before=expected_sha,checkpoint_sha256_after=sha256_file(checkpoint),historical_sha256_files_checked=sum(not Path(p).is_relative_to(ROOT) for p in prior),changed_historical_files=changed,processed_file_stat_checks=len(stats),changed_processed_files=changed_data,RR_sha256_files_checked=len(rr_before),RR_changes=changed_rr,expected_guard_probe=probe,unexpected_guard_denials=len(guard.unexpected),source_records_read=index.read_records,parameters_unchanged=metrics['parameters_unchanged'],final_test_evaluated=False))
        write_json_new(out/'analysis_result.json',dict(output=str(out),diagnostic_candidate_threshold=None if chosen is None else chosen['threshold'],recommendation=recommendation,overlap=overlap,ranking=ranking,rows=rows))
        print(json.dumps(dict(output=str(out),candidate=chosen,ranking=ranking,overlap=overlap,score_summaries=compact,thresholds=rows),indent=2),flush=True)
