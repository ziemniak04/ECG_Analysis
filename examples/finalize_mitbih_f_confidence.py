"""Seal completed Phase25A diagnostics/tests without another model inference."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.mitbih_execution_guard import HoldoutAccessGuard
ROOT=Path('results/model_evaluation/phase25a_f_diagnostics');OUT=ROOT/'run_001'
if __name__=='__main__':
    with HoldoutAccessGuard(OUT/'final_integrity_guard.json') as guard:
        guard.probe()
        import json,re
        import numpy as np
        from src.prepare_dataset import sha256_file
        from src.ptbxl_training import write_json_new
        from src.mitbih_full_run import csv_new
        from src.mitbih_f_decisions import score_summary,threshold_metrics,THRESHOLDS
        data=np.load(OUT/'validation_scores.npz',allow_pickle=False)
        y=data['y'];p=data['probabilities'];groups=data['patient_groups'];pred=p.argmax(1);margin=data['F_margin']
        summaries=[]
        for name in ('mitdb_group_201_202','mitdb_group_210','OTHER'):
            group=~np.isin(groups,['mitdb_group_201_202','mitdb_group_210']) if name=='OTHER' else groups==name
            for category,mask in [('correct_F',(y==3)&(pred==3)),('false_positive_F',(y!=3)&(pred==3)),('missed_actual_F',(y==3)&(pred!=3))]:
                for var,v in [('F_softmax',p[:,3]),('F_margin',margin)]:summaries.append(dict(patient_group=name,category=category,variable=var,**score_summary(v[group&mask])))
        csv_new(OUT/'F_AGGREGATED_GROUP_COMPARISON.csv',summaries)
        # Round-trip deterministic reconstruction from saved validation outputs, no inference.
        result=json.loads((OUT/'analysis_result.json').read_text())
        assert [threshold_metrics(y,p,t) for t in THRESHOLDS]==result['rows']
        tests=(ROOT/'tests_001.log').read_text();m=re.search(r'(\d+) passed, (\d+) warnings in',tests)
        if not m or 'FAILED ' in tests:raise ValueError('Tests not passed')
        tg=json.loads((ROOT/'test_guard_001.json').read_text())
        if tg['exit_status'] or tg['actual_heldout_reads_through_guarded_open']:raise ValueError('Test isolation failed')
        if any(a['test']!='tests/test_mitbih_holdout_guard.py::test_real_heldout_open_is_blocked_before_reader (call)' for a in tg['blocked_file_attempts']):raise ValueError('Unexpected test probe')
        runtime=json.loads((OUT/'runtime_guard.json').read_text())
        assert runtime['unexpected_denial_count']==0 and runtime['exit_exception'] is None
        before=json.loads((OUT/'preservation_before.json').read_text())
        changed=[p for p,h in before.items() if not Path(p).is_relative_to(ROOT) and sha256_file(p)!=h]
        if changed:raise ValueError(f'Historical changes:{changed}')
        appendix='''

## Końcowa interpretacja wyników i testy

Mediana nieskalibrowanego P(F): trafneF=0.605281 (6), fałszyweF=0.844360 (2612), pominięteF=0.022660 (32). Fałszywe predykcje zwykle nie są niskiej pewności; zaledwie25.84% par TP/FP ma wyższy score po stronie TP (remisy0.5). 94.03% fałszywychF mieści się między minimum i maksimum sześciu trafnychF. Nakładanie i odwrócona kolejność confidence utrudniają odseparowanie false alarms samym progiem. To opis modelu, nie diagnoza przyczyn medycznych.

W201/202 jest1235FP, medianaP(F)=0.923045 i brakTP; w210 jest1120FP z medianą0.802676 oraz1TP zP(F)=0.515828. Zatem w210 próg0.60 odrzuca również jedyne trafneF. Dla pozostałych grup dostępne są zagregowane statystyki confidence i marginesów w F_AGGREGATED_GROUP_COMPARISON.csv; pełne rozbicie grup w pozostałych CSV. Szczególnie201/202 zawiera pewne, lecz błędneF — prosty cutoff nie usuwa większości z nich.

Próg0.40:22 odrzucenia,TP6/FP2590,precision0.2311%,recall15.79%,coverage99.90%. Poprawa względem0.2292% precision jest pomijalna praktycznie i niepewna statystycznie. Próg0.60:TP3/FP2189,precision0.1369%,recall7.89%,426UNCERTAIN. Próg0.70:TP1/FP1849,recall2.63%,768UNCERTAIN. Próg0.95:TP1/FP678,precision0.1473%,recall2.63%,1939UNCERTAIN,coverage91.62%. Większość redukcjiFP odbywa się kosztem utraty już rzadkichTP i coverage, bez poprawy precision ponadargmax.

Dla0.95 abstencja według rzeczywistej klasy: N1848/19094=9.68%, S20/491=4.07%, V66/1444=4.57%, F5/38=13.16%, Q0/2067=0%. Accepted-only Accuracy87.37% i MacroF1=0.571168 dotyczą tylko21195 zaakceptowanych przykładów, wobec pełnego argmax Accuracy80.07%/MacroF1=0.556844 na23134. Nie stanowi to poprawy pełnego klasyfikatora. Wszystkie odrzucenia pozostają UNCERTAIN, bez reassignment.

Rekomendacja: nie przyjmować nowego progu wdrożeniowego na podstawie tej walidacji. Zachować argmax jako zamrożoną regułę referencyjną, nie jako dowód bezpiecznego wykrywaniaF. Zamiast szukać kolejnych progów, rozważyć jeden osobno zatwierdzony eksperyment z mniejszą względną wagąF w loss. Żadnego eksperymentu25B nie rozpoczęto.

Testy ukierunkowane: **14passed,2warnings** (deprecacja inplace w pandas używanego przez WFDB; bez wyciszania). Polecenie: `.venv/bin/python -m pytest -p examples.pytest_mitbih_holdout_guard tests/test_mitbih_f_decisions.py tests/test_mitbih_cnn_rr_evaluation.py tests/test_mitbih_holdout_guard.py -ra`, z osobnym ECG_HOLDOUT_GUARD_LOG i OMP/OPENBLAS/MKL_NUM_THREADS=1. Test kontrolny celowo próbuje otworzyć chronioną ścieżkę i potwierdza odmowę przed odczytem; brak nieoczekiwanych prób. Odtworzenie CSV ze zapisanych wyników dało identyczne metryki. Historyczne źródła/checkpointy/raporty sprawdzono SHA256; stare eksporty rozmiarem/mtime, adapterRR SHA256. Nie otwierano końcowych shardów do checksumowania. Zakres tej kontroli, SHAcheckpointu i odczytane10 rekordów walidacyjnych zapisano w integrity.json. Pełnego legacyQC nie uruchamiano.

Reprodukcja analizy: `.venv/bin/python examples/analyze_mitbih_f_confidence.py` (tworzy następny nowy run, nie nadpisuje istniejącego). Nie jest to komenda treningu. Bieżący wynik jest zamrożony; źródła i artefakty mają sumy w ARTIFACT_SHA256.json.
'''
        with (OUT/'PHASE25A_F_FALSE_POSITIVE_ANALYSIS.md').open('a') as f:f.write(appendix)
        with (OUT/'PHASE25A_NEXT_EXPERIMENT_RECOMMENDATION.md').open('a') as f:f.write('\nZweryfikowany wynik25A: mediana scoreFP0.844 > TP0.605; próg0.70 pozostawia1TP i1849FP, a0.95 pozostawia1TP i678FP. Żaden próg nie spełnia wspólnego ekranuTP/FP/coverage. Nie wybrano nowej reguły decyzyjnej. To uzasadnia badanie loss-weight jako kontrolowanej hipotezy, bez obietnicy poprawy. Testy14passed/2warnings; model pozostaje zamrożony.\n')
        write_json_new(OUT/'TEST_RESULTS.json',dict(passed=int(m[1]),warnings=int(m[2]),guarded=True,unexpected_test_access=False,metric_roundtrip_exact=True,changed_historical_files=changed))
    # Seal after the last guard log has been closed; all hashes below concern audit outputs only.
    write_json_new(OUT/'ARTIFACT_SHA256.json',{str(p.relative_to(OUT)):sha256_file(p) for p in OUT.rglob('*') if p.is_file()})
    print('Phase25A sealed;14 tests passed; no training or test evaluation.')
