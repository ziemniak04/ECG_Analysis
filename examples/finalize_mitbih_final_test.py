"""Seal Phase26 outputs; only saved results/metadata, no model or test-shard reads."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.mitbih_execution_guard import HoldoutAccessGuard
ROOT=Path('results/model_evaluation/phase26_mitbih_final_test');OUT=ROOT/'run_001'

def read(p):
    import json
    return json.loads(Path(p).read_text())

if __name__=='__main__':
    with HoldoutAccessGuard(ROOT/'final_integrity_guard.json') as guard:
        guard.probe()
        import json,re,time
        from src.prepare_dataset import sha256_file
        from src.ptbxl_training import write_json_new
        protocol=read(OUT/'pre_evaluation_manifest.json');claim=read(ROOT/'evaluation_claim.json');complete=read(ROOT/'evaluation_completed.json')
        digest=sha256_file(OUT/'pre_evaluation_manifest.json')
        assert digest==claim['protocol_sha256']==(OUT/'pre_evaluation_manifest.sha256').read_text().strip()
        assert complete['checkpoint_sha256']==protocol['checkpoint_sha256']==sha256_file(protocol['checkpoint'])
        for p,h in protocol['source_files_sha256'].items():
            if sha256_file(p)!=h:raise ValueError('Frozen inference source changed')
        before=read(OUT/'preservation_before.json')
        changed=[p for p,h in before.items() if not Path(p).is_relative_to(ROOT) and (not Path(p).is_file() or sha256_file(p)!=h)]
        stats=read(OUT/'dataset_stats_before.json')
        changed_data=[p for p,s in stats.items() if not Path(p).is_file() or [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=s]
        rr=read(OUT/'RR_preservation_before.json');changed_rr=[p for p,h in rr.items() if sha256_file(p)!=h]
        if changed or changed_data or changed_rr:raise ValueError('Historical model/data/source artifact changed')
        io=read(OUT/'authorized_IO_audit.json')
        if io['denied_open_events'] or any(Path(p).suffix=='.dat' for p in io['allowed_open_events']):raise ValueError('Unexpected authorized I/O')
        for path in io['allowed_open_events']:
            if Path(path).suffix not in ('.npz','.hea','.atr'):raise ValueError('Unknown source I/O type')
        for filename in ('preflight_guard.json','report_guard.json'):
            log=read(ROOT/filename)
            if log['exit_exception'] is not None or log['unexpected_denial_count']:raise ValueError('Unexpected deny-guard access')
        testtext=(ROOT/'final_post_evaluation_tests.log').read_text();match=re.search(r'(\d+) passed, (\d+) warnings in',testtext)
        if not match or 'FAILED ' in testtext:raise ValueError('Tests did not pass')
        tests,warnings=map(int,match.groups());testguard=read(ROOT/'final_post_evaluation_test_guard.json')
        if testguard['exit_status'] or testguard['actual_heldout_reads_through_guarded_open']:raise ValueError('Test access violation')
        if any(a['test']!='tests/test_mitbih_holdout_guard.py::test_real_heldout_open_is_blocked_before_reader (call)' for a in testguard['blocked_file_attempts']):raise ValueError('Unexpected test read attempted')
        m=read(OUT/'final_test_metrics.json');v=read('results/model_training/phase24b_mitbih_cnn_rr/run_001/comparison/CNN_RR_metrics.json');groups=read(OUT/'patient_group_metrics.json')
        f=m['per_class'][3];q=m['per_class'][4];s=m['per_class'][1]
        appendix=f'''

## Końcowa interpretacja i weryfikacja

Globalne Macro F1 wzrosło względem walidacji {v['macro_f1']:.6f} → {m['macro_f1']:.6f}, Macro AUPRC {v['macro_auprc']:.6f} → {m['macro_auprc']:.6f}, Accuracy {v['accuracy']:.4%} → {m['accuracy']:.4%}. To opis dwóch różnych kohort, bez wyboru modelu po teście. Nie jest to równomierna poprawa wszystkich klas.

**S pozostaje słaba:** precision {s['precision']:.4%}, recall {s['recall']:.4%}, {s['fn']} pominiętych i{s['fp']} FP; pogorszenie względem walidacji. W207 jest107 rzeczywistychS i0TP; w234 jest50S i0TP. W209181 z187 łącznychTP S pochodzi z jednej grupy. KlasaV ma wysokie recall, ale niższe precision/F1 niż walidacja. N poprawia F1.

**F jest liczbowo znacznie lepsza niż walidacja**, precision {f['precision']:.4%}, recall {f['recall']:.4%}, F1 {f['f1']:.6f}, TP{f['tp']}/FP{f['fp']}/FN{f['fn']}. Jednak362 z363 F jest w213; wszystkie205 trafnychF pochodzą z213, jedyneF z203 zostało pominięte. PonadtoF jest w teście około9.45razy częstsza procentowo niż w walidacji. Tak skoncentrowany wynik i154FP/158FN nie dowodzą niezawodnej generalizacjiF ani medycznej użyteczności.

**Q:** recall {q['recall']:.4%}, ale precision tylko {q['precision']:.4%}, wobec97.74% w walidacji. Wszystkie{q['fp']} FP Q to N→Q;1248 z1253 pochodzi z207. Grupa217 zawiera1802 z1806 rzeczywistychQ, w niej są wszystkie1545TP Q. Accuracy w207 wynosi tylko {groups['mitdb_group_207']['accuracy']:.4%}; globalna Accuracy83.68% maskuje bardzo słabą generalizację u tego pacjenta. Nie przypisujemy temu medycznej przyczyny na podstawie predykcji.

**Testy po ewaluacji: {tests}passed,{warnings}warnings.** WFDB/pandas deprecation inplace, bez wyciszania. Testy odtwarzają metryki z zapisanych wyników przy zabronionym wywołaniu modelu, sprawdzają blokadę kolejnej oceny przed I/O i regułyRR na syntetycznych timestampach. Nie wykonywały kolejnej rzeczywistej inferencji ani legacyQC. Szczegóły: ../final_post_evaluation_tests.log oraz guardJSON. Poprzednie preflight14passed/2warnings i wcześniejsze post24passed/2warnings zachowano jako osobne logi; nie sumujemy powtórzonych testów.

Sprawdzono SHA256 {sum(not Path(p).is_relative_to(ROOT) for p in before)} chronionych historycznych plików, {len(rr)} plików RR v1 oraz rozmiar/mtime {len(stats)} plików dawnych eksportów; żadnych zmian. SHA oryginalnych9 testowych shardów i ich ATR/HEA weryfikowano W RAMACH autoryzowanego przygotowania; nie otwierano ich ponownie do końcowegoQC. Sumy64bit SHA256 modelu, manifestu i artefaktów są zapisane. Integralność i blokada są w FINAL_INTEGRITY.json oraz ../final_results_lock.json. Nie zmieniono Phase22–25B ani ukończonego PTB-XL19.
'''.replace('Sumy64bit SHA256','Sumy SHA256')
        with (OUT/'PHASE26_FINAL_MITBIH_TEST_REPORT.md').open('a') as fp:fp.write(appendix)
        policy=f'''# Phase26 — zamrożony protokół i integralność

- Checkpoint: `{protocol['checkpoint']}`; SHA256 `{protocol['checkpoint_sha256']}`. Architektura17989, klasy N0/S1/V2/F3/Q4, X[2,252],RR[6]. Wybrany przedtestem Phase24B, nigdy25B.
- Manifest przed-ewaluacyjny SHA256 `{digest}`, zapisany read-only przed dostępem do źródeł. Zapisuje frozen split/source/code SHA, konfiguracjeECG/RR, TRAIN-only mediany/mean/std, pakiety/Python/CPU i autoryzację.
- Globalny exclusive-create `evaluation_claim.json` poprzedził odczyty. Nawet błąd konsumuje próbę; nie ma automatycznego restartu lub wznowienia NN. `evaluation_completed.json` potwierdza zakończenie. Unit test potwierdza, że istniejący claim blokuje preflight przed dostępem do checkpointu/datasetu.
- Guard preflight blokował45 ścieżek. Po jego zamknięciu osobna, wąska audit-allowlista pozwalała tylko na read-only9NPZ i odpowiednieATR/HEA. Nigdy rawDAT. Po wyjściu zakresu pozostaje zamknięta. authorized_IO_audit.json zawiera {len(io['allowed_open_events'])} zdarzeń open do checksum/parsowania/przygotowania — to nie liczba przebiegów NN.
- Model eval/inference_mode, CPU2, batch32, argmax; {m['beat_count']} przykładów dokładnie jeden raz w736batchach z granicami rekordów. Brak Softmax progowania, abstencji, modelselection, aktualizacji wag lub skalera. Wagi i BNbufory bitowo niezmienione po jedynym przebiegu.
- RRfeatures to zamrożona funkcja rr_features, bez nextRR i bez granic międzyrecordowych. transform_rr używa wyłącznie starego TRAINstate. Nie wywołano fit_training_transform. Nowe testowe RR są tylko w wynikach26, nie w RRv1.
- Raportowanie i testy działają pod przywróconą deny-blokadą. Czytają jedynie zapisane predykcje testowe w katalogu wyników. Test real_saved_results patchuje Module.__call__ tak, aby kolejna inferencja przerwała test; metryki odtworzono dokładnie.
- Historyczny24Aincydent: legacyQC odczytał9shardów bez użycia do uczenia/scalowania/modelowej oceny/selekcji. Nie deklarujemy nigdy wcześniej nieodczytanego testu ani idealnie dziewiczej niezależności. Późniejsza izolacja i zamrożona selekcja są audytowalne, z ujawnionym wcześniejszym naruszeniemI/O.
- Testy: {tests}passed/{warnings}warnings, bez legacyQC, dodatkowej rzeczywistej inferencji lub źródłowych odczytów. Metryki zero-division=null; macro obejmuje5klas, AP nieinterpolowaneOVR.
- Koniec: `final_results_lock.json` rejestruje test jako użyty i SHA manifestu artefaktów. Nowe pliki wynikowe mają uprawnienia read-only; to nie nieodwracalny sprzętowyWORM. Nie wolno powtarzać testu, dobierać innych modeli/progów lub używać wyników do kolejnego tuningu. Dalszy rozwój wymaga osobnego planu i niezależnych danych/protokołu.
- Techniczny handoff research-only jest możliwy przy zachowaniu pełnego kontraktuECG/RR. Surowe uploady nadal wymagają oddzielnie zwalidowanego detektora beatów; nie zaimplementowano go ani Streamlit. Prototyp nie jest klinicznie zwalidowanym urządzeniem diagnostycznym.
'''
        with (OUT/'PHASE26_TEST_PROTOCOL_AND_INTEGRITY.md').open('x') as fp:fp.write(policy)
        write_json_new(OUT/'FINAL_INTEGRITY.json',dict(status='passed',historical_sha256_checks=sum(not Path(p).is_relative_to(ROOT) for p in before),historical_changes=changed,dataset_stat_checks=len(stats),dataset_changes=changed_data,RR_v1_sha256_checks=len(rr),RR_changes=changed_rr,checkpoint_sha256=protocol['checkpoint_sha256'],manifest_sha256=digest,exactly_one_final_model_pass=True,beats=23392,groups=9,parameters_unchanged=True,test_used=True,no_additional_test_inference=True,tests_passed=tests,test_warnings=warnings,historical_24A_read_acknowledged=True,authorized_IO_events=len(io['allowed_open_events']),denied_authorized_scope_events=io['denied_open_events']))
    write_json_new(OUT/'ARTIFACT_SHA256.json',{str(p.relative_to(OUT)):sha256_file(p) for p in OUT.rglob('*') if p.is_file()})
    claim_once_payload=dict(status='final MIT-BIH test USED and results LOCKED; no rerun or post-test model selection',checkpoint_sha256=protocol['checkpoint_sha256'],run=str(OUT),artifacts_sha256=sha256_file(OUT/'ARTIFACT_SHA256.json'),metric_sha256=sha256_file(OUT/'final_test_metrics.json'),claim_sha256=sha256_file(ROOT/'evaluation_claim.json'),completion_sha256=sha256_file(ROOT/'evaluation_completed.json'),time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),guard_sha256={p.name:sha256_file(p) for p in ROOT.glob('*guard.json')})
    with (ROOT/'final_results_lock.json').open('x') as f:json.dump(claim_once_payload,f,indent=2)
    (ROOT/'final_results_lock.json').chmod(0o444)
    for p in OUT.rglob('*'):
        if p.is_file():p.chmod(0o444)
    print(json.dumps(dict(status='completed, sealed and read-only',tests=tests,warnings=warnings,metrics={k:m[k] for k in ('accuracy','macro_f1','macro_auprc')},inference_time=read(OUT/'inference_timing.json')),indent=2))
