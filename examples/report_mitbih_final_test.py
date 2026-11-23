"""Phase26 REPORT ONLY: original final-test files stay under the deny guard."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.mitbih_execution_guard import HoldoutAccessGuard
ROOT=Path('results/model_evaluation/phase26_mitbih_final_test');OUT=ROOT/'run_001'

def read(p):
    import json
    return json.loads(Path(p).read_text())

def table(rows,keys):
    def cell(v):return 'nieokreślone' if v is None else (f'{v:.6f}' if isinstance(v,float) else str(v))
    return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+'\n'.join('| '+' | '.join(cell(r[k]) for k in keys)+' |' for r in rows)

if __name__=='__main__':
    with HoldoutAccessGuard(ROOT/'report_guard.json') as guard:
        guard.probe()
        import json
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from src.mitbih_final_reporting import collect_saved_predictions,metrics_from_saved
        from src.mitbih_full_run import csv_new
        from src.ptbxl_training import write_json_new
        from src.prepare_dataset import sha256_file
        if not (ROOT/'evaluation_completed.json').exists():raise PermissionError('No completed single test inference to report')
        protocol=read(OUT/'pre_evaluation_manifest.json');completion=read(ROOT/'evaluation_completed.json')
        if completion['checkpoint_sha256']!=protocol['checkpoint_sha256'] or sha256_file(OUT/'pre_evaluation_manifest.json')!=(OUT/'pre_evaluation_manifest.sha256').read_text().strip():raise ValueError('Frozen manifest identity differs')
        a=collect_saved_predictions(OUT,protocol);m,groups=metrics_from_saved(a)
        validation=read('results/model_training/phase24b_mitbih_cnn_rr/run_001/comparison/CNN_RR_metrics.json')
        timing=read(OUT/'inference_timing.json');cm=np.asarray(m['confusion_matrix'])
        write_json_new(OUT/'final_test_metrics.json',dict(**m,decision_rule='frozen five-class argmax',timing=timing,checkpoint_sha256=protocol['checkpoint_sha256'],final_test_used=True))
        write_json_new(OUT/'confusion_matrix.json',dict(labels=list('NSVFQ'),axes='rows=true, columns=argmax',counts=cm.tolist()))
        csv_new(OUT/'PHASE26_FINAL_TEST_CLASS_METRICS.csv',m['per_class'])
        csv_new(OUT/'patient_group_class_metrics.csv',[dict(patient_group=g,**r) for g,gm in groups.items() for r in gm['per_class']])
        group_rows=[dict(patient_group=g,beats=gm['beat_count'],accuracy=gm['accuracy'],errors=gm['beat_count']-int(np.trace(gm['confusion_matrix'])),F_support=gm['per_class'][3]['support'],F_TP=gm['per_class'][3]['tp'],F_FP=gm['per_class'][3]['fp'],S_support=gm['per_class'][1]['support'],S_recall=gm['per_class'][1]['recall'],Q_support=gm['per_class'][4]['support'],Q_recall=gm['per_class'][4]['recall']) for g,gm in groups.items()]
        csv_new(OUT/'patient_group_summary.csv',group_rows);write_json_new(OUT/'patient_group_metrics.json',groups)
        pred=a['predicted_indices'];p=a['probabilities']
        csv_new(OUT/'final_predictions.csv',[dict(beat_id=str(a['beat_ids'][i]),global_source_index=int(a['global_source_indices'][i]),record_id=str(a['record_ids'][i]),patient_group=str(a['patient_groups'][i]),annotation_index=int(a['annotation_indices'][i]),annotation_position=int(a['annotation_positions'][i]),original_symbol=str(a['original_symbols'][i]),actual_index=int(a['y'][i]),actual_group='NSVFQ'[a['y'][i]],predicted_index=int(pred[i]),predicted_group='NSVFQ'[pred[i]],**{f'uncalibrated_softmax_{c}':float(p[i,j]) for j,c in enumerate('NSVFQ')}) for i in range(len(pred))])
        comparison=[dict(scope='full_validation',beats=validation['beat_count'],accuracy=validation['accuracy'],macro_f1=validation['macro_f1'],macro_auprc=validation['macro_auprc']),dict(scope='final_test_once',beats=m['beat_count'],accuracy=m['accuracy'],macro_f1=m['macro_f1'],macro_auprc=m['macro_auprc'])]
        csv_new(OUT/'PHASE26_VALIDATION_VS_TEST_COMPARISON.csv',comparison)
        classcomparison=[dict(group=c,validation_support=validation['per_class'][i]['support'],test_support=m['per_class'][i]['support'],validation_percentage=100*validation['per_class'][i]['support']/validation['beat_count'],test_percentage=100*m['per_class'][i]['support']/m['beat_count'],validation_precision=validation['per_class'][i]['precision'],test_precision=m['per_class'][i]['precision'],validation_recall=validation['per_class'][i]['recall'],test_recall=m['per_class'][i]['recall'],validation_F1=validation['per_class'][i]['f1'],test_F1=m['per_class'][i]['f1'],test_supporting_groups=sum(gm['per_class'][i]['support']>0 for gm in groups.values())) for i,c in enumerate('NSVFQ')]
        csv_new(OUT/'validation_vs_test_class_distribution_and_metrics.csv',classcomparison)
        plt.rcParams.update({'font.size':11,'savefig.dpi':220})
        fig,ax=plt.subplots(figsize=(7,6.5),constrained_layout=True)
        ax.imshow(np.log1p(cm),cmap='Blues')
        for i in range(5):
            for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if np.log1p(cm[i,j])>6 else 'black')
        ax.set(xticks=range(5),yticks=range(5),xticklabels=list('NSVFQ'),yticklabels=list('NSVFQ'),xlabel='Predicted beat group (argmax)',ylabel='Actual beat group',title='Frozen CNN+RR — final MIT-BIH test\n23392 beats / 9 groups; counts, log1p color')
        fig.savefig(OUT/'PHASE26_FINAL_TEST_CONFUSION_MATRIX.png');plt.close(fig)
        off=sorted([(int(cm[i,j]),'NSVFQ'[i]+'→'+'NSVFQ'[j]) for i in range(5) for j in range(5) if i!=j],reverse=True)[:6]
        f=m['per_class'][3];predicted=[dict(group=c,predicted_count=int(cm[:,i].sum())) for i,c in enumerate('NSVFQ')]
        shifts='\n\n'.join(f"**{r['group']}:** precision {r['validation_precision']} → {r['test_precision']}; recall {r['validation_recall']:.4%} → {r['test_recall']:.4%}; F1 {r['validation_F1']:.6f} → {r['test_F1']:.6f}." for r in classcomparison)
        report=f'''# Phase26 — końcowy test zamrożonego MIT-BIH CNN+RR

## Jedna ocena i identyfikacja

Checkpoint `{protocol['checkpoint']}`, SHA256 `{protocol['checkpoint_sha256']}`. Model17989 parametrów, ECGfloat32[B,2,252], RRfloat32[B,6], pięć logitów N0/S1/V2/F3/Q4. Ten sam Phase24B wybrany PRZED testem; nie oceniano25B ani innych modeli. Argmax bez progów, abstencji lub tuningu. Brak treningu/fine-tuningu. Protokół i skaler zapisano przed otwarciem testu w pre_evaluation_manifest.json (plik read-only+SHA256); globalny evaluation_claim.json utworzono wyłącznie raz przed dostępem, a evaluation_completed.json oznacza test jako użyty i blokuje kolejną inferencję. To blokada aplikacyjna z uprawnieniami read-only, nie sprzętowy WORM.

## Kohorta i RR

Wszystkie **23392 beaty,9 rekordów i9 rozłącznych grup**, bez wspólnych grup z TRAIN/VAL. N19246/S575/V1402/F363/Q1806, zgodnie z zamrożonym manifestem. Wszystkie źródłowe NPZ/header/ATR sprawdzono checksumami w autoryzowanym przygotowaniu. Zachowano oryginalne kanały i ich kolejność, pozycje/symbole/indeksy adnotacji, ID/grupy i stabilne beat_id. Nie przesuwano adnotacji i nie przetwarzano ponownie ECG.

Testowe RR zapisano osobno w test_rr_features/, bez zmiany adaptera v1: previousRR, mediana do8 wcześniejszych interwałów Z WYŁĄCZENIEM bieżącego previousRR, iloraz oraz3maski. Wyłącznie historia w jednym rekordzie; bez nextRR i bez klas jako wejść liczbowych. Wykorzystano także poprawne beaty pominięte w pierwotnej segmentacji na granicach. Zastosowano dokładnie zapisane TRAIN-only mediany/mean/std. Nie fitowano ani nie adaptowano parametrów do testu. RR_preparation_integrity.json dokumentuje kolejność, braki, checksumy i finite wartości. ECG pozostaje oryginalnym eksportem offline:HP0.5/LP45Hz order4 zero-phase, bez notch; per-record/channel Z-score przed oknami[p−90,p+162). To kontrakt offline, nie real-time.

## Wyniki finalne

{table(comparison,['scope','beats','accuracy','macro_f1','macro_auprc'])}

{table(m['per_class'],['class_name','support','tp','fp','fn','tn','precision','recall','specificity','f1','auprc'])}

**F:TP={f['tp']},FP={f['fp']},FN={f['fn']}.** Największe pomyłki (true→predicted): {off}. Pełna macierz jest w confusion_matrix.json i PNG; wiersze=true, kolumny=argmax.

{table(predicted,['group','predicted_count'])}

Accuracy=trace(CM)/N; precision=TP/(TP+FP), recall=TP/(TP+FN), specificity=TN/(TN+FP), F1=2TP/(2TP+FP+FN). Macro F1 to średnia5 klas, bez wyrzucania trudnych klas. Macro AUPRC to średnia nieinterpolowanych one-vs-rest Average Precision (grupowane remisy), nie trapezoidalna PR-area. Brak mianownika→null; unsupported recall/F1/AP→null, macro→null jeżeli komponent nieokreślony. Żadnych fikcyjnych zer dla nieobecnych grup. Wyniki Softmax pozostają nieskalibrowane.

CPU2 wątki: suma samych forward-passów **{timing['forward_seconds']:.6f}s**, **{timing['inference_ms_per_beat']:.6f}ms/beat**, batch32 (ostatni batch osobno w każdym rekordzie), {timing['batch_count']} batchy. Eval+inference_mode, każdy z23392 beatów przewidziany dokładnie raz. Czas obejmujący całe autoryzowane przygotowanie/inferencję/zapis: {timing['total_authorized_seconds']:.2f}s. Forward timing wyklucza I/O, RR, Softmax i metryki; jeden pomiar zależy od hosta/cache. Wagi/bufory/checksum modelu nie zmieniły się.

## Walidacja kontra test — opis, nie ponowna selekcja

{table(classcomparison,['group','validation_support','test_support','validation_percentage','test_percentage','test_supporting_groups'])}

{shifts}

Największa zmiana częstości dotyczyF:38/23134 walidacji versus363/23392 testu. Precision zależy od częstości i składu pacjentów, dlatego różnicy nie wolno przypisywać wyłącznie jakości modelu. Nie łączymy walidacji i testu w jedną estymację. Żadnego progu/modelu/feature nie zmieniono po wyniku.

## Pacjenci i ograniczenia

{table(group_rows,['patient_group','beats','accuracy','errors','F_support','F_TP','F_FP','S_support','S_recall','Q_support','Q_recall'])}

Beaty w obrębie pacjenta/rekordu są skorelowane:23392 beaty nie oznaczają23392 niezależnych osób. Test obejmuje tylko9 grup technicznych, a klasy mogą dominować u pojedynczego pacjenta (liczby grup w tabeli). KlasaF jest nadal oceniana na małej liczbie niezależnych grup; nawet dobry wynik liczbowy nie dowodzi niezawodnego rozpoznawaniaF. Błędy per grupa opisujemy, nie wykorzystujemy do zmiany modelu. Rozbieżności rytmu/morfologii, leadów, adnotacji i symboli oraz heterogenicznaQ ograniczają transfer.

N/S/V/F/Q to AAMI-style grupy uderzeń, NIE pięć konkretnych diagnoz. N zawiera takżeL/R/e/j; S grupujeA/a/J/S, V=V/E, F=F, Q=/fQ. Symbolf (fusion paced-normal) nie jest F (fusion ventricular-normal). Nie wywodzić z tych grup potwierdzenia AFIB lub innych nazwanych rytmów. Model jest **prototypem badawczym, bez klinicznej walidacji diagnostycznej**.

## Historyczny incydent i zakres niezależności

W24A legacyQC niezamierzenie odczytał9 końcowych shardów. Według udokumentowanego incydentu dane nie weszły do treningu, fitu RR/scalera, modelowej inferencji ani selekcji. Nie deklarujemy, że pliki testowe nigdy wcześniej nie były czytane ani że jest to idealnie dziewiczy test. Późniejsze etapy stosowały guardy; obecna selekcja checkpointu/argmax/scalera została zamrożona przed autoryzowanym testem. To audytowalna niezależność procedury uczenia i selekcji z jawnym wcześniejszym naruszeniem I/O, a nie gwarancja braku wszelkiego wcześniejszego kontaktu z plikami. authorized_IO_audit.json rozróżnia odczyty przygotowania/checksum/ATR od jednego przebiegu NN; oryginalny plik może być otwarty więcej niż raz do walidacji bez powtarzania inferencji.

## Integracja i blokada po teście

Technicznie dostępne są zamrożony checkpoint, konfiguracja, skaler i kontrakt wejściowy do research-only handoff. Surowy upload wymaga osobnego, zwalidowanego detektora beatów i metadanych fs/kanałów; NIE dostarczono detektora. Wyniki z ekspertowymi adnotacjami nie dowodzą skuteczności end-to-end na uploadach. Izolowane okno nie dostarcza historiiRR ani per-record normalizacji. Nie wdrożono Streamlit.

Końcowy test jest teraz wykorzystany. Nie wolno na jego podstawie dopasowywać modelu, progów, cech lub preprocessingu ani porównywać innych modeli na tym teście. Ewentualne dalsze badania wymagają nowego niezależnego protokołu/danych i osobnej decyzji. Zachowano Phase22–25B i PTB-XL19. Testy/integrity po ewaluacji są w FINAL_INTEGRITY.json oraz PHASE26_TEST_PROTOCOL_AND_INTEGRITY.md i nie uruchamiają dodatkowej rzeczywistej inferencji. Zatrzymano po26.
'''
        with (OUT/'PHASE26_FINAL_MITBIH_TEST_REPORT.md').open('x') as f:f.write(report)
        print(json.dumps(dict(metrics=m,timing=timing,patient_groups=group_rows),indent=2))
