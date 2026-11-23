"""One authorized held-out inference; metrics/report can use saved predictions only."""
import argparse
import csv
import json
from pathlib import Path
import sys
import platform
import numpy as np

from src.ptbxl_final_evaluation import prepare,evaluate_once,DEFAULT_OUTPUT,CHECKPOINT
from src.ptbxl_metrics import CLASS_NAMES,classification_metrics
from src.ptbxl_protocol import file_sha256
from src.ptbxl_training import write_json_new


def csv_new(path,rows):
    with path.open('x',newline='') as target:
        writer=csv.DictWriter(target,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize_saved(output):
    """No model or waveform access: compute metrics/plots from the single saved pass."""
    output=Path(output)
    integrity=json.loads((output/'inference_integrity.json').read_text())
    if integrity['read_records']!=1958 or not integrity['model_parameters_and_buffers_unchanged']:
        raise ValueError('Complete frozen inference required')
    if (output/'test_metrics.json').exists():raise FileExistsError('Test metrics already saved; do not overwrite')
    with np.load(output/'test_predictions.npz',allow_pickle=False) as stored:
        data={key:stored[key] for key in stored.files}
    y,prob=data['labels'],data['probabilities']
    if len(y)!=1958 or len(set(data['ecg_id'].tolist()))!=1958 or set(data['strat_fold'].tolist())!={10}:
        raise ValueError('Saved test coverage or identity differs')
    if not np.array_equal(data['predicted_labels'],prob.argmax(axis=1)):
        raise ValueError('Saved decisions must be original argmax')
    metrics=classification_metrics(y,prob)
    write_json_new(output/'test_metrics.json',metrics)
    csv_new(output/'test_class_metrics.csv',metrics['per_class'])
    validation=json.loads((CHECKPOINT.parent/'best_validation_metrics.json').read_text())
    rows=[]
    for key in ('accuracy','macro_f1','macro_auprc'):
        rows.append({'metric':key,'validation_fold_9':validation[key],'test_fold_10':metrics[key],
                     'test_minus_validation':metrics[key]-validation[key]})
    for i,name in enumerate(CLASS_NAMES):
        for key in ('precision','recall','specificity','f1','auprc'):
            left,right=validation['per_class'][i][key],metrics['per_class'][i][key]
            rows.append({'metric':name+'_'+key,'validation_fold_9':left,'test_fold_10':right,
                         'test_minus_validation':right-left if left is not None and right is not None else None})
    csv_new(output/'validation_test_comparison.csv',rows)
    predictions=[]
    for i,eid in enumerate(data['ecg_id']):
        predictions.append({'ecg_id':int(eid),'patient_id':int(data['patient_id'][i]),'strat_fold':10,
            'true_class':CLASS_NAMES[y[i]],'predicted_class':CLASS_NAMES[data['predicted_labels'][i]],
            **{f'p_{name}':float(prob[i,j]) for j,name in enumerate(CLASS_NAMES)}})
    csv_new(output/'test_predictions.csv',predictions)
    errors=[r for r in predictions if r['true_class']!=r['predicted_class']]
    if errors:csv_new(output/'test_errors.csv',errors)
    sbrad_fp=[r for r in predictions if r['predicted_class']=='SBRAD' and r['true_class']!='SBRAD']
    if sbrad_fp:csv_new(output/'sbrad_false_positives.csv',sbrad_fp)
    cm=np.asarray(metrics['confusion_matrix'])
    breakdown=[{'predicted_class':name,'actual_class':other,'false_positive_records':int(cm[j,i])}
               for i,name in enumerate(CLASS_NAMES) for j,other in enumerate(CLASS_NAMES) if i!=j]
    csv_new(output/'false_positive_sources.csv',breakdown)
    with (output/'test_confusion_matrix.csv').open('x',newline='') as target:
        writer=csv.writer(target);writer.writerow(['true/predicted',*CLASS_NAMES])
        for name,row in zip(CLASS_NAMES,cm):writer.writerow([name,*row.tolist()])
    cohort=json.loads((output/'test_cohort_manifest.json').read_text())
    distribution=[{'class':name,'records':int((y==i).sum()),'percentage':float(100*(y==i).mean()),
                   'patients':len({r['patient_id'] for r in cohort if r['label']==i})} for i,name in enumerate(CLASS_NAMES)]
    csv_new(output/'test_class_distribution.csv',distribution)
    false_analysis={'SBRAD_false_positives':len(sbrad_fp),
        'SBRAD_false_positive_patients':len({r['patient_id'] for r in sbrad_fp}),
        'SBRAD_false_positives_by_true_class':{name:int(cm[i,3]) for i,name in enumerate(CLASS_NAMES) if i!=3},
        'SR_to_any_target_arrhythmia_false_alerts':int(cm[0,1:].sum()),
        'AFIB_recall':metrics['per_class'][1]['recall'],
        'note':'Source rhythm labels are reference; SR is not proof of a globally healthy ECG.'}
    write_json_new(output/'false_positive_analysis.json',false_analysis)
    write_json_new(output/'report_context.json',{'metrics':metrics,'validation_reference':{
        key:validation[key] for key in ('record_count','accuracy','macro_f1','macro_auprc','per_class')},
        'distribution':distribution,'false_positive_analysis':false_analysis})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    normalized=cm/cm.sum(axis=1,keepdims=True)
    for ax,matrix,title in zip(axes,(cm,normalized),('Test fold 10 — counts','Test fold 10 — row-normalized')):
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=float(matrix.max()))
        for i in range(4):
            for j in range(4):
                text=str(int(matrix[i,j])) if title.endswith('counts') else f'{100*matrix[i,j]:.1f}%'
                ax.text(j,i,text,ha='center',va='center',color='white' if matrix[i,j]>.5*matrix.max() else 'black')
        ax.set(xticks=range(4),xticklabels=CLASS_NAMES,yticks=range(4),yticklabels=CLASS_NAMES,
               xlabel='Predicted class (argmax)',ylabel='True class',title=title)
    with (output/'TEST_CONFUSION_MATRIX.png').open('xb') as target:fig.savefig(target,format='png',dpi=300)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    x=np.arange(4)
    for ax,key in zip(axes,('recall','precision')):
        ax.bar(x-.18,[r[key] if r[key] is not None else np.nan for r in validation['per_class']],.36,label='Validation fold 9')
        ax.bar(x+.18,[r[key] if r[key] is not None else np.nan for r in metrics['per_class']],.36,label='Final test fold 10')
        ax.set(xticks=x,xticklabels=CLASS_NAMES,ylabel=key.capitalize(),ylim=(0,1),title='Descriptive comparison only')
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    with (output/'VALIDATION_TEST_CLASS_METRICS.png').open('xb') as target:fig.savefig(target,format='png',dpi=300)
    plt.close(fig)
    return metrics


def write_report(output):
    """Publish after tests, from saved metrics only; never re-enters inference."""
    output=Path(output);ctx=json.loads((output/'report_context.json').read_text())
    pre=json.loads((output/'preflight.json').read_text());timing=json.loads((output/'inference_integrity.json').read_text())
    log=(output/'relevant_tests.log').read_text()
    import re
    summary=next((line for line in reversed(log.splitlines()) if re.search(r'\d+ passed',line)), '')
    if not summary or re.search(r'\d+ failed',log):raise ValueError('Relevant tests must pass first')
    metrics,val=ctx['metrics'],ctx['validation_reference']
    def number(value):return 'nieokreślone' if value is None else f'{value:.6f}'
    per='\n'.join('| '+' | '.join([r['class'],str(r['support']),*[str(r[k]) for k in ('tp','fp','fn','tn')],
        *[number(r[k]) for k in ('precision','recall','specificity','f1','auprc')]])+' |' for r in metrics['per_class'])
    distribution='\n'.join(f"| {r['class']} | {r['records']} | {r['patients']} | {r['percentage']:.2f}% |" for r in ctx['distribution'])
    compare='\n'.join(f"| {k} | {val[k]:.6f} | {metrics[k]:.6f} | {metrics[k]-val[k]:+.6f} |" for k in ('accuracy','macro_f1','macro_auprc'))
    cm='\n'.join('| '+name+' | '+' | '.join(str(v) for v in row)+' |' for name,row in zip(CLASS_NAMES,metrics['confusion_matrix']))
    sources=ctx['false_positive_analysis']['SBRAD_false_positives_by_true_class']
    report=f'''# Phase 19 — końcowa ocena PTB-XL na niewykorzystanym foldzie 10

**Model jest prototypem badawczym, nie klinicznie zwalidowanym urządzeniem diagnostycznym.**
Wykonano jeden pełny przebieg inferencji wybranego wcześniej modelu B na oficjalnym
foldzie 10. Nie użyto wyników testowych do wyboru modelu, epoki, progów ani parametrów.
Nie trenowano ani nie dostrajano modelu. Nie zastosowano progu 0,65 ani abstencji.

## Tożsamość zamrożonego rozwiązania

- Checkpoint: `{pre['checkpoint']}`.
- SHA256: `{pre['checkpoint_sha256']}`; identyczne przed i po inferencji.
- Wybrana epoka 6 konfiguracji B: nieważona CrossEntropyLoss w Phase 18C.
- Istniejąca architektura `ptbxl-rhythm-cnn-v1`, **74116 parametrów**.
- Wejście: float32 `[B,12,1000]`, 12 odprowadzeń, 10 s przy 100 Hz.
- Klasy: **SR=0, AFIB=1, STACH=2, SBRAD=3**; wyjście cztery logity.
- Jedyna reguła: argmax, coverage **100%**, abstencje **0**. Softmax służy do
  wyznaczenia AP/AUPRC, nie jest traktowany jako skalibrowane prawdopodobieństwo kliniczne.

Przed odczytem testowych sygnałów sprawdzono SHA checkpointu, konfigurację modelu,
hash źródeł treningu/architektury, tożsamość eksportu i konfigurację preprocessingu.
Użyto wyłącznie istniejących NPZ z `data/processed/ptbxl_rhythm_v1/`, manifest:
`{pre['manifest']}`. Nie powtarzano filtrowania lub normalizacji.
Eksport zawiera HP 0,5 Hz i LP 45 Hz (Butterworth rzędu 4, zero-phase), bez notch,
Z-score oddzielnie per rekord i odprowadzenie (ddof=0, epsilon=1e-8).
Sygnały wejściowe są bezwymiarowe po normalizacji. Zachowano kolejność
I, II, III, AVR, AVL, AVF, V1–V6. Pełne ustawienia: `preflight.json`.

## Test cohort i separacja pacjentów

Oficjalny fold 10: **1958 ECG, 1711 różnych pacjentów**, rosnąca kolejność ecg_id,
bez podzbiorów, pominięć, resamplingu lub zmian etykiet.
Zbiory pacjentów są rozłączne z treningiem (13749 pacjentów, foldy 1–8) i walidacją
(1763 pacjentów, fold 9). Nie utworzono nowych podziałów. Test zawiera dokładnie
oryginalny fold 10 wybranego wcześniej czteroklasowego cohortu PTB-XL 1.0.3.

| Klasa | ECG | Różni pacjenci klasy | Udział ECG |
|---|---:|---:|---:|
{distribution}

Liczb pacjentów klas nie należy sumować: jeden pacjent może mieć różne etykiety
w różnych zapisach. Oryginalne identyfikatory i fold są w `test_cohort_manifest.json`.
Każdy użyty rekord przeszedł istniejące kontrole SHA256, wymiarów, dtype, skończonych
wartości, konfiguracji i zgodności metadanych z oryginalną metadaną PTB-XL.

## Końcowe wyniki testowe

- **Accuracy: {metrics['accuracy']:.6f} ({100*metrics['accuracy']:.2f}%)**.
- **Macro F1: {metrics['macro_f1']:.6f}**.
- **Macro AUPRC: {metrics['macro_auprc']:.6f}**.
- **Recall AFIB: {metrics['per_class'][1]['recall']:.6f}**.
- **FP SBRAD: {metrics['per_class'][3]['fp']}**.

| Klasa | Support | TP | FP | FN | TN | Precision | Recall | Specificity | F1 | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{per}

Wszystkie metryki dotyczą całych 1958 testowych ECG i pełnego coverage.
One-vs-rest: recall=TP/(TP+FN), specificity=TN/(TN+FP), precision=TP/(TP+FP),
F1=2TP/(2TP+FP+FN). Macro F1 to nieważona średnia czterech F1. Macro AUPRC oznacza
nieważoną średnią nieinterpolowanego average precision one-vs-rest, z grupowaniem
remisów; nie jest polem metodą trapezów. Wartości nieokreślone byłyby JSON null,
nie zastępuje się ich fikcyjnymi zerami. Wyniki zachowano w `test_metrics.json`.

## Confusion matrix i false positives

Wiersze = rzeczywista etykieta, kolumny = argmax modelu.

| Prawdziwa / predykcja | SR | AFIB | STACH | SBRAD |
|---|---:|---:|---:|---:|
{cm}

Fałszywe predykcje SBRAD pochodzą z: **{sources}**.
Precision SBRAD wynosi **{number(metrics['per_class'][3]['precision'])}**, recall
**{number(metrics['per_class'][3]['recall'])}**. Źródłowa etykieta SR nie oznacza
całkowicie zdrowego ECG. Pomyłka AFIB→SBRAD jest błędem typu arytmii, a nie
fałszywym alarmem obecności dowolnej arytmii.
Liczba SR→AFIB/STACH/SBRAD wynosi **{ctx['false_positive_analysis']['SR_to_any_target_arrhythmia_false_alerts']}**.
Przyczyn fizjologicznych błędów nie ustalano na podstawie samych wyników modelu.
Pełny rozkład: `false_positive_sources.csv`; błędy per ECG: `test_errors.csv`
oraz `sbrad_false_positives.csv`, jeśli takie błędy występują.

![Test — macierz liczebności i odsetków wierszy](TEST_CONFUSION_MATRIX.png)

## Opisowe porównanie z walidacją

Walidacja to odrębny fold 9 (1958 ECG / 1763 pacjentów); test to fold 10
(1958 ECG / 1711 pacjentów). Nie jest to porównanie sparowane ani test istotności.
Nie wybierano na podstawie poniższych różnic żadnego nowego modelu lub reguły.

| Metryka | Walidacja 9 | Final test 10 | Test minus walidacja |
|---|---:|---:|---:|
{compare}

Wszystkie porównania per klasa: `validation_test_comparison.csv`.
Różnice dotyczą innych pacjentów i zapisów; nie dowodzą przewagi klinicznej.

![Recall i precision — wyłącznie opisowe porównanie](VALIDATION_TEST_CLASS_METRICS.png)

## Czas inferencji CPU i niezmienność

CPU, **2 wątki**, batch **16**, `model.eval()` i `torch.inference_mode()`.
Wykonano **123 batche**, dokładnie jeden forward każdego rekordu; ostatni batch
ma 6 ECG. Nie wykonywano rozgrzewki lub ponownego forward w celu sprawdzenia reload.

- Suma czasu samego forward: **{timing['forward_total_seconds']:.6f} s**.
- Amortyzowany forward: **{timing['forward_ms_per_record']:.6f} ms/ECG**.
- Pełny przebieg z odczytem/weryfikacją NPZ i zapisem batchy:
  **{timing['inference_wall_seconds_including_loading_and_batch_persistence']:.6f} s**.

Pierwsze wywołanie (cold start) jest uwzględnione; pomiaru nie powtarzano w celu
uzyskania lepszego czasu. Nie jest to opóźnienie pojedynczego żądania aplikacji
ani pomiar całej przyszłej ścieżki od surowego ECG. Pomiary zależą od obciążenia CPU.
Porównanie tensorów `state_dict`, w tym buforów BatchNorm, oraz SHA checkpointu
przed/po potwierdziło niezmienność; gradientów nie tworzono.

## Testy, powtarzalność i blokada ponownej oceny

Wynik odpowiednich testów: **{summary.strip('= ')}**.
Testy nowego evaluatora używają jawnych syntetycznych fixture, nie rzeczywistych
testowych sygnałów. Kontrolują argmax bez progu 0,65, pełne pokrycie, kształty,
niezmienność parametrów, zakaz odczytu niewłaściwych foldów i wyłączność uruchomienia.
Ostrzeżenia WFDB/pandas zapisano w `relevant_tests.log`, nie tłumiono ich.

Trwały znacznik `results/model_evaluation/phase19_final_test/evaluation_claim.json`
blokuje następny pełny przebieg, także do innego katalogu wyników. Nie należy go
usuwać, aby ponownie oceniać lub dobierać model na teście. Batche predykcji
zapisano oddzielnie. Metryki i dokumenty można audytować z zapisanych predykcji,
bez kolejnego wywołania modelu. Przetworzone sygnały i wcześniejsze wyniki są zachowane.
Seed 42, PyTorch {timing['torch']}, NumPy {timing['numpy']}, Python {sys.version.split()[0]},
platforma {platform.platform()}. Plik `inference_integrity.json` zawiera pomiary i audyt.

## Ograniczenia i zakończenie

To jeden held-out fold wyselekcjonowanego czteroklasowego cohortu jednej bazy,
nie prospektywne badanie kliniczne ani niezależny zewnętrzny ośrodek.
Model nie obejmuje wszystkich arytmii, wieloetykietowych zapisów i wszystkich
pacjentów PTB-XL. Etykiety odnoszą się do 10 s, a nie lokalizacji zdarzeń w czasie.
Nierównowaga klas i małe liczebności mniejszości ograniczają interpretację;
rekordy tego samego pacjenta nie są niezależnymi obserwacjami.
Nie wyliczano przedziałów ufności. Model nie ma mechanizmu odrzucania nieznanych klas;
argmax zawsze wybiera jedną z czterech. Prawdopodobieństwa nie były kalibrowane.
Preprocessing offline używa całego rekordu, filtrów zero-phase i normalizacji
per odprowadzenie; nie dowodzi działania przyczynowego w czasie rzeczywistym.

Wielokrotny wybór modeli na foldzie 9 mógł dopasowywać decyzje do walidacji;
fold 10 został użyty raz dopiero po zamrożeniu modelu i reguły.
**Nie wolno wykorzystywać tego testu do dalszego doboru architektury, wag lub progów.**
W tym etapie niczego nie dostrajano po wynikach testowych.
**To prototyp badawczy, nie klinicznie zwalidowane urządzenie diagnostyczne.**
Nie uruchomiono Streamlit, treningu MIT-BIH ani kolejnego etapu.
'''
    with (output/'PHASE19_FINAL_TEST_REPORT.md').open('x') as target:target.write(report)
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','evaluate-once','summarize-saved','report-saved'))
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    if args.stage=='prepare':result=prepare(args.output_dir)
    elif args.stage=='evaluate-once':result=evaluate_once(args.output_dir)
    elif args.stage=='summarize-saved':result=summarize_saved(args.output_dir)
    else:result=write_report(args.output_dir)
    print(json.dumps(result,indent=2),flush=True)
