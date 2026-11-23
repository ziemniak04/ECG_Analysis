"""Summarize one completed Phase-22 run; never evaluates final-test data."""
import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import numpy as np
from src.mitbih_full_run import summarize_run, validated_history
from src.prepare_dataset import sha256_file
from src.ptbxl_training import write_json_new


def integrity(output, audit):
    before=json.loads((audit/'preservation_before.json').read_text())
    # This new run's console existed at preflight time and intentionally grows.
    excluded=[p for p in before if Path(p).is_relative_to(output.parent)]
    protected={p:d for p,d in before.items() if p not in excluded}
    changed=[p for p,d in protected.items() if not Path(p).is_file() or sha256_file(p)!=d]
    stats=json.loads((audit/'dataset_stats_before.json').read_text())
    changed_data=[p for p,s in stats.items() if not Path(p).is_file() or [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=s]
    current={str(p) for root in ('data/processed/ecg_dataset_v1','data/processed/ptbxl_rhythm_v1')
             for p in Path(root).rglob('*') if p.is_file()}
    if changed or changed_data or current!=set(stats):
        raise ValueError(f'Prior work changed: {changed}; data: {changed_data}')
    execution=json.loads((output/'execution.json').read_text())
    plan=json.loads((output/'training_plan.json').read_text())
    allowed=set(plan['summary']['train']['record_ids']+plan['summary']['validation']['record_ids'])
    if not set(execution['read_records_including_preflight'])<=allowed or execution['final_test_read']:
        raise ValueError('Unexpected final-test access')
    checks={str(p.relative_to(output)):sha256_file(p) for p in output.glob('*.pt')}
    for p,digest in checks.items():
        if (output/p).with_suffix('.sha256').read_text().strip()!=digest:
            raise ValueError('Checkpoint checksum mismatch')
    result=dict(status='passed',prior_files_sha256_verified=len(protected),changed_prior_files=[],
        mutable_new_run_paths_excluded=excluded,dataset_files_stat_verified=len(stats),
        changed_dataset_files=[],final_test_read=False,ptbxl_reevaluation=False,
        unique_train_validation_records_read=sorted(set(execution['read_records_including_preflight'])),
        checkpoint_sha256=checks)
    write_json_new(output/'analysis/final_integrity_check.json',result)
    return result


def main(output,audit):
    output=Path(output);audit=Path(audit)
    result,best=summarize_run(output)
    result,plan,history,best=validated_history(output)
    check=integrity(output,audit)
    for name in ('preflight.json','reproducibility.json'):
        with (output/name).open('xb') as target:target.write((audit/name).read_bytes())
    console=output.parent/'run_001_console.log'
    if console.exists():
        with (output/'console.log').open('xb') as target:target.write(console.read_bytes())
    execution=json.loads((output/'execution.json').read_text())
    log=(output/'post_training_pytest.log').read_text()
    match=re.search(r'(\d+) passed, (\d+) warnings in ([\d.]+)s',log)
    if not match:raise ValueError('Successful complete post-training tests required before final report')
    m=best['validation'];tr=best['train'];f=m['per_class'][3];s=m['per_class'][1];q=m['per_class'][4]
    cm=np.asarray(m['confusion_matrix'])
    pct=lambda v:'nieokreślone' if v is None else f'{100*v:.2f}%'
    metric_table='\n'.join(f"| {r['class_name']} | {r['support']} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['specificity'])} | {r['f1']:.6f} | {r['tp']} | {r['fp']} | {r['fn']} |" for r in m['per_class'])
    cm_table='\n'.join('| '+c+' | '+' | '.join(str(int(v)) for v in cm[i])+' |' for i,c in enumerate('NSVFQ'))
    epoch_table='\n'.join(f"| {r['epoch']} | {r['train']['loss']:.6f} | {r['validation']['loss']:.6f} | {pct(r['validation']['accuracy'])} | {r['validation']['macro_f1']:.6f} | {r['validation']['macro_auprc']:.6f} | {r['seconds']:.2f} |" for r in history)
    group_rows=[]
    for g,gm in m['per_group'].items():
        for row in gm['per_class']:
            if row['class_name'] in ('S','F','Q'):
                group_rows.append(f"| {g} | {row['class_name']} | {row['support']} | {row['tp']} | {row['fp']} | {pct(row['recall'])} | {pct(row['precision'])} |")
    concentration=[]
    for label in (1,3,4):
        ranked=sorted(((gm['per_class'][label]['fn'],g) for g,gm in m['per_group'].items()),reverse=True)
        concentration.append('FN '+ 'NSVFQ'[label]+': '+', '.join(f'{g}={n}' for n,g in ranked[:3] if n))
    f_fp_groups=sorted(((gm['per_class'][3]['fp'],g) for g,gm in m['per_group'].items()),reverse=True)
    concentration.append('Najwięcej FP F: '+', '.join(f'{g}={n}' for n,g in f_fp_groups[:3] if n))
    errors=sorted(((int(cm[i,j]),'NSVFQ'[i],'NSVFQ'[j]) for i in range(5) for j in range(5) if i!=j and cm[i,j]),reverse=True)
    error_text='; '.join(f'{a} → {b}: {n}' for n,a,b in errors[:6])
    first,last=history[0],history[-1]
    f_assessment=(f"F pozostaje problematyczna: TP={f['tp']}/38, recall {pct(f['recall'])}, precision {pct(f['precision'])}. "
        'Przy tak niskiej czułości i precyzji klasa F nie została wiarygodnie rozpoznana. Dodatkowym ograniczeniem jest tylko 38 walidacyjnych uderzeń i cztery grupy z F. '
        'Nie można na tej podstawie uznać klasy za niezawodnie rozpoznawaną.')
    total=execution['wall_seconds'];sum_epochs=sum(r['seconds'] for r in history)
    report=f'''# Faza 22 — pierwszy pełny trening CNN MIT-BIH

Status: **{result['status']}**, {result['completed_epochs']} ukończonych epok. Wykonano wyłącznie jeden zatwierdzony eksperyment. Nie oceniano końcowego testu MIT-BIH i nie zmieniono zakończonego modelu ani wyników PTB-XL. Wyniki poniżej są **walidacyjne**, nie testowe ani kliniczne.

## Eksperyment i dane

Katalog: `{output}`. Dokładna konfiguracja pochodzi z niezmienionego `PHASE21_MITBIH_TRAINING_CONFIG.json`, kopia w `configuration.json`. Adam, lr 0.001, batch 32, CPU 2 wątki, seed 42, max 15 epok, patience 4, min_delta 0. Architektura fazy 20: 17797 parametrów, wejście float32 `[B,2,252]`, wyjście `[B,5]` logitów; CE otrzymuje logity bez Softmax.

Klasy N=0, S=1, V=2, F=3, Q=4 są grupami uderzeń AAMI-style, nie pięcioma nazwanymi diagnozami rytmu. Użyto istniejących segmentów bez ponownej filtracji/normalizacji i bez zmian etykiet/odprowadzeń.

Trening: **62926 uderzeń / 29 grup** (N52254, S1715, V4389, F401, Q4167). Walidacja: **23134 / 9 grup** (N19094, S491, V1444, F38, Q2067). Zamrożony test: 23392 / 9 grup — wyłącznie liczby z manifestu; jego plików NPZ nie otwierano w treningu ani w analizie. 201 i 202 mają wspólną grupę, razem w walidacji. Grupy wszystkich trzech części są rozłączne.

Zastosowano pierwiastki treningowych wag balanced, znormalizowane do średniej 1: N {plan['class_weights'][0]:.6f}, S {plan['class_weights'][1]:.6f}, V {plan['class_weights'][2]:.6f}, F {plan['class_weights'][3]:.6f}, Q {plan['class_weights'][4]:.6f}. Sprawdzono je niezależnie ze statystyk wyłącznie treningu. Nie wykonywano oversamplingu/undersamplingu, augmentacji ani wyboru wag na teście.

## Czas i wybór checkpointu

- Ukończone epoki: **{result['completed_epochs']}**; status early stopping: **{'tak' if result['status']=='early_stopped' else 'nie'}**.
- Całkowity czas `train_model`: **{total:.2f} s ({total/60:.2f} min)**, łącznie z loading/metrykami/zapisem; bez kontroli preflight.
- Suma zmierzonych przebiegów epok: {sum_epochs:.2f} s, średnio {sum_epochs/len(history):.2f} s/epokę. Czasy epok obejmują przejścia i metryki, a zapis checkpointu następuje po tym pomiarze.
- Najlepsza epoka: **{best['epoch']}**, wybrana przez najwyższy pięcioklasowy Macro F1 na **kompletnych 23134 przykładach walidacyjnych**. Przy remisie wcześniejsza epoka.
- Każda epoka: pełne 62926 treningowych i 23134 walidacyjne, brak duplikatów/brakujących indeksów; 1967/723 batchy.

## Wyniki najlepszego checkpointu — wyłącznie walidacja

- **Macro F1: {m['macro_f1']:.6f}**.
- **Macro AUPRC / average precision: {m['macro_auprc']:.6f}**.
- **Accuracy: {pct(m['accuracy'])}**. Dla porównania, stałe przewidywanie N dałoby {pct(plan['summary']['validation']['class_counts']['N']/m['beat_count'])} na tym samym rozkładzie — to opisowy punkt odniesienia, nie dodatkowy trenowany model.
- Training loss w wybranej epoce: {tr['loss']:.6f}; validation loss: {m['loss']:.6f}.
- CPU forward validation: {m['forward_seconds']:.3f} s, {m['inference_ms_per_beat']:.4f} ms/uderzenie; bez loading/loss/Softmax/metryk. To przepustowość batchy 32, nie opóźnienie pojedynczego badania.

| Klasa | Support | Precision | Recall | Specificity | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{metric_table}

### Macierz pomyłek

Wiersze = rzeczywista klasa, kolumny = argmax. Bez progów i abstencji.

| True / pred | N | S | V | F | Q |
|---|---:|---:|---:|---:|---:|
{cm_table}

Największe kierunki pomyłek: **{error_text}**. Interpretować je łącznie z nierównowagą: N stanowi większość, a niewielka liczba błędów rzadkiej klasy może znacznie zmienić jej recall/precision. S: TP {s['tp']}/{s['support']}, FP {s['fp']}; Q: TP {q['tp']}/{q['support']}, FP {q['fp']}.

**{f_assessment}**

## Przebieg strat i metryk

| Epoka | Train loss | Validation loss | Val accuracy | Val Macro F1 | Val Macro AP | Czas s |
|---|---:|---:|---:|---:|---:|---:|
{epoch_table}

Strata treningowa zmieniła się od {first['train']['loss']:.6f} do {last['train']['loss']:.6f}; walidacyjna od {first['validation']['loss']:.6f} do {last['validation']['loss']:.6f}. Najniższa validation loss wystąpiła w epoce {min(history,key=lambda r:r['validation']['loss'])['epoch']}; model wybierano jednak według Macro F1, nie minimum straty. Training Macro F1 online wzrósł od {first['train']['macro_f1']:.6f} do {last['train']['macro_f1']:.6f}, bez analogicznej poprawy walidacji. Jest to zgodne z problemem generalizacji/przeuczenia, nie dowodzi jednej konkretnej przyczyny. Wahania walidacyjne mogą odzwierciedlać nierównowagę, różnice pacjentów i przeuczenie; sama krzywa nie identyfikuje przyczyny. Training metrics są online, przed kolejnymi aktualizacjami z dropout/BN, więc nie są oceną jednego zamrożonego modelu.

## Pacjenci/grupy i klasy rzadkie

Trening F: 372/401 uderzeń pochodzi z 208. Walidacja F ma cztery grupy, łącznie tylko 38 przykładów. Q w treningu jest zdominowana przez 102/107, a 2062 z 2067 walidacyjnych Q pochodzi z 104. Liczebność uderzeń nie równa się liczbie niezależnych pacjentów. Nie wykonano dodatkowego doboru podziału ani rezygnacji z trudnych pacjentów.

Poniżej metryki S/F/Q według grup. Brak prawdziwych przykładów klasy oznacza nieokreślony recall, nie zero; false positives pozostają policzone.

| Grupa | Klasa | Support | TP | FP | Recall | Precision |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(group_rows)}

Koncentracja błędów: {'; '.join(concentration)}. To lokalizacja pomyłek, nie dowód ich klinicznej lub technicznej przyczyny.

Pełne macierze i metryki wszystkich klas/grup znajdują się w `analysis/patient_group_validation_metrics.json` oraz CSV. Nie podano beat-level przedziałów ufności sugerujących niezależność uderzeń.

## Weryfikacja i odtwarzalność

- Przed startem sprawdzono kształty, skończoność, oryginalne symbole/mapowanie, SHA256 wszystkich **39 treningowych/walidacyjnych** shardów; testowe NPZ nie były dekodowane ani oceniane.
- Dokładne populacje i sumy macierzy wszystkich epok oraz wybór najlepszego modelu zgadzają się z historią. Suma macierzy grup zgadza się z macierzą całej walidacji.
- Wszystkie checkpointy mają poprawne SHA256; najlepszy stan jest identyczny ze stanem best zapisanym w ostatnim checkpoincie. Dwa niezależnie odtworzone modele zwróciły dokładnie identyczne logity na tym samym małym batchu **treningowym** `[8,2,252]`; bez ponownej oceny testu ani dodatkowych aktualizacji.
- Źródła/model/preprocessing/dane i wcześniejsze wyniki: brak zmian. Zweryfikowano SHA256 {check['prior_files_sha256_verified']} wcześniejszych plików oraz niezmienność inwentarza/rozmiarów/mtime {check['dataset_files_stat_verified']} plików obu eksportów. Nowy, rosnący console.log tego runu jest jawnie wyłączony z zakresu wcześniejszych artefaktów.
- Seed 42, deterministyczne algorytmy, shuffle seed+epoch, num_workers=0; dokładne wersje zależności i platforma w `reproducibility.json`. Nie powtarzano eksperymentu, aby oszacować zmienność między seedami.
- Testy po treningu: **{match.group(1)} passed, {match.group(2)} warnings, {match.group(3)} s**. Ostrzeżenia pandas/WFDB pozostawiono widoczne. Testy mechaniki treningu używają małych fixture; historyczne QC nie oznacza oceny nowego modelu na końcowym teście.

## Pliki i bezpieczne wznowienie

- Najlepszy model: `{output/'best_model.pt'}`, SHA256 `{sha256_file(output/'best_model.pt')}`.
- Ostatni pełny checkpoint: `{result['last_checkpoint']}`, obok `.sha256`; zawiera model, najlepszy stan, Adam, RNG i historię.
- Konfiguracja, plan, preflight, dependencies, console/training logs oraz `epoch_XXX_metrics.json` znajdują się w katalogu eksperymentu.
- `analysis/training_history.csv`, `analysis/training_history.json`, `analysis/per_class_validation_metrics.csv`, `analysis/patient_group_class_metrics.csv`, `analysis/confusion_matrix.csv`.
- Wykresy 300 dpi: `analysis/loss_curves.png`, `analysis/validation_macro_f1.png`, `analysis/validation_macro_auprc.png`, `analysis/validation_confusion_matrix.png`, `analysis/validation_confusion_normalized.png`.
- `analysis/checkpoint_reload_check.json`, `analysis/final_integrity_check.json`, `post_training_pytest.log`, sumy kontrolne.

Zapis epok jest nieprzepisywalny. Wznowienie jest możliwe wyłącznie z kompletnego poprawnego checkpointu nieukończonego runu, z identycznym planem/kodem/runtime, do nowego katalogu; częściowa epoka jest odtwarzana. **Obecny zakończony run nie jest automatycznie wznawiany ani przedłużany poza zatwierdzone 15 epok/early stopping.**

## Rekomendowany następny eksperyment — bez uruchamiania

Najbardziej czytelny kolejny kontrolowany punkt odniesienia to **unweighted CE** przy tym samym modelu, seedzie, pacjentach, optymalizatorze i limicie epok, po osobnym zatwierdzeniu. Porównać na tej samej kompletnej walidacji Macro F1/AP, S/F/Q recall i precision oraz błędy N, zamiast zakładać skuteczność wag. Wagi nie naprawią koncentracji F/Q w pojedynczych osobach. Analizę per-group i morfologii błędów można prowadzić bez końcowego testu; nie zmieniać mapowania/wyłączać trudnych grup w reakcji na wynik.

To propozycja, nie twierdzenie o poprawie i nie automatyczny kolejny run. Powtarzana selekcja na dziewięciu grupach walidacyjnych grozi przeuczeniem walidacji; ograniczyć liczbę porównań i z góry ustalić kryteria. Końcowy test pozostaje zamknięty do osobnej zgody na ocenę ostatecznie zamrożonego modelu. Projekt jest prototypem badawczym, nie klinicznie zwalidowanym urządzeniem diagnostycznym. **Koniec fazy 22.**
'''
    with Path('PHASE22_MITBIH_FULL_TRAINING_REPORT.md').open('x') as target:target.write(report)
    print(json.dumps(dict(status='reported',epochs=len(history),best_epoch=best['epoch'],
        accuracy=m['accuracy'],macro_f1=m['macro_f1'],macro_auprc=m['macro_auprc'],seconds=total,
        checkpoint=str(output/'best_model.pt'),per_class=m['per_class']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--audit-dir',type=Path,required=True)
    args=parser.parse_args();main(args.output_dir,args.audit_dir)
