"""Reproducible local-only scientific figures from validated ECG data and labels.

Existing filters/normalizers are reused for display only. Existing exports,
source recordings and reports are immutable. No downloads or model operations.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data_loader import load_recording,DEFAULT_DATA_DIR
from src.preprocessing_comparison import compare_recording,comparison_statistics
from src.prepare_dataset import load_record_shard,sha256_file
from src.label_mapping import DEFAULT_MAPPING
from src.ptbxl_metadata import load_ptb_metadata
from src.ptbxl_rhythm import (select_pilot,load_rhythm_record,preprocess_rhythm_record,
                             load_pilot_record,preprocessing_config)

PTB_ROOT=Path('data/processed/ptbxl_rhythm_v1')
PTB_RUN=PTB_ROOT/'runs/90b449bf54b649509519747f417b9e5e'
MIT_ROOT=Path('data/processed/ecg_dataset_v1')
STAGE_NAMES=('Raw ECG','Filtered ECG','Filtered + Z-score')
COLORS=('#0072B2','#D55E00','#009E73')
RHYTHM_NAMES={'SR':'Sinus rhythm','AFIB':'Atrial fibrillation',
              'STACH':'Sinus tachycardia','SBRAD':'Sinus bradycardia'}


def write_new_json(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)


def save_figure(fig,stem,output_dir):
    """400-dpi PNG and vector PDF, both exclusive-create; embedded PDF fonts."""
    paths=[]
    for suffix in ('png','pdf'):
        path=Path(output_dir)/(stem+'.'+suffix)
        with path.open('xb') as f:
            fig.savefig(f,format=suffix,dpi=400,facecolor='white')
        paths.append(str(path))
    plt.close(fig)
    return paths


def style_axis(ax):
    ax.grid(axis='both',color='#dce2e6',linewidth=.45)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=9)


def stage_figure(time,arrays,lead_names,title,*,annotations=None,annotation_symbols=()):
    """Identical time coordinates and shared raw/filtered amplitude limits per lead."""
    fig,axes=plt.subplots(len(lead_names),3,squeeze=False,sharex=True,
                          figsize=(14.4,2.25*len(lead_names)+1.3),layout='constrained')
    for channel,name in enumerate(lead_names):
        low=min(float(arrays[0][:,channel].min()),float(arrays[1][:,channel].min()))
        high=max(float(arrays[0][:,channel].max()),float(arrays[1][:,channel].max()))
        pad=max(.05,.12*(high-low))
        for col in range(3):
            ax=axes[channel,col];values=arrays[col][:,channel]
            ax.plot(time,values,color=COLORS[col],linewidth=.85)
            if col<2:ax.set_ylim(low-pad,high+pad)
            ax.set_ylabel(f'{name}\n'+('Amplitude (mV)' if col<2 else 'Z-score (dimensionless)'))
            ax.set_xlabel('Time from recording start (s)')
            if channel==0:ax.set_title(STAGE_NAMES[col],fontsize=11,fontweight='semibold')
            ax.set_xlim(float(time[0]),float(time[-1]+time[1]-time[0]))
            if annotations is not None:
                for index,symbol in zip(annotations,annotation_symbols):
                    ax.scatter([time[index]],[values[index]],s=9,color='#9b2226',zorder=3)
                    ax.annotate(symbol,(time[index],values[index]),xytext=(0,6),textcoords='offset points',
                                ha='center',fontsize=7,color='#9b2226')
            style_axis(ax)
    fig.suptitle(title,fontsize=13,fontweight='semibold')
    return fig


def run(output_dir=Path('results/preprocessing/final_visual_validation'),report_path=Path('FINAL_ECG_VISUAL_VALIDATION.md')):
    output_dir,report_path=Path(output_dir),Path(report_path)
    if output_dir.exists() or report_path.exists():
        raise FileExistsError('Use new output/report paths; existing figures and reports are never overwritten')
    output_dir.mkdir(parents=True)
    # Stat snapshots cover immutable exports/sources; SHA256 verifies selected data against manifests.
    protected=set()
    for folder in ('data/processed','data/ptbxl','data/mitdb','src','tests','examples'):
        protected.update(p for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    protected.update(p for p in Path('.').iterdir() if p.is_file())
    snapshot={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in protected}
    mit_manifest=json.loads((MIT_ROOT/'dataset_metadata.json').read_text())
    mit_rows={row['record_id']:row for row in mit_manifest['shards']}
    ptb_manifest=json.loads((PTB_RUN/'PTBXL_DOWNLOAD_MANIFEST.json').read_text())
    assert ptb_manifest['status']=='complete'
    ptb_rows={row['ecg_id']:row for row in ptb_manifest['records']}
    entries=[];records={};stats=[];beats={};selected={};comparisons={}
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.titlepad':10}):
        for rid in ('100','200'):
            record=load_recording(rid);records[rid]=record
            comparison=compare_recording(record,start_seconds=10,duration_seconds=10)
            comparisons[rid]=comparison
            arrays=[comparison.stages[key] for key in ('raw','filtered','filtered_zscore')]
            indices=comparison.annotation_positions-comparison.start_sample
            fig=stage_figure(comparison.time,arrays,record.channel_names,
                f'MIT-BIH record {rid} | 360 Hz | 10–20 s | expert heartbeat symbols',
                annotations=indices,annotation_symbols=comparison.annotation_symbols)
            paths=save_figure(fig,f'mitbih_{rid}_stages',output_dir)
            caption=f'MIT-BIH {rid}, oba oryginalne kanały, 10–20 s. Czerwone punkty i symbole wskazują niezmienione pozycje eksperckich adnotacji. Raw i filtered mają wspólne granice amplitudy w mV; Z-score jest bezwymiarowy, liczony na całym nagraniu przed wyborem fragmentu.'
            entries.append({'kind':'mit_stages','record_id':rid,'paths':paths,'caption':caption,
                            'channel_names':list(record.channel_names),'sampling_frequency':360,
                            'start_sample':comparison.start_sample,'stop_sample_exclusive':comparison.stop_sample,
                            'annotation_positions':comparison.annotation_positions.tolist(),
                            'annotation_symbols':list(comparison.annotation_symbols),
                            'preprocessing_config':json.loads(str(load_record_shard(MIT_ROOT/mit_rows[rid]['file'])['preprocessing_config']))})
            stats.extend(comparison_statistics(comparison))
        # Copy actual exported windows, not newly detected or synthetic beats.
        for rid in ('100','200','208','102'):
            row=mit_rows[rid];path=MIT_ROOT/row['file']
            assert sha256_file(path)==row['sha256']
            archive=load_record_shard(path)
            record=records.get(rid)
            if record is None:record=load_recording(rid);records[rid]=record
            for code_index,code in enumerate(('N','S','V','F','Q')):
                matches=np.flatnonzero((archive['y']==code_index)&(archive['annotation_positions']>=3600))
                if code in beats or not len(matches):continue
                index=int(matches[0]);position=int(archive['annotation_positions'][index])
                symbol=str(archive['original_symbols'][index]);ai=int(archive['annotation_indices'][index])
                assert record.annotation_positions[ai]==position and record.annotation_symbols[ai]==symbol
                assert DEFAULT_MAPPING[symbol]==code
                config=json.loads(str(archive['preprocessing_config']))
                assert config['before_samples']==90 and config['after_samples']==162
                beats[code]={'X':archive['X'][index].copy(),'record_id':rid,'position':position,
                    'original_symbol':symbol,'annotation_index':ai,'export_row':index,'export_sha256':row['sha256'],
                    'patient_group':str(archive['patient_groups'][index]),'channels':archive['channel_names'].tolist(),
                    'class_index':code_index,'before_samples':90,'sampling_frequency':360}
            if len(beats)==5:break
        assert set(beats)=={'N','S','V','F','Q'}
        fig,axes=plt.subplots(2,5,sharex=True,figsize=(14.8,6.2),layout='constrained')
        for column,code in enumerate(('N','S','V','F','Q')):
            beat=beats[code];time=(np.arange(252)-90)/360
            for channel in range(2):
                ax=axes[channel,column];ax.plot(time,beat['X'][channel],color=COLORS[0],linewidth=1.05)
                ax.axvline(0,color='#9b2226',ls='--',lw=.8)
                ax.scatter([0],[beat['X'][channel,90]],color='#9b2226',s=13,zorder=3)
                ax.set_xlabel('Time relative to annotation (s)')
                ax.set_ylabel(f"{beat['channels'][channel]}\nZ-score (dimensionless)")
                if channel==0:ax.set_title(f"{code} ← {beat['original_symbol']} | record {beat['record_id']}\nsample {beat['position']}",fontsize=10)
                ax.set_xlim(-.25,.45);style_axis(ax)
        fig.suptitle('MIT-BIH | five AAMI-style heartbeat classes | actual exported 252-sample windows',fontsize=13,fontweight='semibold')
        paths=save_figure(fig,'mitbih_aami_heartbeat_examples',output_dir)
        beat_meta={code:{k:v for k,v in beat.items() if k!='X'} for code,beat in beats.items()}
        entries.append({'kind':'mit_beats','paths':paths,'examples':beat_meta,
            'caption':'Po jednym rzeczywistym wyeksportowanym uderzeniu N/S/V/F/Q, oba kanały w oryginalnej kolejności. Linia t=0 oznacza pozycję adnotacji, nie automatycznie wyznaczony nowy R-peak. Q pokazuje konkretny zachowany symbol, nie całą heterogeniczną klasę. Okna [p−90,p+162) mają 252 próbki przy 360 Hz.'})
        metadata=load_ptb_metadata()
        for item in select_pilot(metadata,per_class=1):
            eid=item['ecg_id'];code=item['label'];row=ptb_rows[eid]
            path=PTB_ROOT/row['file'];assert sha256_file(path)==row['sha256']
            archive=load_pilot_record(path)
            loaded=load_rhythm_record(metadata,eid)
            assert loaded.source_hashes==row['source_hashes']
            assert json.loads(str(archive['original_metadata']))==item
            processed=preprocess_rhythm_record(loaded)
            np.testing.assert_array_equal(processed.X,archive['X'])
            assert int(archive['y'])==item['y']
            leads=('II','V1','V5');lead_indices=[loaded.channel_names.index(name) for name in leads]
            time=np.arange(1000)/100
            arrays=[loaded.raw[:,lead_indices],processed.filtered[:,lead_indices],archive['X'][lead_indices].T]
            fig=stage_figure(time,arrays,leads,
                f'PTB-XL {eid:05d} | {code}: {RHYTHM_NAMES[code]} | 100 Hz | full 10 s | metadata label')
            paths=save_figure(fig,f'ptbxl_{code}_{eid:05d}_stages',output_dir)
            entries.append({'kind':'ptb_stages','ecg_id':eid,'rhythm_code':code,'class_index':item['y'],
                'patient_id':item['patient_id'],'strat_fold':item['strat_fold'],'paths':paths,
                'original_scp_statements':item['original_scp_statements'],'displayed_leads':list(leads),
                'export_sha256':row['sha256'],'source_hashes':loaded.source_hashes,
                'caption':f'PTB-XL ecg_id={eid}, {code} ({RHYTHM_NAMES[code]}), odprowadzenia II/V1/V5. Etykieta wynika z jedynego kodu rytmu spośród wszystkich 12, nie z oceny wykresu. Całe 10 s, filtrowanie 0,5–45 Hz bez notch i Z-score per rekord/per lead; wynik identyczny z gotowym eksportem.'})
            selected[code]={'identity':item,'X':archive['X'],'channels':loaded.channel_names}
        fig,axes=plt.subplots(4,2,sharex=True,figsize=(14.4,9.3),layout='constrained')
        for row,code in enumerate(('SR','AFIB','STACH','SBRAD')):
            example=selected[code]
            for col,name in enumerate(('II','V1')):
                ax=axes[row,col];channel=example['channels'].index(name)
                ax.plot(np.arange(1000)/100,example['X'][channel],color=COLORS[0],linewidth=.85)
                ax.set_title(f"{code}: {RHYTHM_NAMES[code]} | ECG {example['identity']['ecg_id']:05d} | {name}",fontsize=10)
                ax.set_ylabel('Z-score (dimensionless)');ax.set_xlabel('Time from recording start (s)')
                ax.set_xlim(0,10);style_axis(ax)
        fig.suptitle('PTB-XL | four record-level rhythm labels | exported normalized ECG | 100 Hz',fontsize=13,fontweight='semibold')
        paths=save_figure(fig,'ptbxl_four_rhythm_examples',output_dir)
        entries.append({'kind':'ptb_overview','paths':paths,
            'examples':{code:example['identity'] for code,example in selected.items()},
            'caption':'Cztery etykiety rytmu PTB-XL, po jednym rzeczywistym 10-sekundowym rekordzie, odprowadzenia II i V1. Wykresy pokazują dokładne tablice gotowego eksportu po filtracji i Z-score. Nie jest to diagnoza, estymacja jakości klasyfikacji ani dowód reprezentatywności populacji.'})
    # Data-source provenance includes exact SHA256 for every source used.
    source_hashes={}
    mit_sources=mit_manifest['source_files_sha256']
    for rid in records:
        for suffix in ('.hea','.dat','.atr'):
            name=rid+suffix;path=DEFAULT_DATA_DIR/name
            digest=sha256_file(path);assert digest==mit_sources[name]
            source_hashes[str(path)]=digest
    for code,example in selected.items():
        item=example['identity'];base=Path('data/ptbxl')/item['filename_lr']
        for suffix in ('.hea','.dat'):
            path=base.with_suffix(suffix);source_hashes[str(path)]=sha256_file(path)
    assert all(p.is_file() and (p.stat().st_size,p.stat().st_mtime_ns)==snapshot[str(p)] for p in protected)
    import csv
    with (output_dir/'mitbih_display_statistics.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)
    result={'status':'passed','selection_rule':{'MIT_stage_records':['100','200'],'MIT_stage_interval_seconds':[10,20],
        'MIT_beats':'First exported beat at sample >=3600 per class, search record order 100,200,208,102; no morphology/quality optimization',
        'PTB':'First eligible ecg_id per class via existing Phase-13 selection, exactly one of all 12 rhythm codes'},
        'figures':entries,'figure_count':len(entries),'files_per_figure':['400-dpi PNG','vector PDF'],
        'ptb_configuration':preprocessing_config(),'sources_sha256':source_hashes,
        'prior_files_preserved':len(protected),'preservation':'size and mtime unchanged; selected exports/sources SHA256 match existing manifests',
        'code_sha256':{str(p):sha256_file(p) for p in [Path(__file__),Path('src/preprocessing.py'),Path('src/preprocessing_comparison.py'),Path('src/ptbxl_rhythm.py')]}}
    write_new_json(output_dir/'visual_validation_manifest.json',result)
    report=['# Finalna wizualna walidacja preprocessingu ECG\n',
        'Wyłącznie rzeczywiste, lokalne MIT-BIH i PTB-XL. Nowe figury demonstracyjne; nie zmieniono danych, metod preprocessingu ani wcześniejszych raportów. Nie pobierano danych i nie trenowano modeli.\n',
        '## Metoda i wybór przykładów\n',
        'MIT-BIH: pełne nagranie przetworzono istniejącym compare_recording, potem wycięto 10–20 s. Z-score jest liczony per nagranie i kanał, dlatego 10-sekundowy fragment nie musi mieć średniej 0 i std 1. Uderzenia skopiowano bez zmian z wyeksportowanych NPZ; wybrano pierwsze kompletne okno po 10 s dla każdej klasy według kolejności rekordów 100,200,208,102. Każdy symbol i pozycję sprawdzono w oryginalnym .atr i istniejącym mapowaniu AAMI.\n',
        'PTB-XL: pierwszy kwalifikujący się ecg_id każdej klasy, dokładnie jeden kod spośród wszystkich 12 rozpoznanych kodów rytmu. Nie wybierano rekordu według wyglądu ani likelihood. NaN/brak kodu nie oznacza SR; zero score może oznaczać nieznaną pewność. Odczyt przez WFDB, walidowane filtrowanie i normalizacja; przeliczony wynik float32 jest identyczny z gotowym eksportem. Oryginalne SCP, ID i foldy są w manifeście.\n',
        'Filtry: istniejące Butterworth SOS 0,5 Hz high-pass / 45 Hz low-pass, rząd 4, zero-phase offline. Notch wyłączony; 50/60 Hz jest na lub ponad Nyquistem przy 100 Hz. Normalizacja Z-score niezależna per rekord/per lead; bez wspólnych parametrów z innych pacjentów.\n',
        'Reguła wyboru jest deterministyczna i służy ilustracji. Pojedynczy przykład nie jest statystycznie reprezentatywną próbką klasy, oceną skuteczności ani potwierdzeniem rozpoznania klinicznego.\n',
        '## Figury i podpisy\n']
    for number,entry in enumerate(entries,1):
        report += [f"### Rysunek {number}\n",entry['caption']+'\n',f"![Rysunek {number}: {entry['kind']}]({entry['paths'][0]})\n",f"[Wektorowy PDF]({entry['paths'][1]})\n"]
    report += ['## Wizualna walidacja i ograniczenia\n',
        'W każdej parze raw/filtered użyto wspólnych granic osi amplitudy i tej samej osi czasu. Redukcja składowych wolnozmiennych i wysokoczęstotliwościowych może zmieniać amplitudę oraz kształt przebiegu. Z-score zmienia skalę i offset; nie jest dodatkowym filtrem. Dla pełnych 10 s PTB-XL statystyki są per lead; dla fragmentu MIT-BIH wynik odnosi się do całego nagrania.\n',
        'Markery MIT-BIH i początek względnej osi okna pozostają dokładnie w pozycjach eksperckich. Nie szukano nowych R-peaków ani nie przesuwano sygnału. Zero-phase nie dodaje stałego opóźnienia filtru, ale nie gwarantuje, że wszystkie lokalne ekstrema surowego przebiegu zachowają dokładnie ten sam kształt lub indeks po filtracji.\n',
        'Q jest heterogeniczną grupą AAMI; jeśli wybrany symbol to /, jest to uderzenie stymulowane, nie przykład każdego rodzaju unknown. N nie oznacza zdrowego pacjenta. F to grupa fusion beat, nie migotanie przedsionków. AFIB/STACH/SBRAD są etykietami całego rekordu PTB-XL i nie zastępują klas uderzeń. Oznaczenia są potwierdzone metadanymi/adnotacjami, nie rozpoznawane na podstawie wizualnej morfologii.\n',
        'Filtracja offline może mieć efekty brzegowe, usuwać informacje o amplitudzie i subtelnej morfologii; wizualna kontrola nie dowodzi zachowania wszystkich P/QRS/T ani przydatności klinicznej. Sygnały 100 Hz nie zawierają całej informacji dostępnej przy 500 Hz. Nie wykonano delineacji fal ani oceny medycznej.\n',
        '## Odtwarzanie i artefakty\n',
        'PNG: 400 dpi; dodatkowo PDF z osadzonymi fontami i wektorowymi przebiegami. Osie i jednostki w języku angielskim do wykorzystania w publikacji. Trzy wybrane leads PTB-XL zamiast wszystkich 12; MIT-BIH pokazuje oba oryginalne kanały, bez zamiany nazw.\n',
        'Pliki: results/preprocessing/final_visual_validation/. Manifest zawiera wybrane beat indices, symbole, pozycje, ecg_id/patient_id/fold, konfiguracje, SHA256 i podpisy. CSV opisuje dokładnie pokazane fragmenty MIT-BIH, nie globalne statystyki populacji.\n',
        'Uruchomienie z katalogu projektu: `.venv/bin/python -m examples.final_ecg_visual_validation`. Skrypt odmawia nadpisania istniejącego raportu/katalogu; kolejne wykonanie wymaga nowych ścieżek przez --output-dir i --report-path.\n']
    with report_path.open('x') as f:f.write('\n'.join(report))
    print(json.dumps({'figures':len(entries),'beats':beat_meta,'ptb_ids':{c:x['identity']['ecg_id'] for c,x in selected.items()}},indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('results/preprocessing/final_visual_validation'))
    parser.add_argument('--report-path',type=Path,default=Path('FINAL_ECG_VISUAL_VALIDATION.md'))
    args=parser.parse_args()
    run(args.output_dir,args.report_path)
