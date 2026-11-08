"""Separate PTB-XL 100 Hz rhythm adapter and bounded pilot; never MIT beat windows."""
from dataclasses import dataclass
from pathlib import Path
from numbers import Integral
import json
import re
from urllib.request import urlopen
import numpy as np
import wfdb
from .ptbxl_metadata import CORE_TARGETS,QUALITY_COLUMNS,propose_exclusive_cohort
from .preprocessing import filter_ecg, ECGNormalizer
from .prepare_dataset import sha256_file

CLASS_NAMES = CORE_TARGETS
CLASS_INDICES = {name:i for i,name in enumerate(CLASS_NAMES)}
EXPECTED_LEADS = ('I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6')
OFFICIAL_BASE_URL = 'https://physionet.org/files/ptb-xl/1.0.3/'
DEFAULT_WAVEFORM_ROOT = Path(__file__).resolve().parents[1]/'data/ptbxl'
DEFAULT_CONFIG = {
    'version':'ptbxl-rhythm-pilot-v1','dataset_version':'1.0.3','sampling_frequency':100.0,
    'duration_seconds':10.0,'source_shape':[1000,12],'output_shape':[12,1000],
    'output_dtype':'float32','label_dtype':'int64','class_names':list(CLASS_NAMES),
    'class_indices':CLASS_INDICES,'expected_leads':list(EXPECTED_LEADS),
    'lead_policy':'Require original header order exactly; never relabel or reorder silently',
    'source_units':'mV','output_units':'dimensionless',
    'filtering':{'highpass_hz':0.5,'lowpass_hz':45.0,'order':4,'notch_hz':None,'quality_factor':30.0},
    'filter_mode':'SOS forward/backward zero-phase, offline',
    'normalization':{'method':'zscore','scope':'recording_per_lead','epsilon':1e-8,'ddof':0},
    'cohort_rule':'Exactly one of all 12 original rhythm codes and that code is SR/AFIB/STACH/SBRAD',
    'nyquist_hz':50.0,'notch_policy':'Disabled; 50/60 Hz invalid at 100 Hz',
    'selection_rule':'First two eligible ecg_id values ascending per class, class order SR/AFIB/STACH/SBRAD',
    'no_segmentation':True,'no_resampling':True,'no_splits':True,'no_balancing':True}


def preprocessing_config():
    """Return a detached reproducible pilot configuration."""
    return json.loads(json.dumps(DEFAULT_CONFIG))


def record_identity(data,ecg_id):
    """Resolve the source metadata row and independently recheck exclusive rhythm eligibility."""
    if isinstance(ecg_id,bool) or not isinstance(ecg_id,Integral):
        raise ValueError('ecg_id must be an integer')
    rows=np.flatnonzero(data.ecg_ids==ecg_id)
    if len(rows)!=1:
        raise ValueError(f'Unknown/nonunique ECG identity {ecg_id}')
    index=int(rows[0])
    codes=sorted(set(data.statements[index])&set(data.rhythm_codes))
    if len(data.rhythm_codes)!=12 or len(codes)!=1 or codes[0] not in CLASS_NAMES:
        raise ValueError('Record is not in the Phase-12 exclusive four-class cohort')
    row=data.records.iloc[index]
    path=row['filename_lr']
    match=re.fullmatch(r'records100/(\d{5})/(\d{5})_lr',path) if isinstance(path,str) else None
    if match is None or int(match[2])!=ecg_id:
        raise ValueError('filename_lr does not match ECG identity')
    def plain(value):
        import pandas as pd
        if pd.isna(value):
            return None
        return value.item() if isinstance(value,np.generic) else value
    return {'ecg_id':int(ecg_id),'patient_id':int(data.patient_ids[index]),'strat_fold':int(data.folds[index]),
        'label':codes[0],'y':CLASS_INDICES[codes[0]],'filename_lr':path,'filename_hr':row['filename_hr'],
        'original_scp_codes_cell':row['scp_codes'],'original_scp_statements':dict(data.statements[index]),
        'quality_metadata':{name:plain(row[name]) for name in QUALITY_COLUMNS if name in row}}


def select_pilot(data,per_class=2):
    """Select at most 8 real metadata records. No fallback/replacement by signal quality."""
    if isinstance(per_class,bool) or not isinstance(per_class,Integral) or per_class not in (1,2):
        raise ValueError('Small pilot only: per_class must be 1 or 2')
    proposal=propose_exclusive_cohort(data)
    result=[]
    for code in CLASS_NAMES:
        rows=sorted((r for r in proposal['record_audit'] if r['proposed_label']==code),key=lambda r:r['ecg_id'])
        if len(rows)<per_class:
            raise ValueError(f'Not enough eligible actual metadata rows for {code}')
        result.extend(record_identity(data,r['ecg_id']) for r in rows[:per_class])
    return result


def pilot_files(selection):
    """Enumerate ONLY selected official records100 .hea/.dat paths."""
    if not 1<=len(selection)<=8 or len({r['ecg_id'] for r in selection})!=len(selection):
        raise ValueError('Need a unique bounded pilot selection (at most 8 ECGs)')
    result=[]
    for row in selection:
        base=row['filename_lr']
        if re.fullmatch(r'records100/\d{5}/\d{5}_lr',base) is None:
            raise ValueError('Invalid low-rate path')
        for suffix in ('.hea','.dat'):
            result.append({'ecg_id':row['ecg_id'],'relative_path':base+suffix,'url':OFFICIAL_BASE_URL+base+suffix})
    return result


def download_pilot(selection,waveform_root=DEFAULT_WAVEFORM_ROOT,*,timeout=12):
    """Bounded official downloads, exclusive writes, no metadata/waveform overwrites.

    Existing files are left alone for adapter validation. Bytes are buffered only
    for one file, capped at 1 MiB (pilot records are expected to be far smaller).
    Network read failures create no destination file; all errors are reported.
    Files left by a local write failure must be reviewed, never auto-replaced.
    """
    root=Path(waveform_root)
    results=[]
    for item in pilot_files(selection):
        path=root/item['relative_path']
        status={**item}
        if path.exists():
            status.update({'status':'already_present','bytes':path.stat().st_size,'sha256':sha256_file(path)})
        else:
            try:
                with urlopen(item['url'],timeout=timeout) as response:
                    payload=response.read(1024*1024+1)
                if not payload or len(payload)>1024*1024:
                    raise ValueError('Empty or unexpectedly large pilot file')
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as output:
                    output.write(payload)
                status.update({'status':'downloaded','bytes':len(payload),'sha256':sha256_file(path)})
            except Exception as exc:
                status.update({'status':'download_failed','error':f'{type(exc).__name__}: {exc}'})
        results.append(status)
    return results


@dataclass(frozen=True)
class LoadedRhythmRecord:
    identity: dict
    raw: np.ndarray
    channel_names: tuple[str,...]
    sampling_frequency: float
    source_hashes: dict


@dataclass(frozen=True)
class ProcessedRhythmRecord:
    loaded: LoadedRhythmRecord
    filtered: np.ndarray
    X: np.ndarray
    y: np.int64
    configuration: dict


def load_rhythm_record(data,ecg_id,waveform_root=DEFAULT_WAVEFORM_ROOT):
    """Read a real WFDB record with strict identity/header/size/checksum checks.

    Patient ID/fold are verified against the metadata lookup; PTB headers do not
    independently identify the patient, so no separate biometric verification is claimed.
    Expected official low-rate format: one shared 16-bit .dat, 1000x12, in mV.
    """
    identity=record_identity(data,ecg_id)
    base=Path(waveform_root)/identity['filename_lr']
    paths=[base.with_suffix('.hea'),base.with_suffix('.dat')]
    missing=[str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing pilot files: '+', '.join(missing))
    try:
        header=wfdb.rdheader(str(base))
        if header.record_name!=base.name:
            raise ValueError('Header record_name does not match metadata filename_lr')
        if header.n_sig!=12 or header.sig_len!=1000 or header.fs!=100:
            raise ValueError('Need 12 leads, 1000 samples, 100 Hz')
        if tuple(header.sig_name)!=EXPECTED_LEADS:
            raise ValueError('Unexpected original lead order/names; refusing silent reorder')
        if tuple(header.units)!=('mV',)*12:
            raise ValueError('Expected physical units mV; no implicit unit conversion')
        if any(name!=base.name+'.dat' for name in header.file_name) or header.fmt!=['16']*12:
            raise ValueError('Expected one official 16-bit data file with matching identity')
        if any(v not in (None,0) for v in header.byte_offset) or header.samps_per_frame!=[1]*12:
            raise ValueError('Unexpected byte offset or multiplexing format')
        if paths[1].stat().st_size!=1000*12*2:
            raise ValueError('Corrupt/unexpected data-file size (expected 24000 bytes)')
        if not np.isfinite(header.adc_gain).all() or np.any(np.asarray(header.adc_gain)<=0):
            raise ValueError('Invalid physical calibration')
        record=wfdb.rdrecord(str(base),physical=False)
        digital=record.d_signal
        if digital.shape!=(1000,12):
            raise ValueError('Unexpected digital ECG shape')
        calculated=np.sum(digital.astype(np.int64),axis=0)%65536
        if any(value is None for value in header.checksum) or not np.array_equal(calculated,np.asarray(header.checksum)%65536):
            raise ValueError('WFDB channel checksum mismatch')
        raw=record.dac(inplace=False)
        if raw.shape!=(1000,12) or not np.isfinite(raw).all():
            raise ValueError('ECG contains nonfinite values or invalid dimensions')
    except Exception as exc:
        raise ValueError(f'Cannot validate PTB-XL ecg_id {ecg_id}: {exc}') from exc
    return LoadedRhythmRecord(identity,np.asarray(raw,dtype=np.float64),tuple(header.sig_name),float(header.fs),
        {path.name:sha256_file(path) for path in paths})


def preprocess_rhythm_record(loaded,*,config=None):
    """Reuse existing filters and Z-score independently per 10-second record/lead."""
    selected=preprocessing_config() if config is None else json.loads(json.dumps(config))
    if selected['sampling_frequency']!=100 or selected['normalization']!={'method':'zscore','scope':'recording_per_lead','epsilon':1e-8,'ddof':0}:
        raise ValueError('Pilot contract requires 100 Hz and documented per-record Z-score')
    filters=selected['filtering']
    # This guard is explicit: neither 50 nor 60 Hz is valid at fs=100 Hz.
    if filters.get('notch_hz') is not None:
        raise ValueError('Notch is disabled in this 100 Hz pilot; 50/60 Hz is invalid at Nyquist')
    if loaded.raw.shape!=(1000,12) or loaded.sampling_frequency!=100 or loaded.channel_names!=EXPECTED_LEADS:
        raise ValueError('Loaded record does not match the 100 Hz / 12-lead pilot contract')
    filtered=filter_ecg(loaded.raw,loaded.sampling_frequency,**filters)
    normalizer=ECGNormalizer('zscore',epsilon=selected['normalization']['epsilon']).fit(
        filtered,scope='recording',source_ids=[str(loaded.identity['ecg_id'])],channel_names=loaded.channel_names)
    normalized=normalizer.transform(filtered,channel_names=loaded.channel_names)
    X=np.ascontiguousarray(normalized.T,dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError('Processed output must be finite')
    selected['fitted_normalization']=normalizer.to_dict()
    return ProcessedRhythmRecord(loaded,filtered,X,np.int64(loaded.identity['y']),selected)


def export_pilot_record(processed,path):
    """Write one processed record, scalar label and original metadata, without pickle."""
    identity=processed.loaded.identity
    arrays={'X':processed.X,'y':np.array(processed.y,dtype=np.int64),
        'ecg_id':np.array(identity['ecg_id'],dtype=np.int64),'patient_id':np.array(identity['patient_id'],dtype=np.int64),
        'strat_fold':np.array(identity['strat_fold'],dtype=np.int64),'rhythm_code':np.array(identity['label']),
        'original_scp_codes':np.array(json.dumps(identity['original_scp_statements'],sort_keys=True)),
        'original_metadata':np.array(json.dumps(identity,sort_keys=True,allow_nan=False)),
        'channel_names':np.array(processed.loaded.channel_names),'class_names':np.array(CLASS_NAMES),
        'sampling_frequency':np.array(processed.loaded.sampling_frequency),
        'preprocessing_configuration':np.array(json.dumps(processed.configuration,sort_keys=True,allow_nan=False)),
        'source_hashes':np.array(json.dumps(processed.loaded.source_hashes,sort_keys=True))}
    with Path(path).open('xb') as output:
        np.savez_compressed(output,**arrays)
    restored=load_pilot_record(path)
    if any(not np.array_equal(arrays[key],restored[key]) for key in arrays):
        raise ValueError('Pilot export/reload mismatch')
    return Path(path)


def load_pilot_record(path):
    """Validate a single pilot export; no dataset splitting or model inference."""
    with np.load(path,allow_pickle=False) as archive:
        data={key:archive[key] for key in archive.files}
    required={'X','y','ecg_id','patient_id','strat_fold','rhythm_code','original_scp_codes','original_metadata',
              'channel_names','class_names','sampling_frequency','preprocessing_configuration','source_hashes'}
    if not required.issubset(data):
        raise ValueError('Missing pilot export fields')
    if data['X'].shape!=(12,1000) or data['X'].dtype!=np.float32 or not np.isfinite(data['X']).all():
        raise ValueError('Expected finite float32 [12,1000] X')
    if data['y'].shape!=() or data['y'].dtype!=np.int64 or int(data['y']) not in range(4):
        raise ValueError('Invalid scalar int64 rhythm label')
    if tuple(data['channel_names'])!=EXPECTED_LEADS or data['class_names'].tolist()!=list(CLASS_NAMES) or float(data['sampling_frequency'])!=100:
        raise ValueError('Invalid channel, class or sampling metadata')
    identity=json.loads(str(data['original_metadata']))
    for field in ('ecg_id','patient_id','strat_fold'):
        if data[field].shape!=() or data[field].dtype!=np.int64 or int(data[field])!=identity[field]:
            raise ValueError('Original metadata/ID mismatch')
    if str(data['rhythm_code'])!=CLASS_NAMES[int(data['y'])] or str(data['rhythm_code'])!=identity['label']:
        raise ValueError('Class mapping mismatch')
    if json.loads(str(data['original_scp_codes']))!=identity['original_scp_statements']:
        raise ValueError('Original statements mismatch')
    return data
