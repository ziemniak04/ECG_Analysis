"""ONE authorized Phase26 attempt. An existing claim blocks all reruns."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.mitbih_execution_guard import HoldoutAccessGuard
ROOT=Path('results/model_evaluation/phase26_mitbih_final_test')
if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    with HoldoutAccessGuard(ROOT/'preflight_guard.json') as old:
        old.probe()
        from src.mitbih_final_test import preflight,execute_once
        out=ROOT/'run_001'
        values=preflight(out)
    print('Pre-evaluation verification passed; frozen manifest saved before any test read.',flush=True)
    execute_once(out,*values,old)
