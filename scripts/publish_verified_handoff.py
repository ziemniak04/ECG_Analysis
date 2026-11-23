"""Review or publish the explicit handoff manifest. Never add directories or force push.

Default is read-only audit. --publish is only for the owner's normal terminal:
requires writable .git, working network/auth, unchanged audited files and no divergent remote feature commits.
Remote main may advance independently and is never a push target. Existing patient metadata is untracked, never deleted.
"""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'docs/PUBLICATION_MANIFEST.json'
EXPECTED_ORIGIN='https://github.com/ziemniak04/ECG_Analysis.git'
EXPECTED_BRANCH='feature/ptbxl-1dcnn'
TEXT_SUFFIXES={'.py','.md','.txt','.json','.csv','.toml','.yaml','.yml'}
DENIED_SUFFIXES={'.npz','.npy','.dat','.hea','.atr','.qrs','.ann','.edf','.pt','.pth','.ckpt','.pkl','.pickle','.pem','.key','.h5','.hdf5','.mat'}
DENIED_PARTS={'.git','.venv','.venv-preprocessing','__pycache__','data','results','.aws','.ssh','.codex','.agents','.cache'}
PATTERNS=[re.compile(r'gh[pousr]_[A-Za-z0-9]{25,}'),re.compile(r'github_pat_[A-Za-z0-9_]{30,}'),re.compile(r'sk-[A-Za-z0-9_-]{24,}'),re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),re.compile(r'^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',re.M),re.compile(r'https?://[^\s/:@]+:[^\s/@]+@'),re.compile(r'''(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[=:]\s*["']([^"'\n]{16,})["']''')]


def git(*args,check=True):
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0')
    return subprocess.run(['git',*args],cwd=ROOT,env=env,check=check,text=True,capture_output=True)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_text(path):
    p=Path(path)
    if p.is_absolute() or '..' in p.parts or set(p.parts)&DENIED_PARTS or p.suffix.lower() in DENIED_SUFFIXES or p.name.startswith('.env') or p.name in ('secrets.toml','.git-credentials'):raise ValueError('Forbidden publication path: '+path)
    if p.suffix not in TEXT_SUFFIXES and p.name!='.gitignore':raise ValueError('Unexpected file type: '+path)
    return p


def scan(path,content):
    if len(content)>1024*1024:raise ValueError('File exceeds explicit 1MiB audit cap: '+path)
    text=content.decode('utf-8')
    if '\0' in text:raise ValueError('Binary content: '+path)
    hits=[i for i,pattern in enumerate(PATTERNS) if pattern.search(text)]
    if hits:raise ValueError('Potential secret; inspect locally without printing values: '+path+' patterns='+str(hits))


def review():
    m=json.loads(MANIFEST.read_text());paths=[r['path'] for r in m['files']]
    if len(paths)!=len(set(paths)):raise ValueError('Duplicate allowlist entries')
    if m['branch']!=EXPECTED_BRANCH:raise ValueError('Only the approved feature branch may be published')
    if git('remote','get-url','origin').stdout.strip()!=EXPECTED_ORIGIN:raise ValueError('Unexpected origin')
    if git('branch','--show-current').stdout.strip()!=m['branch']:raise ValueError('Branch changed since audit')
    for row in m['files']:
        p=ROOT/safe_text(row['path'])
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:raise ValueError('Audited file changed: '+row['path'])
        scan(row['path'],p.read_bytes())
    scan('docs/PUBLICATION_MANIFEST.json',MANIFEST.read_bytes())
    for root in ('src','examples','tests','scripts'):
        actual={str(p.relative_to(ROOT)) for p in (ROOT/root).rglob('*.py') if '__pycache__' not in p.parts}
        missing=actual-set(paths)
        if missing:raise ValueError('Important source not audited: '+str(sorted(missing)))
    print('Reviewed safe files:',len(paths)+1,'bytes:',sum(r['bytes'] for r in m['files'])+MANIFEST.stat().st_size)
    print('Branch:',m['branch'],'origin:',EXPECTED_ORIGIN)
    print('Untrack only; preserve local files:',', '.join(m['untrack_local_only']))
    return m


def remote_tip(branch):
    r=git('ls-remote','--heads','origin','refs/heads/'+branch)
    return r.stdout.split()[0] if r.stdout.strip() else None


def expected_hashes(m):
    return {**{r['path']:r['sha256'] for r in m['files']},'docs/PUBLICATION_MANIFEST.json':digest(MANIFEST)}


def verify_tree(m,*,index=False):
    expected=expected_hashes(m);paths=[]
    command=('ls-files','--stage','-z') if index else ('ls-tree','-r','-z','HEAD')
    for row in git(*command).stdout.split('\0'):
        if not row:continue
        meta,path=row.split('\t',1)
        if index:
            mode,oid,stage=meta.split()
            if stage!='0':raise ValueError('Unmerged staged file: '+path)
        else:
            mode,kind,oid=meta.split()
            if kind!='blob':raise ValueError('Unsupported committed entry: '+path)
        if mode not in ('100644','100755'):raise ValueError('Unsupported file mode: '+path)
        safe_text(path);blob=subprocess.check_output(['git','cat-file','blob',oid],cwd=ROOT)
        scan(path,blob)
        if hashlib.sha256(blob).hexdigest()!=expected.get(path):raise ValueError('Git content differs from audited bytes: '+path)
        paths.append(path)
        if index:print(str(len(blob)).rjust(8),path)
    if set(paths)!=set(expected):raise ValueError('Git tree differs from complete safe allowlist')


def check_feature_remote(branch,tip):
    if branch!=EXPECTED_BRANCH:raise ValueError('Only feature/ptbxl-1dcnn is authorized; never main')
    if tip is not None:
        ref='refs/remotes/origin/'+branch
        found=git('show-ref','--verify','--hash',ref,check=False)
        if found.returncode or found.stdout.strip()!=tip:raise ValueError('Fetched feature ref differs from remote; inspect before continuing')
        if git('merge-base','--is-ancestor',tip,'HEAD',check=False).returncode:
            raise ValueError('Remote feature contains new commits/divergence: STOP; do not overwrite teammate work')
    # Main divergence is EXPECTED and informational for this separately published branch.
    main=git('rev-parse','--verify','refs/remotes/origin/main',check=False)
    if main.returncode:raise ValueError('Cannot inspect fetched origin/main')
    ahead=git('rev-list','--count','HEAD..refs/remotes/origin/main').stdout.strip()
    print('origin/main:',main.stdout.strip(),'commits absent locally:',ahead,'(left unchanged; no merge/rebase)')
    return main.stdout.strip()


def publish(m):
    branch=m['branch']
    if branch!=EXPECTED_BRANCH:raise ValueError('Only the approved feature branch is authorized')
    head=git('rev-parse','HEAD').stdout.strip();reuse=head!=m['base_commit']
    if git('diff','--cached','--name-only').stdout.strip():raise ValueError('Existing staged changes: inspect first, do not mix work')
    if reuse:
        # Reuse only the exact audited publication commit, not a similarly named commit.
        if git('rev-parse','HEAD^').stdout.strip()!=m['base_commit']:raise ValueError('Unexpected local history; inspect instead of creating a duplicate')
        verify_tree(m)
        print('Reusing existing verified local publication commit:',head)
    git('fetch','origin')
    tip=remote_tip(branch);main_before=check_feature_remote(branch,tip)
    if reuse and tip==head:
        print('VERIFIED: remote feature already contains the audited commit:',head)
        print('Compare: https://github.com/ziemniak04/ECG_Analysis/compare/main...'+branch)
        return
    # Excluded old data may only remain in history already shared with origin.
    # Check BASE, not the new safe commit (which intentionally untracks those files).
    old=git('rev-list',m['base_commit'],'--not','--remotes=origin','--',*m['untrack_local_only']).stdout.strip()
    if old:raise ValueError('Unpublished base history contains excluded dataset/record manifests; STOP, do not rewrite/upload it automatically')
    if not reuse:
        for p in m['untrack_local_only']:git('rm','--cached','--ignore-unmatch','--',p)
        paths=[r['path'] for r in m['files']]+['docs/PUBLICATION_MANIFEST.json']
        for start in range(0,len(paths),40):git('add','--',*paths[start:start+40])
        verify_tree(m,index=True)
        print(git('diff','--cached','--stat').stdout);git('diff','--cached','--check')
        if git('diff','--cached','--quiet',check=False).returncode!=1:raise ValueError('No valid new staged diff')
        if remote_tip(branch)!=tip:raise ValueError('Remote feature changed during audit; STOP before commit')
        print(git('commit','-m','Add completed ECG AI pipelines and Streamlit integration handoff').stdout)
        head=git('rev-parse','HEAD').stdout.strip();verify_tree(m)
    if remote_tip(branch)!=tip:raise ValueError('Remote feature changed; STOP before push, keep the local commit')
    pushed=git('push','--set-upstream','origin','HEAD:refs/heads/'+EXPECTED_BRANCH)
    print(pushed.stdout);print(pushed.stderr)
    if remote_tip(branch)!=head:raise ValueError('Remote feature commit verification failed')
    print('VERIFIED remote branch:',branch,'commit:',head)
    print('main before:',main_before,'main now:',remote_tip('main'),'(main was never pushed by this script)')
    print('Create PR only; do not auto-merge: https://github.com/ziemniak04/ECG_Analysis/compare/main...'+branch)
    print(git('status','--short').stdout or 'Working tree clean (ignored local files remain).')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    try:
        manifest=review()
        if args.publish:publish(manifest)
        else:print('Read-only review completed. No staging, commit or network operation performed.')
    except (ValueError,OSError,subprocess.CalledProcessError) as error:
        print('STOP:',str(error),file=sys.stderr)
        if isinstance(error,subprocess.CalledProcessError):print(error.stderr.strip(),file=sys.stderr)
        sys.exit(1)
