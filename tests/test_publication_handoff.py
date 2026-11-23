"""Offline publication safety checks: no Git writes, network, ECG or model calls."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('publication',Path(__file__).parents[1]/'scripts/publish_verified_handoff.py')
publication=importlib.util.module_from_spec(spec);spec.loader.exec_module(publication)

@pytest.mark.parametrize('path',['data/record.csv','results/model.pt','.venv/file.py','source.npz','secret.pem','../outside.py','/absolute.py','.env','__pycache__/module.py'])
def test_forbidden_paths_rejected(path):
    with pytest.raises(ValueError):publication.safe_text(path)


def test_secrets_detected_without_printing_value():
    value='ghp_'+'A'*30
    with pytest.raises(ValueError) as error:publication.scan('fixture.py',value.encode())
    assert value not in str(error.value)
    with pytest.raises(ValueError):publication.scan('fixture.py',b'\x00binary')
    with pytest.raises(ValueError):publication.scan('fixture.py',b'A'*(1024*1024+1))


def test_remote_feature_divergence_stops_before_staging(monkeypatch):
    calls=[];base='a'*40;remote='b'*40
    def fake(*args,**kwargs):
        calls.append(args);out='';code=0
        if args==('rev-parse','HEAD'):out=base
        elif args[0]=='ls-remote':out=remote+'\trefs/heads/'+publication.EXPECTED_BRANCH+'\n'
        elif args[:2]==('show-ref','--verify'):out=remote
        elif args[0]=='merge-base':code=1
        return SimpleNamespace(stdout=out,stderr='',returncode=code)
    monkeypatch.setattr(publication,'git',fake)
    with pytest.raises(ValueError,match='Remote feature contains new commits'):
        publication.publish(dict(base_commit=base,branch=publication.EXPECTED_BRANCH))
    assert not any(c[0] in ('add','rm','commit','push') for c in calls)


def test_main_ahead_is_informational_not_an_automatic_merge(monkeypatch):
    calls=[]
    def fake(*args,**kwargs):
        calls.append(args)
        return SimpleNamespace(stdout='a2fe7c5' if args[0]=='rev-parse' else '16',stderr='',returncode=0)
    monkeypatch.setattr(publication,'git',fake)
    assert publication.check_feature_remote(publication.EXPECTED_BRANCH,None)=='a2fe7c5'
    assert not any(c[0] in ('add','commit','push','merge','rebase','reset','clean') for c in calls)


def test_main_cannot_be_a_push_target(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No Git writes for a main target')
    monkeypatch.setattr(publication,'git',forbidden)
    with pytest.raises(ValueError):publication.publish({'branch':'main'})


def test_unpublished_dataset_history_stops_before_staging(monkeypatch):
    calls=[];base='a'*40
    def fake(*args,**kwargs):
        calls.append(args);out=''
        if args==('rev-parse','HEAD'):out=base
        elif args[0]=='rev-list':out='unpublished-dataset-commit'
        return SimpleNamespace(stdout=out,stderr='',returncode=0)
    monkeypatch.setattr(publication,'git',fake)
    monkeypatch.setattr(publication,'check_feature_remote',lambda branch,tip:'main-tip')
    with pytest.raises(ValueError,match='Unpublished base history'):
        publication.publish(dict(base_commit=base,branch=publication.EXPECTED_BRANCH,untrack_local_only=['data/ptbxl/metadata/ptbxl_database.csv']))
    assert not any(c[0] in ('add','rm','commit','push') for c in calls)


@pytest.mark.parametrize('already_pushed',[True,False])
def test_verified_existing_commit_is_reused_never_duplicated(monkeypatch,already_pushed):
    base='a'*40;head='c'*40;calls=[];pushed=[False];verified=[]
    def fake(*args,**kwargs):
        calls.append(args);out=''
        if args==('rev-parse','HEAD'):out=head
        elif args==('rev-parse','HEAD^'):out=base
        elif args[0]=='push':
            assert args==('push','--set-upstream','origin','HEAD:refs/heads/'+publication.EXPECTED_BRANCH)
            pushed[0]=True
        elif args[0] in ('add','rm','commit','merge','rebase','reset','clean'):raise AssertionError('Do not repeat/mix completed work')
        return SimpleNamespace(stdout=out,stderr='',returncode=0)
    def remote(branch):
        if branch=='main':return 'main-tip'
        return head if already_pushed or pushed[0] else None
    monkeypatch.setattr(publication,'git',fake);monkeypatch.setattr(publication,'remote_tip',remote)
    monkeypatch.setattr(publication,'verify_tree',lambda m,**kw:verified.append(True))
    monkeypatch.setattr(publication,'check_feature_remote',lambda branch,tip:'main-tip')
    publication.publish(dict(base_commit=base,branch=publication.EXPECTED_BRANCH,untrack_local_only=[]))
    assert len(verified)==1 and pushed[0]==(not already_pushed)
    assert not any(c[0]=='commit' for c in calls)
