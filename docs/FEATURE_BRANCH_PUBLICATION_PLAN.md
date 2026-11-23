# Oddzielna publikacja AI — bez integracji z main

Zweryfikowany lokalny stan: `feature/ptbxl-1dcnn`, baza `b5859b6b1b6d026428fcd5a4364c1951d4c62502`. Zapisany `origin/main`: `a2fe7c589743321f28b47706fdf0529174ba5222`,16 commitów ponad bazą. To dwie gałęzie pracy. Publikacja nowego commitu na feature NIE wymaga merge/rebase main i nie aktualizuje main.

## Przejrzane zdalne commity, chronologicznie

- `954f4b3`: loadery i podziały pacjentowe (`src/ml/data.py`).
- `213c0c3`: dokumentacja pobierania MIT-BIH (`docs/README.md`, `docs/ml_pipeline.md`).
- `8578749`: ignorowanie checkpointów/plików tymczasowych (`.gitignore`).
- `f73edd6`: rezydualna architektura (`src/ml/models.py`).
- `80f6ad4`: RR features w osobnym pipeline main.
- `c68eb00`: raport klas per split.
- `dce89d8`: pętla treningu (`src/ml/train.py`).
- `692eaea`: ewaluacja (`src/ml/evaluate.py`).
- `1b51f8f`: agregaty metryk/wizualizacje walidacyjne w `results/ml/`.
- `d13972e`: opis modeli i ograniczeń.
- `4377d10`: `.vscode` i Streamlit secrets ignore.
- `b940643`: inference helpers i parity tests (`src/ml/inference.py`, `tests/test_inference.py`).
- `needa794`: viewer i testy (`src/ml/viewer.py`, `tests/test_viewer.py`).
- `963109d`: dashboard (`streamlit_app.py`, `.streamlit/config.toml`).
- `ad1a7d8`: wyniki testowe innych modeli w `results/ml/`.
- `a2fe7c5`: aktualizacja dokumentacji, aplikacji i decyzji modelowej.

Odczytano wyłącznie historię/bloby kodu i dokumentacji; nie wykonano zdalnego kodu, treningu ani ewaluacji. Pliki `streamlit_app.py`, `src/ml/*`, `docs/ml_pipeline.md`, `docs/README.md`, `.streamlit/config.toml` i testy Anny NIE są nadpisywane lub kopiowane do lokalnego worktree.

## Nakładanie zmian i ryzyko integracji

Bezpośrednie przecięcie zmienionych ścieżek względem `b5859b6` to **`.gitignore`**. Main dodaje26 ścieżek oprócz tej zmiany. Inne pliki AI mają odrębne ścieżki, ale istnieją istotne konflikty semantyczne:

| Obszar | Main, przejrzany a2fe7c5 | Gałąź AI | Wymagany ręczny przegląd PR |
|---|---|---|---|
| Modele | `src/ml/models.py`,residual `ECGNet` | `src/models/ptbxl_cnn.py`, `mitbih_cnn_rr.py` | Nie zamieniać checkpointów między klasami. |
| Checkpoint schema | `state_dict`, `in_channels`, `n_classes`, `n_extra`, `extra_mean/std` | `model_state_dict`, model/preprocessing configuration i RR_transform | Jawny adapter formatów lub wybór właściwego loadera; nie samo przemianowanie pliku. |
| MIT-BIH RR |5kolumn:preRR,postRR,mean10,pre/mean,post/mean |6kolumn:past-only previousRR,median8 wyłączając current previous,ratio,3maski | Wagi i scaler nie są kompatybilne; nie ucinać/dopisywać kolumn. |
| MIT-BIH split |Procedura70/15/15 opisana w main |Zamrożone29/9/9 grup | Wyników różnych kohort nie scalać i nie traktować jako sparowanego porównania. Nie zmieniać już użytych testów. |
| UI/inferencja |Streamlit i detektor/helpers już istnieją na main |Handoff dotyczy osobno zweryfikowanych modeli AI | Kod UI main nie jest automatycznie zwalidowany dla naszego checkpointu/kontraktuRR. |
| Dokumentacja |`docs/ml_pipeline.md`, `docs/README.md` |Root README, phase reports, nowy handoff | Zachować obie proweniencje i uniknąć globalnych sprzecznych deklaracji. |
| Ignore rules |m.in. niezakotwiczone `models/`, `.vscode/`, Streamlit secrets |rozszerzony zakaz danych/wag/caches | Przy rozwiązaniu konfliktu `models/` rozważyć `/models/` dla checkpointów: nie wykluczać nowych źródeł `src/models/`. Zachować zasady secrets i `.vscode`. |

Nowe `.gitignore` nie usuwa już śledzonych wyników Anny z `results/ml/`; skrypt nie untrackuje tych ścieżek. Lokalne jawne wykluczenia danych obejmują5 starszych śledzonych metadanych/manifestów z bazy; używany jest wyłącznie `git rm --cached`, bez kasowania plików na dysku. To będą widoczne usunięcia wpisów w drzewie commitu, wymagające przeglądu PR. Nie są to pliki Streamlit Anny. Przed ewentualnym późniejszym checkout/merge należy zachować osobne lokalne kopie danych — publikujący skrypt nie wykonuje checkout ani merge i nie usuwa danych.

## Publikacja w normalnym terminalu VS Code

```bash
cd ~/Pulpit/ECG-APP
.venv/bin/python scripts/publish_verified_handoff.py
.venv/bin/python scripts/publish_verified_handoff.py --publish
```

Domyślna kontrola jest read-only/offline. `--publish` robi fetch, ale nigdy merge/rebase/reset/clean. Sprawdza WYŁĄCZNIE możliwość bezpiecznej aktualizacji feature branch: nowa historia na main jest oczekiwana i informacyjna. Obca/nowsza historia na zdalnej feature zatrzymuje wykonanie. Sprawdza wykluczenia, jawny allowlist, sekret/rozmiar/blob hash przed i po commicie. Nie publikuje nieznanej lokalnej historii z danymi. Tworzy jeden commit tylko gdy potrzebny; istniejący równoważny commit jest sprawdzany po rodzicu i pełnym drzewie, a potem ponownie używany. Nigdy nie wykonuje push na main ani force-push. Zatrzymuje się na błędzie uprawnień/network/auth, bez automatycznego retry.

## PR po potwierdzonym push

Compare URL: https://github.com/ziemniak04/ECG_Analysis/compare/main...feature/ptbxl-1dcnn

Wybrać base=`main`,compare=`feature/ptbxl-1dcnn`. Najpierw sprawdzić Files changed, konflikt `.gitignore`, usunięcie śledzenia danych i różne kontrakty modeli. Następnie utworzyć PR i uzgodnić adapter/model z Anną. Nie scalać automatycznie i nie uruchamiać ponownie końcowych testów modeli. Brak Streamlit w bazowanym na starszym commicie worktree feature nie oznacza usunięcia nowych plików Anny z main: Git porównuje zmiany do wspólnej bazy. Ostateczny efekt integracji wymaga jednak ręcznego przeglądu PR.
