# MIT-BIH — interfejs danych i protokół przyszłego treningu

Faza 20 przygotowuje drugi, osobny model w tym samym ECG-APP. Nie uruchamia pełnego treningu ani oceny testowej. Dotychczasowe wyniki PTB-XL pozostają zamknięte.

## Lokalizacje i odtwarzalność

- Eksport: `data/processed/ecg_dataset_v1/dataset_metadata.json` oraz `recordings/<record_id>.npz` — bez zmian.
- Zamrożony protokół: `configs/mitbih_patient_split_v1.json`.
- SHA256 protokołu: `8c5cc28d8dcae170fd222432d9f247fd19207ba01be60763241da4b7cae61831`.
- Audyt i smoke: `results/model_setup/phase20_mitbih/run_002/`.
- Loader: `src/mitbih_torch.py`; indeks/protokół/wagi: `src/mitbih_protocol.py`.
- Architektura: `MITBIH_MODEL_ARCHITECTURE.md` i `src/models/mitbih_cnn.py`.

Źródłowy manifest jest przypięty sumą kontrolną w protokole. Każdy shard jest sprawdzany SHA256 przy braku w cache, a następnie walidowany istniejącym loaderem eksportu: kształty, typy, skończoność, symbole i ich mapowanie, metadane. Zmiana rozmiaru lub czasu modyfikacji pliku w cache powoduje błąd. Brakujące/uszkodzone próbki nie są po cichu pomijane.

## Podział przed treningiem — grupy pacjentów, nie losowe uderzenia

Nie istnieją tu oficjalne foldy analogiczne do PTB-XL. Zamrożono własny, jawny protokół dla wszystkich 48 nagrań i 47 **technicznych grup podmiotów**, nie prawdziwych tożsamości. Relacja 201/202 pochodzi z udokumentowanego powiązania; obydwa nagrania mają `mitdb_group_201_202` i są razem w walidacji. Zbiory grup są rozłączne i łącznie obejmują cały eksport.

| Część | Grupy | Nagrania | Uderzenia | N | S | V | F | Q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Trening | 29 | 29 | 62926 | 52254 | 1715 | 4389 | 401 | 4167 |
| Walidacja | 9 | 10 | 23134 | 19094 | 491 | 1444 | 38 | 2067 |
| Końcowy test | 9 | 9 | 23392 | 19246 | 575 | 1402 | 363 | 1806 |
| Suma | 47 | 48 | 109452 | 90594 | 2781 | 7235 | 802 | 8040 |

| Część | Grupy z N | z S | z V | z F | z Q |
|---|---:|---:|---:|---:|---:|
| Trening | 28 | 18 | 21 | 10 | 5 |
| Walidacja | 9 | 6 | 8 | 4 | 2 |
| Końcowy test | 9 | 7 | 7 | 2 | 2 |

Nagrania treningowe: 100, 101, 102, 103, 106, 107, 108, 109, 111, 113, 114, 115, 116, 117, 118, 122, 123, 124, 200, 208, 214, 215, 219, 220, 221, 230, 231, 232, 233.

Walidacja: 104, 105, 119, 121, 201, 202, 205, 210, 222, 223.

Końcowy test: 112, 203, 207, 209, 212, 213, 217, 228, 234. To jedynie inwentarz metadanych; nie wykonano na nim predykcji nowego CNN.

### Uzasadnienie i granice podziału

Najpierw odczytano i porównano etykiety/metadane wszystkich 48 shardów; bez dekodowania X i bez wyników modelu. Nominalny cel to 60/20/20 uderzeń, przy stałych liczbach grup 29/9/9. Reguły kotwiczą dominujące źródła F (208 trening, 213 test; 223 i 205 walidacja, 233 trening), Q (102 i 107 trening, 104 walidacja, 217 test) oraz największe źródło S (232 trening). Dwa treningowe źródła Q reprezentują odmienne konfiguracje odprowadzeń.

Następnie sprawdzono 20000 deterministycznych przydziałów pozostałych grup, seed 42. Warunki minimalnej liczby grup z daną klasą w każdej części: N/S/V ≥3, F/Q ≥2. Spośród 5197 poprawnych kandydatów wybrano minimum jawnej funkcji: suma kwadratów odchyleń udziałów każdej klasy od 60/20/20 plus 0.5 razy analogiczna suma dla liczby uderzeń. Rozstrzyganie remisów jest deterministyczne. Wszystkie reguły ustalono przed uczeniem; nie było optymalizacji według ECG ani predykcji. Rzeczywiste udziały uderzeń: 57.49/21.14/21.37%; nie należy ich nazywać dokładnym podziałem 60/20/20. Nie wolno później przestawiać grup w odpowiedzi na wyniki modelu.

**Ograniczenie F:** 734 z 802 przykładów (91.52%) pochodzi z dwóch grup: 208 i 213. Nie da się proporcjonalnie rozdzielić tych dwóch osób między trzy części. Walidacja ma tylko 38 F. W teście 362/363 F pochodzi z 213, mimo dwóch grup z dodatnią liczbą F. **Ograniczenie Q:** 8025/8040 (99.81%) pochodzi z czterech grup; walidacja 2062/2067 z 104, test 1802/1806 z 217. Liczba uderzeń znacznie zawyża efektywną różnorodność pacjentów. Dodatnia liczba przykładów nie usuwa tego ograniczenia; późniejsze wyniki F/Q będą podatne na różnice między osobami i odprowadzeniami. Nie zmieniamy mapowania ani nie usuwamy tych klas.

Ten protokół obejmuje nagrania stymulatorowe i nie jest standardowym benchmarkiem ograniczonym do 44 nagrań; porównania literaturowe muszą uwzględniać tę różnicę.

## Minimalny interfejs PyTorch (bez pętli treningowej)

Uruchamiać z katalogu projektu w istniejącym `.venv`:

```python
import torch
from torch.utils.data import DataLoader
from src.mitbih_protocol import MITBIHExportIndex, load_split, training_class_weights
from src.mitbih_torch import MITBIHBeatDataset, collate_mitbih
from src.models.mitbih_cnn import MITBIHBeatCNN

index = MITBIHExportIndex()
split = load_split(index)
train = MITBIHBeatDataset(index, split, partition='train', max_cached_shards=2)
loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=0,
                    generator=torch.Generator().manual_seed(42),
                    collate_fn=collate_mitbih)
batch = next(iter(loader))
model = MITBIHBeatCNN()
logits = model(batch['X'])  # float32 [B, 5], bez Softmax
labels = batch['y']        # long [B], N=0 S=1 V=2 F=3 Q=4
# Opcjonalnie; wszystkie i wyłącznie grupy treningowe:
weights = torch.from_numpy(training_class_weights(index, split, split['groups']['train']))
criterion = torch.nn.CrossEntropyLoss(weight=weights)
```

Każda próbka zawiera ponadto `record_id`, `patient_group`, `original_symbol`, `annotation_position`, `annotation_index`, `channel_names`, `sampling_frequency`, `source_row` i `global_source_index`. Własny collate zachowuje zgodność metadanych w batchu. Eksport nie jest modyfikowany; tensor X jest kopią pojedynczego uderzenia, więc operacje in-place nie zmienią cache.

Indeks jest wyłącznie metadanymi. X ładuje się dopiero przy `__getitem__`. Skompresowany NPZ wymaga rozpakowania całego nagrania; nie jest to mmap pojedynczego uderzenia. LRU domyślnie przechowuje najwyżej dwa shardy na proces. Pamięć rośnie z liczbą workerów. Losowe mieszanie wszystkich uderzeń może często wypierać shardy z cache — przed pełnym treningiem trzeba zmierzyć przepustowość i dobrać limit cache/workery w dostępnej pamięci. Faza 20 nie przebudowuje eksportu.

Walidacja może być później utworzona `partition='validation'`. `partition='test'` domyślnie zgłasza `PermissionError` jeszcze przed odczytem waveformu. Istnieje jawne `allow_final_test=True`, przeznaczone wyłącznie do późniejszej, osobno zatwierdzonej końcowej oceny. To zabezpieczenie interfejsu, nie system kontroli uprawnień. W fazie 20 go nie użyto dla rzeczywistych danych.

## Opcjonalna obsługa nierównowagi

Wagi `n_train/(5*n_class_train)` dla N/S/V/F/Q: około **0.240847, 7.338309, 2.867441, 31.384539, 3.020206**. Funkcja wymaga jawnej, dokładnej listy wszystkich grup treningowych; odrzuca grupy walidacyjne/testowe, duplikaty i niepełną listę. Etykiety walidacji/testu nie służą do obliczania wag. Wagi włączono tylko w jednorazowym smoke teście, aby sprawdzić zgodność z CE; nie ustalono jeszcze optymalnej strategii pełnego uczenia. Bardzo duża waga F wymaga ostrożności. Nie wykonywano oversamplingu, undersamplingu ani augmentacji.

## Reprodukcja i następny krok

Istniejące wyniki są nieprzepisywalne. `examples/setup_mitbih_model.py` ma etapy `audit`, `freeze`, `smoke`; audyt wymaga nowego katalogu, freeze odmawia nadpisania zamrożonego protokołu, smoke odmawia powtórnego zapisu. Do porównania audytu można użyć nowej, nieistniejącej lokalizacji:

```bash
.venv/bin/python -m examples.setup_mitbih_model audit --output-dir results/model_setup/phase20_mitbih/audit_recheck_001
.venv/bin/python -m pytest tests/test_mitbih_setup.py
.venv/bin/python -m pytest
```

Nie uruchamiać ponownie freeze/smoke w istniejącym runie. Odtwarzanie konfiguracji nie wymaga powtarzania aktualizacji wag. Przed przyszłym pełnym treningiem: zatwierdzić oddzielny eksperyment, zmierzyć I/O i RAM, skonfigurować deterministyczny trening i checkpointy oraz wybór modelu wyłącznie na walidacji. Test ma pozostać zamknięty do jednej końcowej oceny. Nie przenosić etykiet rytmów PTB-XL ani jego progów do tego zadania.
