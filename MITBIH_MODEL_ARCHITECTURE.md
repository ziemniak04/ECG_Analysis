# MIT-BIH — osobny bazowy model uderzeń, faza 20

Model `src/models/mitbih_cnn.py::MITBIHBeatCNN` jest niezależny od zakończonego modelu rytmów PTB-XL. Nie korzysta z jego wag i nie zmienia jego architektury. Konfigurację przechowuje `src/mitbih_model_config.py::BeatCNNConfig` (wersja `mitbih-beat-cnn-v1`).

## Kontrakt tensorów

Wejście: `torch.float32 [B, 2, 252]`, wartości skończone. Każdy przykład to jedno uderzenie wycięte z oryginalnych adnotacji; częstotliwość 360 Hz, zakres `[p-90, p+162)`, adnotacja pod indeksem 90. Model nie wykrywa uderzeń w ciągłym sygnale. Wyjście: pięć surowych logitów `[B, 5]`, w kolejności **N, S, V, F, Q**. Etykiety: `torch.long [B]`, liczby 0–4. Przed `CrossEntropyLoss` nie należy stosować Softmax.

| Warstwa | Ustawienia | Wyjście |
|---|---|---|
| Wejście | Dwa oryginalne kanały, bez zmiany kolejności | B × 2 × 252 |
| Blok 1 | Conv1d 2→16, kernel 15, padding 7, bez bias; BatchNorm, ReLU, MaxPool(2) | B × 16 × 126 |
| Blok 2 | Conv1d 16→32, kernel 9, padding 4, bez bias; BatchNorm, ReLU, MaxPool(2) | B × 32 × 63 |
| Blok 3 | Conv1d 32→64, kernel 5, padding 2, bez bias; BatchNorm, ReLU, MaxPool(2) | B × 64 × 31 |
| AdaptiveAvgPool1d | Długość 1, potem Flatten | B × 64 |
| Klasyfikator | Linear 64→32, ReLU, Dropout 0.3, Linear 32→5 | B × 5 |

**17 797 uczonych parametrów**, z niezależną kontrolą arytmetyczną i testem wartości rzeczywistej. Jądra 15/9/5 obejmują lokalne struktury fali; kolejne bloki zwiększają kontekst. Globalne uśrednianie ogranicza rozmiar głowicy. To prosty punkt startowy, a nie architektura wybrana na podstawie wyników walidacji. Jego skuteczność nie jest jeszcze znana.

## Dane i ograniczenia

Korzystamy wyłącznie z gotowego `data/processed/ecg_dataset_v1/`: filtracja offline zero-phase HP 0.5 Hz / LP 45 Hz, brak notch, Z-score oddzielnie dla całego nagrania i kanału, potem segmentacja. Faza 20 nie filtruje ani nie normalizuje ponownie. Normalizacja całego nagrania wymaga tego samego kontekstu przy późniejszej inferencji; nie jest procedurą przyczynową dla pojedynczego napływającego uderzenia. Nie wyznacza wspólnych statystyk z połączonych pacjentów.

Indeksy kanałów nie oznaczają zawsze tych samych odprowadzeń; loader zachowuje oryginalne nazwy, w tym odmienną konfigurację nagrania 114 oraz zapisy V5/V2. Nie przeprowadzono harmonizacji odprowadzeń. N oznacza grupę normal-related, nie diagnozę zdrowego pacjenta; Q obejmuje m.in. uderzenia stymulatorowe i nie jest jedną chorobą. Duże `F` i małe `f` mają różne znaczenie. To klasyfikacja uderzeń, a nie diagnoza rytmu AFIB/STACH/SBRAD.

## Co zweryfikowano

Jeden rzeczywisty batch treningowy `[10,2,252]`, pięć logitów, skończona strata, niezerowe skończone gradienty, jedna aktualizacja Adam oraz zgodność zapisu/odczytu wag. Zapis `smoke_after_one_update.pt` jest oznaczony **not trained model / not for evaluation or application**. Nie zmierzono Accuracy, F1 ani skuteczności medycznej i nie wybrano modelu na walidacji.
