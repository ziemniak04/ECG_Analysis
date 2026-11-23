# PTB-XL — pierwsza architektura 1D-CNN, faza 16

Wersja: `ptbxl-rhythm-cnn-v1`. Implementacja: `src/models/ptbxl_cnn.py`, konfiguracja `src/ptbxl_model_config.py` i jej zapis `configs/ptbxl_cnn_v1.json`. Dotyczy wyłącznie czterech rytmów PTB-XL; MIT-BIH pozostaje osobnym zadaniem. Magdalena przejmuje modelowanie/trening/ewaluację, Anna przede wszystkim Streamlit i integrację.

**Stan weryfikacji: architektura zaimplementowana, ale uruchomienie PyTorch zablokowane brakiem pakietu i DNS.** Liczbę parametrów obliczono niezależnie arytmetycznie; rzeczywisty forward/backward i licznik `nn.Module` wymagają uruchomienia po instalacji. Nie ma wyników klasyfikacji ani wytrenowanych wag.

## Kontrakt

Wejście float32 `[B,12,1000]`, 10 s / 100 Hz / 12 odprowadzeń. Wyjście `[B,4]` — **surowe logity**, bez Softmax w modelu. Mapowanie: SR=0, AFIB=1, STACH=2, SBRAD=3. `CrossEntropyLoss` otrzymuje logity i etykiety torch.long `[B]`, nie one-hot. Wejściowe dane są już przefiltrowane i znormalizowane w eksporcie; model i Dataset nie powtarzają preprocessingu.

| Etap | Operacja | Wyjście | Parametry trenowalne |
|---|---|---|---:|
| Input | Gotowy zapis PTB-XL | `[B,12,1000]` | 0 |
| Block 1 | Conv1d 12→32, k=15, padding=7, bias=False; BN32; ReLU; MaxPool1d(2) | `[B,32,500]` | 5824 |
| Block 2 | Conv1d 32→64, k=9, padding=4, bias=False; BN64; ReLU; MaxPool1d(2) | `[B,64,250]` | 18560 |
| Block 3 | Conv1d 64→128, k=5, padding=2, bias=False; BN128; ReLU; MaxPool1d(2) | `[B,128,125]` | 41216 |
| Pool | AdaptiveAvgPool1d(1), Flatten | `[B,128]` | 0 |
| Head | Linear128→64, ReLU, Dropout(0.3), Linear64→4 | `[B,4]` | 8516 |
| Razem | Bez wag pretrained | | **74116** |

Parametry BN to skala i offset; running statistics są buforami, nie parametrami trenowalnymi. Model podczas konstrukcji porównuje rzeczywisty licznik z arytmetyką konfiguracji. W bieżącym środowisku tego runtime check nie wykonano.

## Uzasadnienie i ograniczenia

Kerneli 15/9/5 próbek nie wybrano na podstawie test fold. Przy 100 Hz obejmują lokalne wzorce w kolejnych skalach; trzy bloki ograniczają wielkość i koszt obliczeń. Bias Conv pominięto przed BN. Global average pooling agreguje cały 10-sekundowy zapis i pozwala uniknąć dużej warstwy Flatten całej sekwencji. Dropout w głowie stanowi prostą regularyzację.

Pole recepcyjne lokalnych cech przed globalnym poolingiem to 54 próbki (~0,54 s), krok 8 próbek. Globalne uśrednianie obejmuje pełne 10 s, ale traci część informacji o kolejności i odstępach między zdarzeniami. To ograniczenie baseline dla rytmów wymagających dłuższego kontekstu, nie dowód wystarczalności tej architektury. Ewentualne zmiany należy porównywać w przyszłej fazie wyłącznie na walidacji, bez użycia fold 10 do wyboru modelu.

Nie estymujemy tu medycznej skuteczności ani nie obiecujemy rozpoznawania wszystkich arytmii. Cztery etykiety pochodzą z jawnie wybranej kohorty record-level, a SR nie oznacza zdrowego ECG.

## Uruchomienie po instalacji

```python
import torch
from src.models.ptbxl_cnn import PTBXLRhythmCNN

model = PTBXLRhythmCNN()
print(model.parameter_count())  # oczekiwane 74116; runtime do sprawdzenia
# Właściwy smoke test czyta rzeczywiste rekordy z training folds:
# .venv/bin/python -m examples.phase16_model_smoke --output-dir results/model_setup/phase16_verified
```

CPU jest domyślnym urządzeniem smoke testu; nie wykryto dostępnego nvidia-smi ani urządzeń /dev/nvidia*, a działającej zgodności GPU nie potwierdzono. Nie należy zakładać CUDA. W przyszłym środowisku GPU sprawdzić również `torch.cuda.is_available()` i rzeczywistą operację tensorową przed wyborem urządzenia.
