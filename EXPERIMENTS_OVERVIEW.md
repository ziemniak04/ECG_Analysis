# Ukończone eksperymenty ECG-APP

Stan bieżący: PTB-XL Phase19 i MIT-BIH Phase26 zamknięte. Dwa zadania pozostają oddzielne. Poniższe liczby pochodzą z zapisanych raportów/metryk, bez ponownego uruchamiania modeli. Bezpieczne agregaty w `docs/experiments/`; szczegółowe predykcje i audyty per pacjent pozostają lokalne.

| Zadanie/faza | Kohorta | Accuracy | Macro F1 | Macro AUPRC | Decyzja |
|---|---|---|---|---|---|
| PTB-XL18 | pełna walidacja9 |0.9019|0.757428|0.857313|weighted baseline |
| PTB-XL18C B | pełna walidacja9 |0.9433|0.832422|0.878126|wybrany unweighted,argmax |
| PTB-XL19 | finalny test10,1958ECG |0.927477|0.792973|0.850302|jednorazowy, zamknięty |
| MIT-BIH22 | walidacja,23134beaty |0.841489|0.354215|0.416958|morphology CNN,sqrt-balanced |
| MIT-BIH23A | ta sama walidacja |0.855970|0.342391|0.414563|unweighted,nie zastąpił22 |
| MIT-BIH24B | ta sama walidacja |0.800683|0.556844|0.614498|CNN+RR,wybrany |
| MIT-BIH25B | ta sama walidacja |0.829990|0.486294|0.599149|F-weight×0.5,nie zastąpił24B |
| MIT-BIH26 | finalny test,23392beaty |0.836825|0.618694|0.690384|wyłącznie24B,argmax,zamknięty |

Phase25A: progiF nie dały użytecznego kompromisu; brak nowej reguły.25B zmniejszył FP F2612→1236 przyTP6/38, ale pogorszyłQ i MacroF1. W26 F TP205/FP154/FN158;362/363 rzeczywistychF należy do jednej grupy, więc lepsza liczba nie dowodzi szerokiej generalizacji. W19 SBRAD precision42.71%/recall65.08%; nie eliminujemy false alarms przez ukryte abstencje. SR nie oznacza zdrowego ECG.

Macro AUPRC=średnia nieinterpolowanych Average Precision OVR, klasy stałe. Brak mianownika→null, nie fikcyjne zero. Wagi klas liczone tylko na treningu, balancing wyłącznie w treningu po patient separation. Nigdy nie traktować beatów jednego pacjenta jako niezależnych osób.

Testy końcowe zostały wykorzystane. Nie wybierać na ich podstawie nowych modeli, progów lub cech. MIT-BIH24A miał udokumentowany QC-odczyt9testowych shardów, bez uczenia/scalowania/modelowej oceny/selekcji; nie deklarujemy pierwotnie nigdy nieodczytanego testu. Późniejszy guard i26pre-evaluation manifest zachowują audyt procedury, z jawnym incydentem.

Szczegóły wdrożeniowe i ścieżki checkpointów: `STREAMLIT_INTEGRATION_HANDOFF.md`. Integracja tych konkretnych checkpointów ze Streamlit z osobnego origin/main nie jest ukończona; nie neguje to istnienia aplikacji i innych modeli na main. Zestawienia mają charakter badań prototypu, nie klinicznej walidacji.
