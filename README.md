# NLP Project: Enron Social Network Analysis

## Opis

„Analiza sieci znajomości i powiązań na podstawie komunikacji tekstowej.”  
Głównym celem jest porównanie jawnej sieci komunikacji e-mailowej z siecią wzbogaconą o informacje wydobyte metodami NLP/NER z treści wiadomości.

## Plan pracy

1. Wczytanie i oczyszczenie podzbioru Enron Email Corpus → preprocessing  
2. Budowa grafu jawnego na podstawie metadanych `sender → recipient`  
3. Ekstrakcja osób wspominanych w treści e-maili z użyciem NER  
4. Budowa grafu wzbogaconego o ukryte powiązania  
5. Obliczenie metryk SNA, szczególnie betweenness centrality  
6. Przygotowanie tabel i statycznych wizualizacji do raportu  

## Struktura projektu

```text
.
├── src/
│   ├── config.py            # ścieżki (data/, outputs/)
│   ├── io/                  # wczytywanie CSV, próbki
│   ├── preprocess/          # parsowanie wiadomości, filtr Enron, daty
│   ├── nlp/                 # spaCy NER, klasyfikacja relacji, aliasy
│   ├── graphs/              # graf metadanych, graf NER, porównanie, lista węzłów
│   ├── analysis/            # metryki SNA, wizualizacja statyczna
│   ├── pipeline/            # pełna orkiestracja do krawędzi CSV
│   └── utils/               # pomocnicze (np. normalizacja nazw)
├── data/
│   ├── raw/                 # oryginalny CSV z Enron Email Corpus
│   └── processed/           # próbki i dane po preprocessingu
├── outputs/
│   ├── figures/
│   └── tables/
├── README.md
└── requirements.txt
```

CLI jest wbudowane w moduły (`if __name__ == "__main__"`). Uruchamiasz je jako moduły z katalogu głównego repo, np. `python3 -m src.analysis.analyze`.

| Zadanie | Moduł |
|--------|--------|
| Próbka z CSV | `src.io.load_data` |
| Preprocessing | `src.preprocess.preprocessing` |
| Graf metadanych | `src.graphs.graph_builder` |
| Graf NER | `src.graphs.graph_ner_builder` |
| Metryki / ranking | `src.analysis.analyze` |
| Porównanie grafów | `src.graphs.graph_comparison` |
| Lista węzłów | `src.graphs.node_list` |
| Pipeline NER + krawędzie | `src.pipeline.full_pipeline` |

## Wymagania

- Python 3.10+  
- Wirtualne środowisko `.venv`  
- Biblioteki z `requirements.txt`  

## Instalacja

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Uruchamianie modułów (`python -m`)

Wszystkie komendy poniżej zakładają **bieżący katalog = katalog główny repozytorium**, żeby Python widział pakiet `src`.

### Dane

Do `data/raw/` należy dodać plik CSV z podzbiorem Enron Email Corpus.  
Argument `--nrows` w `load_data` wybiera pierwsze N wierszy z pliku CSV — wygodne do szybkich testów technicznych, ale **nie gwarantuje** próbki z konkretnego przedziału czasowego.

Filtrowanie po dacie odbywa się w preprocessingu, bo data wiadomości jest parsowana z nagłówka `Date` wewnątrz surowej kolumny `message`.

### Przykładowy przepływ

Wczytanie małej próbki:

```bash
python3 -m src.io.load_data --nrows 100 --output data/processed/emails_sample.csv
```

Parsowanie nagłówków i czyszczenie treści:

```bash
python3 -m src.preprocess.preprocessing --input data/processed/emails_sample.csv --output data/processed/emails_preprocessed.csv
```

Parsowanie z oknem czasowym (można podać sam `--start-date` lub sam `--end-date`):

```bash
python3 -m src.preprocess.preprocessing \
  --input data/raw/emails.csv \
  --output data/processed/emails_preprocessed.csv \
  --start-date 2001-01-01 \
  --end-date 2001-03-31
```

Dla analizy konkretnego okresu najlepiej jako `--input` podać pełny plik `data/raw/emails.csv`, a nie próbkę z `--nrows`.

Graf jawny `sender → recipient`:

```bash
python3 -m src.graphs.graph_builder --input data/processed/emails_preprocessed.csv --output outputs/tables/metadata_edges.csv
```

Metryki grafu (betweenness, stopnie ważone):

```bash
python3 -m src.analysis.analyze \
  --edges outputs/tables/metadata_edges.csv \
  --ranking-output outputs/tables/metadata_node_metrics.csv \
  --summary-output outputs/tables/metadata_graph_summary.csv
```

Graf NER z osobnego CSV (np. po zapisaniu ramki z kolumną `mentioned_people` — lista musi być wczytana jako lista, nie jako surowy string z CSV):

```bash
python3 -m src.graphs.graph_ner_builder --input outputs/data/ner_input.csv --output outputs/tables/ner_edges.csv
```

Porównanie grafów metadanych vs NER (hidden brokers):

```bash
python3 -m src.graphs.graph_comparison
```

Lista węzłów z pliku krawędzi:

```bash
python3 -m src.graphs.node_list --input outputs/tables/metadata_edges.csv --output outputs/tables/nodes_list.csv
```

### Pipeline z NER (spaCy `en_core_web_sm`)

Jednym poleceniem: preprocessing w pamięci, NER, zapis `metadata_edges.csv` i `ner_edges.csv`:

```bash
python3 -m src.pipeline.full_pipeline --nrows 1000
```

Domyślnie `--nrows` wynosi **1000**. Użyj `--nrows 0` aby wczytać **cały** plik wejściowy (może być wolne i pamięciożerne).

## Autorzy

Blagoja Mladenov, Jakub Kordel, Miłosz Andruczyk
