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
| Klasyfikacja relacji LLM | `src.nlp.relation_classifier` |
| Pipeline NER + krawędzie | `src.pipeline.full_pipeline` |

## Wymagania

- Python 3.10+  
- Wirtualne środowisko `.venv`  
- Biblioteki z `requirements.txt`  
- Ollama do lokalnej klasyfikacji relacji LLM  

## Instalacja

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Ollama instalujesz poza środowiskiem Pythona. Na Linuksie najprościej:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Po instalacji pobierz model. Dla tego projektu dobrym punktem startowym jest Llama 3:

```bash
ollama pull llama3
```

Jeżeli komputer ma mniej zasobów, można użyć mniejszego modelu, np.:

```bash
ollama pull llama3.2:3b
```

## Uruchamianie modułów (`python -m`)

Wszystkie komendy poniżej zakładają **bieżący katalog = katalog główny repozytorium**

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

### Semantyczna klasyfikacja relacji lokalnym LLM

Klasyfikuje tylko podzbiór najważniejszych krawędzi, czyli pary
`source -> target` o największej wadze. Moduł zbiera kilka przykładowych wiadomości
dla każdej krawędzi, wysyła je do lokalnego modelu przez Ollama i zapisuje tabelę z
etykietą relacji, pewnością oraz krótkim uzasadnieniem.

Przykładowe uruchomienie dla grafu jawnego:

```bash
ollama serve
ollama pull llama3

python3 -m src.nlp.relation_classifier \
  --edges outputs/tables/metadata_edges.csv \
  --emails data/processed/emails_preprocessed.csv \
  --output outputs/tables/metadata_llm_relation_labels.csv \
  --top-k 20 \
  --model llama3
```

Flaga 'dry-run' sprawdza czy moduł poprawnie znajduje kontekst dla krawędzi:

```bash
python3 -m src.nlp.relation_classifier \
  --edges outputs/tables/metadata_edges_JAWNY.csv \
  --emails data/processed/emails_preprocessed_enron_filter.csv \
  --output outputs/tables/metadata_llm_relation_labels_dry_run.csv \
  --top-k 10 \
  --dry-run
```

Obsługiwane etykiety: `formal_sluzbowa`, `wspolpraca`, `konflikt`,
`delegowanie_zadan`, `informacyjna`, `towarzyska`, `niejednoznaczna`.

Lista węzłów z pliku krawędzi:

```bash
python3 -m src.graphs.node_list --input outputs/tables/metadata_edges.csv --output outputs/tables/nodes_list.csv
```

### Pipeline z NER (spaCy `en_core_web_sm`)

Full pipeline: preprocessing danych email, budowa grafu NER, budowa grafu jawnego z metadanch, zapis `metadata_edges.csv`,
`ner_edges.csv` oraz wzbogaconego pliku maili `data/processed/emails_with_ner.csv`:

```bash
python3 -m src.pipeline.full_pipeline --nrows 1000
```

Domyślnie `--nrows` wynosi **1000**. Użyj `--nrows 0` aby wczytać **cały** plik wejściowy.

## Rekomendowany pipeline end-to-end

Poniższe komendy uruchamiają pełny przepływ projektu: graf jawny, graf NER, metryki
i klasyfikację semantyczną relacji lokalnym LLM.

1. Aktywuj środowisko:

```bash
source .venv/bin/activate
```

2. Wygeneruj krawędzie grafu jawnego i NER:

```bash
python3 -m src.pipeline.full_pipeline \
  --input data/raw/emails.csv \
  --nrows 1000 \
  --processed-output data/processed/emails_with_ner.csv
```

Wyniki:

- `outputs/tables/metadata_edges.csv` — graf jawny z metadanych `sender -> recipient`.
- `outputs/tables/ner_edges.csv` — graf niejawny z osób wspominanych w treści.
- `data/processed/emails_with_ner.csv` — maile po preprocessingu z kolumną `mentioned_people`; ten plik jest potrzebny do klasyfikacji krawędzi NER.

3. Policz metryki grafu jawnego:

```bash
python3 -m src.analysis.analyze \
  --edges outputs/tables/metadata_edges.csv \
  --ranking-output outputs/tables/metadata_node_metrics.csv \
  --summary-output outputs/tables/metadata_graph_summary.csv
```

4. Policz metryki grafu NER:

```bash
python3 -m src.analysis.analyze \
  --edges outputs/tables/ner_edges.csv \
  --ranking-output outputs/tables/ner_node_metrics.csv \
  --summary-output outputs/tables/ner_graph_summary.csv
```

5. Porównaj graf jawny i NER:

```bash
python3 -m src.graphs.graph_comparison
```

6. Uruchom lokalny serwer Ollama w osobnym terminalu:

```bash
ollama serve
```
  
Jeżeli Ollama działa jako usługa systemowa, ten krok może nie być potrzebny. Możesz
sprawdzić dostępne modele poleceniem:

```bash
ollama list
```

7. Sprawdź klasyfikator bez odpytania modelu:

```bash
python3 -m src.nlp.relation_classifier \
  --edges outputs/tables/metadata_edges.csv \
  --emails data/processed/emails_with_ner.csv \
  --output outputs/tables/metadata_llm_relation_labels_dry_run.csv \
  --top-k 10 \
  --dry-run
```

8. Sklasyfikuj najważniejsze krawędzie grafu jawnego:

```bash
python3 -m src.nlp.relation_classifier \
  --edges outputs/tables/metadata_edges.csv \
  --emails data/processed/emails_with_ner.csv \
  --output outputs/tables/metadata_llm_relation_labels.csv \
  --top-k 20 \
  --model llama3
```

9. Sklasyfikuj najważniejsze krawędzie grafu NER:

```bash
python3 -m src.nlp.relation_classifier \
  --edges outputs/tables/ner_edges.csv \
  --emails data/processed/emails_with_ner.csv \
  --output outputs/tables/ner_llm_relation_labels.csv \
  --top-k 20 \
  --model llama3
```

Jeżeli używasz innego modelu, zmień `--model`, np.:

```bash
--model llama3.2:3b
```

Domyślny adres Ollama to `http://localhost:11434`. Jeżeli Ollama działa gdzie indziej,
podaj adres jawnie:

```bash
--ollama-host http://localhost:11434
```

Najważniejsze wyniki do raportu:

- `metadata_node_metrics.csv` i `ner_node_metrics.csv` — rankingi węzłów, w tym Betweenness Centrality.
- `hidden_brokers.csv` — kandydaci na ukrytych pośredników informacyjnych.
- `metadata_llm_relation_labels.csv` i `ner_llm_relation_labels.csv` — semantyczne etykiety najważniejszych relacji.

## Autorzy

Blagoja Mladenov, Jakub Kordel, Miłosz Andruczyk
