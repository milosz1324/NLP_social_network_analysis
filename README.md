# NLP Project: Enron Social Network Analysis

## Opis

"Analiza sieci znajomości i powiązań na podstawie komunikacji tekstowej."
Głównym celem jest porównanie jawnej sieci komunikacji e-mailowej z siecią wzbogaconą
o informacje wydobyte metodami NLP/NER z treści wiadomości.

## Plan pracy

1. Wczytanie i oczyszczenie podzbioru Enron Email Corpus. -> preprocessing
2. Budowa grafu jawnego na podstawie metadanych `sender -> recipient`.
3. Ekstrakcja osób wspominanych w treści e-maili z użyciem NER.
4. Budowa grafu wzbogaconego o ukryte powiązania.
5. Obliczenie metryk SNA, szczególnie betweenness centrality.
6. Przygotowanie tabel i statycznych wizualizacji do raportu.

## Struktura projektu

```text
.
├── run_pipeline.py
├── data/
│   ├── raw/          # oryginalne pliki danych, np. CSV z Enron Email Corpus
│   └── processed/    # oczyszczone lub pośrednie dane
├── docs/             # dokumentacja projektowa
├── notebooks/        # eksperymenty i eksploracja danych
├── outputs/
│   ├── figures/      # wygenerowane wykresy i grafy
│   └── tables/       # tabele wynikowe do raportu
├── src/              # kod źródłowy projektu
├── README.md
└── requirements.txt
```

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

## Dane

Do katalogu `data/raw/` należy dodać plik CSV z podzbiorem Enron Email Corpus.
Na start najlepiej pracować na mniejszej próbce, żeby szybko iterować nad preprocessingiem i grafami.
Argument `--nrows` w `load_data.py` wybiera pierwsze N wierszy z pliku CSV, więc służy tylko do szybkich testów technicznych. Nie gwarantuje, że w próbce znajdą się wiadomości z konkretnego przedziału czasowego.

Filtrowanie po dacie odbywa się dopiero w preprocessingu, bo data wiadomości jest parsowana z nagłówka `Date` wewnątrz surowej kolumny `message`.

## Kolejny krok

Wczytanie małej próbki do szybkiego testu:

```bash
python3 -m src.load_data --nrows 100 --output data/processed/emails_sample.csv
```

Parsowanie nagłówków i czyszczenie treści:

```bash
python3 -m src.preprocessing --input data/processed/emails_sample.csv --output data/processed/emails_preprocessed.csv
```

Parsowanie tylko dla wybranego przedziału czasowego:

```bash
python3 -m src.preprocessing \
  --input data/raw/emails.csv \
  --output data/processed/emails_preprocessed.csv \
  --start-date 2001-01-01 \
  --end-date 2001-03-31
```

Argumenty `--start-date` i `--end-date` są opcjonalne. Jeśli zostaną podane, preprocessing zachowa tylko wiadomości z datą w wybranym zakresie, włącznie z datą początkową i końcową. Dla analizy konkretnego okresu najlepiej podać jako `--input` pełny plik `data/raw/emails.csv`, a nie próbkę stworzoną przez `--nrows`.

Zbudowanie grafu `sender -> recipient`:

```bash
python3 -m src.graph_builder --input data/processed/emails_preprocessed.csv --output outputs/tables/metadata_edges.csv
```

Analiza metryk grafu:

```bash
python3 -m src.analysis --edges outputs/tables/metadata_edges.csv --ranking-output outputs/tables/metadata_node_metrics.csv --summary-output outputs/tables/metadata_graph_summary.csv
```


## Pipeline z NER i modelem językowym spaCy en_core_web_sm
W katalogu root uruchomić:
python3 run_pipeline.py --nrows 1000

Lub bez podania --nrows, domyślne nrows to 100

## Autorzy

Blagoja Mladenov, Jakub Kordel, Miłosz Andruczyk
