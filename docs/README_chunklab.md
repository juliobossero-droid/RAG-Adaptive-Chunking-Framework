# chunklab

Banco de pruebas de estrategias de chunking para RAG sobre sentencias judiciales (fase de
**segmentacion**; la evaluacion de recuperacion y el agente son etapas posteriores).
Marco teorico y justificacion de cada decision: documento "Marco teorico y fundamentacion
metodologica del banco de pruebas de chunking".

## Estrategias (config.yaml)
| nombre | kind | rol |
|---|---|---|
| `fixed` | `fixed_tokens` | estrategia 1: ventanas de tokens con solapamiento |
| `semantic` | `semantic_dp` | estrategia 2: heuristica semantica, optimo exacto por programacion dinamica |
| `louvain` | `louvain_graph` | estrategia 3: comunidades de Louvain (fiel a los articulos) |
| `legal` | `legal_structural` | estrategia 4: AUTOS Y VISTOS / CONSIDERANDO / RESUELVO, con fallback |
| `fixed_nooverlap`, `sentence_pack` | controles | atribucion causal (no son opciones del agente) |
| `louvain_contiguous` | variante | desactivada por defecto |

Todas comparten: mismas oraciones, mismo tokenizador, mismo tope `max_tokens`. Ningun chunk
puede truncarse en el embedding (el pipeline aborta si la configuracion lo permitiera).

## Instalacion (otro equipo)
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pip freeze > requirements.lock                           # guardar junto con los resultados
python -m pytest -q                                      # debe dar todo en verde
```

## Uso
```bash
chunklab diagnose-headings --config config.yaml          # sin modelos: revisa deteccion de encabezados
chunklab run --config config.yaml --dry-run              # valida config y corpus
chunklab run --config config.yaml --limit 3              # PRIMERA CORRIDA REAL: 3 documentos
chunklab inspect --config config.yaml --doc-id <id>      # mirar los chunks a ojo
chunklab run --config config.yaml                        # corpus completo (reanudable)
```
La corrida se puede **interrumpir y reanudar**: cada documento se guarda en `shards/` de forma
atomica y se omite si el hash de configuracion y de documento coinciden. Si cambia la
configuracion, aborta (use un `run.label` nuevo o `--force`).

## Salidas (`runs/<label>/`)
`chunks/<estrategia>.jsonl` (texto, offsets, tokens, metadatos) · `chunk_stats.csv` ·
`doc_stats.csv` (encabezados detectados, fallback, atomizacion) · `issues.csv` ·
`errors.jsonl` · `manifest.json` (config, versiones, huellas del corpus) · `shards/`.

## Lo que hay que hacer ANTES de confiar en los resultados
1. **Primera corrida real con `--limit 3`** y `inspect`: el adaptador del tokenizador fue
   probado con un tokenizador rapido local, pero NO con el modelo real (sin acceso a la red
   en el entorno de desarrollo). Confirmar en el log `max_seq_length` y `limite efectivo`.
2. **`diagnose-headings`** sobre el corpus y revisar ~30 documentos a mano: el detector es una
   heuristica. Mirar el % sin encabezados (los documentos en fallback usan solo la DP).
3. **Validar el segmentador de oraciones** con una muestra anotada (ver marco teorico, sec. 6).
4. **Recalibrar `tau` (Louvain) y `lam` (DP)** con el modelo elegido, SOLO con datos de
   desarrollo; los valores 0,72 y 0,4 vienen de trabajos con otros modelos.
5. Congelar la configuracion (anotar `config_hash` del manifest) antes de abrir el test.

## Decisiones que se apartan de los articulos previos
* Heuristica semantica: programacion dinamica exacta (no voraz) y objetivo ponderado por n.
  La forma de la penalizacion por tamano no estaba especificada: usamos |n-ideal|/(max-min).
* Louvain: se agregan fusion de comunidades chicas y division de las que exceden el tope.
* Estructural: RESUELVO se conserva entero si cabe en el tope; si no, se divide y se marca
  `resuelvo_overflow`. Sin encabezados: fallback a la DP (`fallback: true`).
* Tamano en tokens del modelo (no en caracteres).
