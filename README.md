Framework para la Selección Adaptativa y Trazable de Estrategias de Segmentación en Sistemas RAG
Descripción
Este repositorio contiene la Prueba de Concepto (PoC) y los desarrollos experimentales asociados a la tesis doctoral:

"Diseño y evaluación de un framework para la selección adaptativa y trazable de estrategias de segmentación documental en sistemas RAG". 【1-04de38】

La investigación se desarrolla en el marco del Doctorado en Ingeniería con mención en Tecnología de la Información de la Universidad Nacional de La Matanza (UNLaM). 【1-04de38】

Objetivo
Diseñar y evaluar un framework capaz de seleccionar, de manera anticipada, adaptativa y trazable, la estrategia de segmentación más adecuada para cada documento dentro de un sistema Retrieval-Augmented Generation (RAG), utilizando características estructurales y semánticas calculables antes de realizar la segmentación. 【1-04de38】

Pregunta de Investigación
¿En qué condiciones la selección anticipada de estrategias de segmentación mejora la recuperación de información respecto de una única estrategia global y qué beneficios adicionales aportan mecanismos de decisión de mayor complejidad, como clasificadores y agentes basados en modelos de lenguaje? 【1-04de38】

Arquitectura General
Documento
    │
    ▼
Extracción de Descriptores
    │
    ▼
Perfil Documental
    │
    ▼
Selector de Estrategia
    │
    ▼
Segmentación
    │
    ▼
Indexación Vectorial
    │
    ▼
Recuperación
    │
    ▼
Registro Trazable
La arquitectura busca separar componentes independientes del dominio de aquellos específicos del contexto documental evaluado. 【1-04de38】【2-3d62b2】

Estrategias de Segmentación Implementadas
Actualmente se incluyen:

Segmentación fija
Segmentación fija sin solapamiento
Segmentación semántica
Segmentación basada en grafos (Louvain)
Segmentación estructural jurídica
Empaquetado por oraciones
【1-04de38】【2-3d62b2】

Tecnologías
Python 3.13
ChromaDB
Sentence Transformers
Ollama
LangChain
Pytest
YAML
Modelos Embeddings Multilingües
【2-3d62b2】

Corpus Experimental
La evaluación se realiza sobre un corpus de resoluciones judiciales en español utilizado para investigación académica. Debido a restricciones de confidencialidad y protección de datos, los documentos originales no forman parte de este repositorio. 【1-04de38】【2-3d62b2】

Estado Actual
La PoC demuestra la viabilidad técnica del flujo completo:

Caracterización documental.
Selección basada en reglas.
Segmentación.
Indexación.
Recuperación.
Registro trazable de decisiones.
El objetivo de esta etapa es validar la factibilidad de la arquitectura, no demostrar superioridad estadística. 【2-3d62b2】

Estructura del Proyecto
.
├── chunklab/
├── config/
├── corpus/
├── docs/
├── experiments/
├── tests/
├── logs/
├── results/
└── README.md
Resultados Previos
Investigaciones preliminares asociadas al proyecto fueron presentadas en eventos científicos como:

JAIIO
CAEDI
CACIC
CONAIISI
ReDDI
Los resultados obtenidos sugieren que el rendimiento de las estrategias de segmentación depende de las características documentales y que no existe una estrategia universalmente superior para todos los escenarios. 【1-04de38】

Reproducibilidad
El proyecto sigue principios de investigación reproducible:

Configuraciones versionadas.
Registro de parámetros.
Trazabilidad de decisiones.
Control de versiones.
Pruebas automatizadas.
Registro de resultados experimentales.
【1-04de38】【2-3d62b2】

Aspectos Éticos
Los expedientes judiciales utilizados para investigación están sujetos a restricciones de confidencialidad. Este repositorio no publica documentos originales, datos sensibles ni información personal. 【1-04de38】【2-3d62b2】

Autor
Julio César Bossero
Doctorando en Ingeniería con mención en Tecnología de la Información
Universidad Nacional de La Matanza (UNLaM) 【1-04de38】

Licencia
MIT License

Citación
Si utiliza este trabajo en investigaciones académicas, por favor cite:

Bossero, J. C. Framework para la selección adaptativa y trazable de estrategias de segmentación documental en sistemas RAG. Universidad Nacional de La Matanza.
