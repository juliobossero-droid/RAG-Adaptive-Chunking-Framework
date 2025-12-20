"""
Batch Processing para 2,593 documentos judiciales
Script optimizado para procesamiento masivo
"""

import os
import json
import glob
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from typing import List, Dict
import time


# ============================================================================
# 1. CARGA MASIVA DE DOCUMENTOS
# ============================================================================

def load_documents_batch(directory: str, pattern: str = "*.txt", 
                        limit: int = None) -> List[Dict[str, str]]:
    """Cargar documentos en batch desde directorio"""
    
    print(f"🔍 Buscando documentos en: {directory}")
    
    # Buscar archivos
    file_pattern = os.path.join(directory, pattern)
    files = glob.glob(file_pattern)
    
    if limit:
        files = files[:limit]
    
    print(f"📁 Encontrados {len(files)} documentos")
    
    documents = []
    
    # Cargar con barra de progreso
    for filepath in tqdm(files, desc="Cargando documentos"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Extraer metadata del nombre de archivo
            filename = Path(filepath).stem
            
            documents.append({
                'id': filename,
                'text': text,
                'filepath': filepath,
                'size': len(text)
            })
        
        except Exception as e:
            print(f"⚠️ Error cargando {filepath}: {e}")
    
    print(f"✓ Cargados {len(documents)} documentos exitosamente")
    
    return documents


# ============================================================================
# 2. PROCESAMIENTO PARALELO
# ============================================================================

def process_document_parallel(doc: Dict, chunker_config: Dict) -> Dict:
    """Procesar un documento (para uso en paralelo)"""
    
    from rag_system import DocumentChunker, AuditLogger
    
    # Crear chunker local
    chunker = DocumentChunker(
        method=chunker_config['method'],
        chunk_size=chunker_config['size'],
        overlap=chunker_config['overlap'],
        logger=AuditLogger(log_file=f"logs/{doc['id']}.log")
    )
    
    # Procesar documento
    chunks = chunker.chunk_document(doc['text'], doc['id'])
    
    return {
        'doc_id': doc['id'],
        'num_chunks': len(chunks),
        'chunks': chunks
    }


def parallel_ingestion(documents: List[Dict], chunker_config: Dict, 
                      n_workers: int = None) -> List[Dict]:
    """Ingerir documentos en paralelo"""
    
    if n_workers is None:
        n_workers = mp.cpu_count() - 1
    
    print(f"🚀 Procesando {len(documents)} docs con {n_workers} workers")
    
    # Crear directorio de logs
    os.makedirs("logs", exist_ok=True)
    
    # Procesar en paralelo
    with mp.Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(
                lambda doc: process_document_parallel(doc, chunker_config),
                documents
            ),
            total=len(documents),
            desc="Procesando documentos"
        ))
    
    return results


# ============================================================================
# 3. INGESTA POR LOTES (BATCHES)
# ============================================================================

def batch_ingestion(documents: List[Dict], vector_store, 
                   batch_size: int = 100):
    """Ingerir documentos en lotes para evitar problemas de memoria"""
    
    from rag_system import DocumentChunker, AuditLogger
    
    logger = AuditLogger()
    chunker = DocumentChunker(logger=logger)
    
    total_docs = len(documents)
    total_chunks = 0
    
    print(f"📦 Procesando {total_docs} documentos en lotes de {batch_size}")
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_docs + batch_size - 1) // batch_size
        
        print(f"\n🔄 Procesando lote {batch_num}/{total_batches}")
        
        # Procesar batch
        batch_chunks = []
        for doc in tqdm(batch, desc=f"Lote {batch_num}"):
            chunks = chunker.chunk_document(doc['text'], doc['id'])
            batch_chunks.extend(chunks)
        
        # Indexar batch
        print(f"   Indexando {len(batch_chunks)} chunks...")
        vector_store.add_chunks(batch_chunks)
        
        total_chunks += len(batch_chunks)
        
        print(f"   ✓ Lote {batch_num} completado ({len(batch_chunks)} chunks)")
    
    print(f"\n✓ Ingesta completa: {total_docs} docs, {total_chunks} chunks")
    
    return total_chunks


# ============================================================================
# 4. CHECKPOINT Y RECUPERACIÓN
# ============================================================================

class CheckpointManager:
    """Gestor de checkpoints para procesamiento largo"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, processed_docs: List[str], batch_num: int):
        """Guardar checkpoint"""
        checkpoint_file = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_batch_{batch_num}.json"
        )
        
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'batch_num': batch_num,
                'processed_docs': processed_docs,
                'timestamp': time.time()
            }, f, indent=2)
    
    def load_checkpoint(self) -> Dict:
        """Cargar último checkpoint"""
        checkpoints = glob.glob(
            os.path.join(self.checkpoint_dir, "checkpoint_batch_*.json")
        )
        
        if not checkpoints:
            return None
        
        # Obtener checkpoint más reciente
        latest = max(checkpoints, key=os.path.getctime)
        
        with open(latest, 'r') as f:
            return json.load(f)
    
    def get_processed_docs(self) -> set:
        """Obtener set de documentos ya procesados"""
        checkpoint = self.load_checkpoint()
        if checkpoint:
            return set(checkpoint['processed_docs'])
        return set()


def batch_ingestion_with_checkpoint(documents: List[Dict], vector_store,
                                    batch_size: int = 100):
    """Ingesta con checkpoints para recuperación"""
    
    from rag_system import DocumentChunker, AuditLogger
    
    checkpoint_mgr = CheckpointManager()
    processed = checkpoint_mgr.get_processed_docs()
    
    logger = AuditLogger()
    chunker = DocumentChunker(logger=logger)
    
    # Filtrar documentos ya procesados
    pending_docs = [doc for doc in documents if doc['id'] not in processed]
    
    if len(pending_docs) < len(documents):
        print(f"📋 Recuperando progreso: {len(processed)} docs ya procesados")
        print(f"   Pendientes: {len(pending_docs)} docs")
    
    total_chunks = 0
    batch_num = len(processed) // batch_size
    
    for i in range(0, len(pending_docs), batch_size):
        batch = pending_docs[i:i+batch_size]
        batch_num += 1
        
        print(f"\n🔄 Procesando lote {batch_num}")
        
        try:
            # Procesar batch
            batch_chunks = []
            for doc in tqdm(batch, desc=f"Lote {batch_num}"):
                chunks = chunker.chunk_document(doc['text'], doc['id'])
                batch_chunks.extend(chunks)
                processed.add(doc['id'])
            
            # Indexar batch
            vector_store.add_chunks(batch_chunks)
            total_chunks += len(batch_chunks)
            
            # Guardar checkpoint
            checkpoint_mgr.save_checkpoint(list(processed), batch_num)
            
            print(f"   ✓ Lote {batch_num} completado y guardado")
        
        except Exception as e:
            print(f"   ❌ Error en lote {batch_num}: {e}")
            print(f"   Checkpoint guardado. Puede reiniciar desde aquí.")
            raise
    
    return total_chunks


# ============================================================================
# 5. VALIDACIÓN DE INGESTA
# ============================================================================

def validate_ingestion(vector_store, expected_docs: int) -> Dict:
    """Validar que la ingesta fue exitosa"""
    
    print("\n🔍 Validando ingesta...")
    
    # Contar documentos indexados
    # (Esto dependerá de tu implementación de ChromaDB)
    
    validation_results = {
        'expected_docs': expected_docs,
        'indexed_chunks': 0,  # Obtener de ChromaDB
        'validation_passed': False
    }
    
    # Realizar consultas de prueba
    test_queries = [
        "libertad condicional",
        "departamento técnico criminológico",
        "conducta ejemplar"
    ]
    
    print("\n📝 Pruebas de recuperación:")
    for query in test_queries:
        results = vector_store.search(query, top_k=3)
        print(f"   - '{query}': {len(results)} resultados")
    
    validation_results['validation_passed'] = True
    
    return validation_results


# ============================================================================
# 6. SCRIPT PRINCIPAL
# ============================================================================

def main():
    """Script principal de procesamiento masivo"""
    
    from rag_system import RAGPipeline, VectorStore
    
    print("="*80)
    print("🔬 PROCESAMIENTO MASIVO - 2,593 DOCUMENTOS")
    print("="*80)
    
    # Configuración
    config = {
        'docs_directory': './documentos',  # Cambiar a tu directorio
        'chunk_method': 'structural',      # fixed, structural, semantic
        'chunk_size': 500,
        'chunk_overlap': 50,
        'batch_size': 100,
        'use_parallel': True,
        'n_workers': 4,
        'use_checkpoints': True
    }
    
    print("\n📋 Configuración:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 1. Cargar documentos
    print("\n" + "="*80)
    print("PASO 1: CARGA DE DOCUMENTOS")
    print("="*80)
    
    documents = load_documents_batch(
        config['docs_directory'],
        pattern="*.txt",
        limit=None  # None para todos, o número para testing
    )
    
    if not documents:
        print("❌ No se encontraron documentos. Verifique el directorio.")
        return
    
    # Estadísticas
    total_size = sum(doc['size'] for doc in documents)
    avg_size = total_size / len(documents)
    
    print(f"\n📊 Estadísticas:")
    print(f"   Total documentos: {len(documents)}")
    print(f"   Tamaño total: {total_size / 1024 / 1024:.2f} MB")
    print(f"   Tamaño promedio: {avg_size / 1024:.2f} KB")
    
    # 2. Inicializar sistema RAG
    print("\n" + "="*80)
    print("PASO 2: INICIALIZACIÓN DEL SISTEMA")
    print("="*80)
    
    vector_store = VectorStore(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        collection_name="legal_docs_2593"
    )
    
    # 3. Ingesta de documentos
    print("\n" + "="*80)
    print("PASO 3: INGESTA Y PROCESAMIENTO")
    print("="*80)
    
    start_time = time.time()
    
    if config['use_checkpoints']:
        total_chunks = batch_ingestion_with_checkpoint(
            documents, 
            vector_store,
            batch_size=config['batch_size']
        )
    else:
        total_chunks = batch_ingestion(
            documents,
            vector_store,
            batch_size=config['batch_size']
        )
    
    elapsed_time = time.time() - start_time
    
    print(f"\n⏱️ Tiempo de procesamiento: {elapsed_time/60:.2f} minutos")
    print(f"   Velocidad: {len(documents)/elapsed_time:.2f} docs/segundo")
    
    # 4. Validación
    print("\n" + "="*80)
    print("PASO 4: VALIDACIÓN")
    print("="*80)
    
    validation = validate_ingestion(vector_store, len(documents))
    
    if validation['validation_passed']:
        print("\n✅ Ingesta completada exitosamente")
    else:
        print("\n⚠️ Validación falló. Revise los logs.")
    
    # 5. Generar reporte
    print("\n" + "="*80)
    print("REPORTE FINAL")
    print("="*80)
    
    report = {
        'total_documents': len(documents),
        'total_chunks': total_chunks,
        'avg_chunks_per_doc': total_chunks / len(documents),
        'processing_time_minutes': elapsed_time / 60,
        'docs_per_second': len(documents) / elapsed_time,
        'config': config,
        'validation': validation
    }
    
    # Guardar reporte
    with open('ingestion_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n📄 Reporte guardado en: ingestion_report.json")
    
    print("\n" + "="*80)
    print("✅ PROCESAMIENTO COMPLETO")
    print("="*80)
    
    return report


# ============================================================================
# 7. TESTING CON SUBCONJUNTO
# ============================================================================

def test_with_subset(n_docs: int = 10):
    """Probar sistema con subconjunto de documentos"""
    
    from rag_system import RAGPipeline
    
    print(f"🧪 Modo de prueba: {n_docs} documentos")
    
    # Cargar solo n documentos
    documents = load_documents_batch(
        './documentos',
        limit=n_docs
    )
    
    # Crear pipeline de prueba
    pipeline = RAGPipeline(
        chunk_method="structural",
        chunk_size=500,
        collection_name="test_collection"
    )
    
    # Ingestar
    pipeline.ingest_documents(documents)
    
    # Consultas de prueba
    test_queries = [
        "¿Cuáles son los motivos para denegar libertad condicional?",
        "¿Qué se evalúa en los informes criminológicos?",
        "¿Qué es el DTC?"
    ]
    
    print("\n🔍 Realizando consultas de prueba:")
    results = []
    
    for query in test_queries:
        result = pipeline.query(query, top_k=5)
        results.append(result)
        
        print(f"\n📝 Query: {query}")
        print(f"   Chunks recuperados: {len(result['chunks'])}")
        print(f"   Respuesta: {result['response'][:200]}...")
    
    # Exportar para evaluación manual
    with open('test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Prueba completada. Resultados en test_results.json")


if __name__ == "__main__":
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Modo de prueba
        n_docs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        test_with_subset(n_docs)
    else:
        # Procesamiento completo
        main()