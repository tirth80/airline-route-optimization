"""
RAG Quality Check DAG

Schedule: Daily at 6:00 AM
Purpose: Test RAG system health and retrieval quality
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys

default_args = {
    'owner': 'tirth',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'rag_quality_check',
    default_args=default_args,
    description='Daily RAG system health check',
    schedule_interval='0 6 * * *',  # 6:00 AM daily
    catchup=False,
    tags=['rag', 'quality', 'monitoring']
)


def check_vector_store(**context):
    """Check ChromaDB health"""
    sys.path.insert(0, '/opt/airflow/project')
    
    from rag.vectorstore.chroma_store import FlightKnowledgeStore
    
    store = FlightKnowledgeStore()
    count = store.count()
    
    print(f"Vector store document count: {count}")
    
    if count < 10:
        raise Exception(f"Vector store has too few documents: {count}")
    
    return count


def test_retrieval(**context):
    """Test retrieval with sample queries"""
    sys.path.insert(0, '/opt/airflow/project')
    
    from rag.vectorstore.chroma_store import FlightKnowledgeStore
    
    store = FlightKnowledgeStore()
    
    test_queries = [
        "best airline",
        "JFK delays",
        "cost of delays"
    ]
    
    results = {}
    for query in test_queries:
        search_results = store.search(query, top_k=3)
        results[query] = len(search_results)
        print(f"Query '{query}': {len(search_results)} results")
    
    return results


def test_rag_pipeline(**context):
    """Test full RAG pipeline"""
    sys.path.insert(0, '/opt/airflow/project')
    
    from rag.pipeline import FlightDelayRAG
    
    rag = FlightDelayRAG()
    
    test_questions = [
        "Which airline is best for on-time flights?",
        "What time should I fly to avoid delays?"
    ]
    
    for q in test_questions:
        result = rag.ask(q)
        
        if not result['answer']:
            raise Exception(f"Empty answer for: {q}")
        
        print(f"Q: {q}")
        print(f"A: {result['answer'][:100]}...")
        print(f"Sources: {len(result['sources'])}")
        print("---")
    
    return True


def generate_report(**context):
    """Generate daily health report"""
    ti = context['ti']
    
    doc_count = ti.xcom_pull(task_ids='check_vector_store')
    retrieval_results = ti.xcom_pull(task_ids='test_retrieval')
    
    report = f"""
    ========================================
    RAG DAILY HEALTH REPORT
    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    ========================================
    
    Vector Store Documents: {doc_count}
    
    Retrieval Test Results:
    {retrieval_results}
    
    Status: HEALTHY
    ========================================
    """
    
    print(report)
    return report


# Tasks
task_check_store = PythonOperator(
    task_id='check_vector_store',
    python_callable=check_vector_store,
    dag=dag
)

task_test_retrieval = PythonOperator(
    task_id='test_retrieval',
    python_callable=test_retrieval,
    dag=dag
)

task_test_pipeline = PythonOperator(
    task_id='test_rag_pipeline',
    python_callable=test_rag_pipeline,
    dag=dag
)

task_report = PythonOperator(
    task_id='generate_report',
    python_callable=generate_report,
    dag=dag
)

# Dependencies
task_check_store >> task_test_retrieval >> task_test_pipeline >> task_report

