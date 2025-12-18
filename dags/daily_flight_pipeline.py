"""
Daily Flight Pipeline DAG

Schedule: Daily at 2:00 AM
Purpose: Fetch flight data, update knowledge base, refresh embeddings
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
import os

# Default arguments
default_args = {
    'owner': 'tirth',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'daily_flight_pipeline',
    default_args=default_args,
    description='Daily pipeline to update flight knowledge base',
    schedule_interval='0 2 * * *',  # 2:00 AM daily
    catchup=False,
    tags=['flight', 'rag', 'daily']
)


def fetch_flight_data(**context):
    """Fetch latest flight data from AviationStack API"""
    sys.path.insert(0, '/opt/airflow/project')
    
    from data.api.aviation_stack import AviationStackAPI
    
    api = AviationStackAPI()
    summaries = api.get_all_airports_summary()
    
    print(f"Fetched data for {len(summaries)} airports")
    for summary in summaries:
        print(f"  {summary['airport']}: {summary['delay_rate']:.1%} delay rate")
    
    return summaries


def update_current_knowledge(**context):
    """Update current status documents in knowledge base"""
    sys.path.insert(0, '/opt/airflow/project')
    
    from datetime import datetime
    from pathlib import Path
    
    # Get data from previous task
    ti = context['ti']
    summaries = ti.xcom_pull(task_ids='fetch_flight_data')
    
    # Create current status document
    content = f"""# Current Flight Status

## Last Updated
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Airport Delay Summary (Today)

| Airport | Delay Rate | Total Flights | Delayed |
|---------|------------|---------------|---------|
"""
    
    for s in summaries:
        content += f"| {s['airport']} | {s['delay_rate']:.1%} | {s['total_flights']} | {s['delayed_flights']} |\n"
    
    # Add comparison note
    content += """
## Comparison with Historical (2019)

Historical average delay rate: 18.9%

"""
    
    # Save to knowledge base
    output_path = Path('/opt/airflow/project/knowledge_base/current/today_status.md')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    
    print(f"Updated: {output_path}")
    return str(output_path)


def refresh_embeddings(**context):
    """Refresh vector store with updated documents"""
    sys.path.insert(0, '/opt/airflow/project')
    
    from rag.chunking.text_chunker import TextChunker
    from rag.vectorstore.chroma_store import FlightKnowledgeStore
    
    chunker = TextChunker()
    store = FlightKnowledgeStore()
    
    # Chunk all documents (historical + current)
    historical_chunks = chunker.chunk_directory('/opt/airflow/project/knowledge_base/historical')
    current_chunks = chunker.chunk_directory('/opt/airflow/project/knowledge_base/current')
    
    all_chunks = historical_chunks + current_chunks
    
    # Rebuild index
    store.clear()
    store.add_chunks(all_chunks)
    
    print(f"Refreshed embeddings: {len(all_chunks)} total chunks")
    return len(all_chunks)


def validate_pipeline(**context):
    """Validate the pipeline ran successfully"""
    sys.path.insert(0, '/opt/airflow/project')
    
    from rag.pipeline import FlightDelayRAG
    
    rag = FlightDelayRAG()
    
    # Test query
    result = rag.ask("What is the current delay status?")
    
    print(f"Validation query successful")
    print(f"Sources: {result['sources']}")
    print(f"Answer preview: {result['answer'][:200]}...")
    
    return True


# Task definitions
task_fetch = PythonOperator(
    task_id='fetch_flight_data',
    python_callable=fetch_flight_data,
    dag=dag
)

task_update_kb = PythonOperator(
    task_id='update_current_knowledge',
    python_callable=update_current_knowledge,
    dag=dag
)

task_refresh = PythonOperator(
    task_id='refresh_embeddings',
    python_callable=refresh_embeddings,
    dag=dag
)

task_validate = PythonOperator(
    task_id='validate_pipeline',
    python_callable=validate_pipeline,
    dag=dag
)

# Task dependencies
task_fetch >> task_update_kb >> task_refresh >> task_validate

