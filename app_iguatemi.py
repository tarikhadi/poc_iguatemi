import streamlit as st
import chromadb
import json
import os
from openai import OpenAI
from chromadb.utils import embedding_functions
import glob
from datetime import datetime
from typing import Dict, List, Any

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Initialize OpenAI client
client = OpenAI()

SYSTEM_PROMPT = f"""Você é um assistente especializado em contratos do Shopping Center Iguatemi. 

Você lida com dois tipos de consultas:

1. Consultas globais sobre todos os contratos (ex.: datas de vencimento dos contratos, número total de lojas)
Para essas consultas, você receberá metadados pré-processados e trechos relevantes dos contratos.
Forneça informações agregadas com base nos metadados fornecidos.
Seja detalhado, mas conciso.

2. Consultas específicas sobre lojas individuais
Foque nos detalhes do contrato da loja em questão.
Forneça informações precisas apenas a partir dos documentos dessa loja.

Sempre:
Dê respostas diretas e baseadas nos documentos e metadados fornecidos.
Se houver falta de informação ou algo estiver incerto, mencione isso explicitamente.
Formate números e datas de maneira consistente.
Mantenha as respostas curtas e objetivas.

Hoje é {datetime.now().strftime("%d de %B de %Y")}"""

def extract_metadata(content: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metadata from contract JSON."""
    loja = content.get('loja', {})
    contrato = content.get('contratos', [{}])[0] if content.get('contratos') else {}
    
    return {
        "store_name": loja.get('nome_fantasia', ''),
        "cnpj": loja.get('cnpj', ''),
        "contract_number": contrato.get('numero_contrato', ''),
        "store_area": contrato.get('objeto', {}).get('area_privativa', ''),
        "contract_start": contrato.get('vigencia', {}).get('data_inicial', ''),
        "contract_end": contrato.get('vigencia', {}).get('data_final', ''),
        "floor": contrato.get('objeto', {}).get('piso', ''),
        "store_number": contrato.get('objeto', {}).get('loja', '')
    }

def load_documents(directory_path: str):
    """Load documents with enhanced metadata handling."""
    try:
        # Initialize embedding function and collection
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small")
        
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection(
            name="contracts",
            embedding_function=embedding_function
        )
        
        # Clear existing documents if any exist
        if len(collection.get()['ids']) > 0:
            collection.delete(where={"store_name": {"$exists": True}})
        
        # Process each JSON file
        json_files = glob.glob(os.path.join(directory_path, "*.json"))
        documents = []
        metadatas = []
        ids = []
        all_metadata = []  # Store complete metadata for session state
        
        for i, file_path in enumerate(json_files):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = json.load(file)
                
                # Extract metadata
                metadata = extract_metadata(content)
                all_metadata.append(metadata)
                
                # Store the entire JSON as a string
                documents.append(json.dumps(content, ensure_ascii=False))
                metadatas.append(metadata)
                ids.append(f"doc_{i}")
        
        # Add documents to collection
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        return collection, all_metadata
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return None, None

def handle_global_query(query: str, metadata_list: List[Dict[str, Any]], collection) -> tuple:
    """Handle global queries using metadata and selective retrieval."""
    # Pre-process metadata based on query type
    if "venc" in query.lower():  # Contract expiration related
        relevant_data = {
            "contracts": [
                {
                    "store": m["store_name"],
                    "end_date": m["contract_end"]
                } for m in metadata_list if m["contract_end"]
            ]
        }
    elif "área" in query.lower() or "area" in query.lower():  # Area related
        relevant_data = {
            "stores": [
                {
                    "store": m["store_name"],
                    "area": m["store_area"]
                } for m in metadata_list if m["store_area"]
            ]
        }
    else:  # For other global queries, get most relevant documents but limit context
        results = collection.query(
            query_texts=[query],
            n_results=100  # Limit to most relevant documents for general queries
        )
        return results, None
    
    return None, relevant_data

def handle_store_query(query: str, collection) -> tuple:
    """Handle store-specific queries."""
    results = collection.query(
        query_texts=[query],
        n_results=1
    )
    return results, None

def get_chat_response(query: str, context: str, metadata_summary: Dict = None) -> str:
    """Get response from GPT-4o based on query type and available information."""
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        
        # Add metadata summary if available
        if metadata_summary:
            messages.append({
                "role": "user",
                "content": f"Metadata summary: {json.dumps(metadata_summary, ensure_ascii=False)}\n\nQuestion: {query}"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {query}"
            })
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting chat response: {str(e)}")
        return None

def main():
    st.title("Shopping Iguatemi - Smart Contracts Assistant")
    
    # Initialize or get the collection and metadata
    if 'collection' not in st.session_state or 'metadata' not in st.session_state:
        collection, metadata = load_documents("/Users/tarikhadi/Desktop/contratos_json")
        if collection and metadata:
            st.session_state.collection = collection
            st.session_state.metadata = metadata
    
    query = st.text_input("Pergunte o que quiser sobre os contratos:")
    
    if query and st.session_state.get('collection'):
        # Determine query type
        is_global_query = any(keyword in query.lower() for keyword in 
                            ['todos', 'todas', 'quantos', 'quantas', 'total', 
                             'geral', 'shopping', 'contratos'])
        
        if is_global_query:
            results, metadata_summary = handle_global_query(
                query, 
                st.session_state.metadata,
                st.session_state.collection
            )
            
            if metadata_summary:
                response = get_chat_response(query, "", metadata_summary)
            else:
                context = "\n\n".join(results['documents'][0]) if results else ""
                response = get_chat_response(query, context)
        else:
            results, _ = handle_store_query(query, st.session_state.collection)
            context = results['documents'][0][0] if results else ""
            response = get_chat_response(query, context)
        
        if response:
            st.write("Resposta:", response)
            
            # Display source information
            with st.expander("Veja as referências"):
                if results:
                    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                        st.write(f"Loja: {metadata['store_name']}")
                        st.write(f"Número do Contrato: {metadata['contract_number']}")
                        st.write("---")
                elif metadata_summary:
                    st.write("Pergunta respondida usando metadados de todos os contratos")

if __name__ == "__main__":
    main()