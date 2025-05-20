# main_rag_script.py
import os
import glob
import hashlib
import pickle
import numpy as np
import pdfplumber # For reading PDFs
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity # For finding similar chunks

# --- Configuration ---
# IMPORTANT: Set your GOOGLE_API_KEY environment variable before running the script.
# You can get an API key from Google AI Studio: https://aistudio.google.com/app/apikey


genai.configure(api_key="AIzaSyCD17iXPuCy2zXbFbXVtZRkb6cnUseolOE")

PDF_DIRECTORY = "papers"  # Create a folder named 'pdfs' and put your PDF files there
EMBEDDING_MODEL = "models/text-embedding-004" # Google's text embedding model
GENERATIVE_MODEL = "gemini-2.5-pro-exp-03-25" # Powerful model for generation (Preview version)
# RAG Parameters
CHUNK_SIZE = 1000  # Number of characters per text chunk
CHUNK_OVERLAP = 100  # Number of characters to overlap between chunks
TOP_K_RESULTS = 50  # Number of relevant chunks to retrieve for the LLM
CACHE_FILE = "pdf_embeddings_cache.pkl" # File to cache embeddings

# --- Helper Functions ---

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    print(f"📄 Extracting text from: {pdf_path}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception as e:
        print(f"⚠️ Error extracting text from {pdf_path}: {e}")
        return None

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Splits text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if start >= len(text): # Ensure we don't go past the end if overlap is large
            break
    return chunks

def get_embeddings(texts, model_name=EMBEDDING_MODEL):
    """Generates embeddings for a list of texts using Google AI."""
    print(f"🧠 Generating embeddings using {model_name} for {len(texts)} chunks...")
    if not texts:
        return []
    try:
        # The API can handle a list of texts directly for batch embedding
        result = genai.embed_content(model=model_name,
                                     content=texts,
                                     task_type="RETRIEVAL_DOCUMENT") # or "SEMANTIC_SIMILARITY"
        return [item for item in result['embedding']] # Extracting the list of embeddings
    except Exception as e:
        print(f"🔴 Error generating embeddings: {e}")
        # Handle specific API errors if needed, e.g., rate limits, invalid input
        if "API key not valid" in str(e):
            print("🔴 Please check if your GOOGLE_API_KEY is correct and has permissions.")
        # Fallback or retry logic could be added here
        return [None] * len(texts) # Return None for embeddings if error occurs for a batch

def calculate_pdf_hash(pdf_path):
    """Calculates an MD5 hash for a PDF file to detect changes."""
    hasher = hashlib.md5()
    with open(pdf_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def build_or_load_vector_store(pdf_directory, embedding_model, cache_file=CACHE_FILE):
    """
    Builds a vector store from PDFs in the directory or loads from cache.
    The vector store will be a list of dictionaries, each containing:
    {'text': original_text_chunk, 'embedding': numpy_array_embedding, 'source': pdf_filename}
    """
    vector_store = []
    processed_files_hashes = {} # To store hashes of already processed files

    # Load cache if it exists
    if os.path.exists(cache_file):
        print(f"💾 Loading embeddings from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            vector_store = cached_data.get('vector_store', [])
            processed_files_hashes = cached_data.get('processed_files_hashes', {})
        print(f"Loaded {len(vector_store)} chunks from {len(processed_files_hashes)} cached files.")

    if not os.path.exists(pdf_directory):
        print(f"🔴 Error: PDF directory '{pdf_directory}' not found.")
        return vector_store # Return empty or cached store if dir doesn't exist

    pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
    if not pdf_files:
        print(f"🟡 No PDF files found in '{pdf_directory}'.")
        return vector_store

    new_files_processed = False
    for pdf_path in pdf_files:
        pdf_filename = os.path.basename(pdf_path)
        current_file_hash = calculate_pdf_hash(pdf_path)

        if pdf_filename in processed_files_hashes and processed_files_hashes[pdf_filename] == current_file_hash:
            print(f"🔄 '{pdf_filename}' is unchanged, using cached embeddings.")
            continue # Skip reprocessing if file hasn't changed

        print(f"✨ Processing new or modified file: {pdf_filename}")
        new_files_processed = True
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            continue

        text_chunks = chunk_text(raw_text)
        if not text_chunks:
            print(f"🟡 No text chunks generated for {pdf_filename}.")
            continue

        chunk_embeddings = get_embeddings(text_chunks, embedding_model)

        for chunk, embedding in zip(text_chunks, chunk_embeddings):
            if embedding is not None: # Only add if embedding was successful
                vector_store.append({
                    "text": chunk,
                    "embedding": np.array(embedding), # Ensure it's a NumPy array for similarity calculation
                    "source": pdf_filename
                })
        processed_files_hashes[pdf_filename] = current_file_hash # Update hash for this processed file

    if new_files_processed or not os.path.exists(cache_file): # Save cache if new files processed or cache doesn't exist
        print(f"💾 Saving updated embeddings to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump({'vector_store': vector_store, 'processed_files_hashes': processed_files_hashes}, f)

    return vector_store

def retrieve_relevant_chunks(query, vector_store, embedding_model, top_k=TOP_K_RESULTS):
    """Retrieves the top_k most relevant text chunks from the vector store for a given query."""
    if not vector_store:
        print("🟡 Vector store is empty. Cannot retrieve chunks.")
        return []

    print(f"🔍 Generating embedding for query: '{query}'")
    query_embedding_response = genai.embed_content(model=embedding_model,
                                         content=query,
                                         task_type="RETRIEVAL_QUERY") # Use "RETRIEVAL_QUERY" for query embedding
    query_embedding = np.array(query_embedding_response['embedding'])

    if query_embedding is None or query_embedding.size == 0: # Check if embedding is None or empty
        print("🔴 Failed to generate embedding for the query or received an empty embedding.")
        return []

    # Calculate cosine similarity
    # Ensure all chunk embeddings are numpy arrays and have the same dimension as query_embedding
    valid_items_for_similarity = [item for item in vector_store if isinstance(item['embedding'], np.ndarray) and item['embedding'].shape == query_embedding.shape]
    
    if not valid_items_for_similarity:
        print("🟡 No valid embeddings found in the vector store to compare with.")
        return []
        
    chunk_embeddings_list = [item['embedding'] for item in valid_items_for_similarity]
    similarities = cosine_similarity(query_embedding.reshape(1, -1), np.array(chunk_embeddings_list))[0]
    
    # Sort by similarity (descending) and get indices relative to valid_items_for_similarity
    sorted_indices_in_valid_list = np.argsort(similarities)[::-1]
    
    relevant_chunks = []
    for i in range(min(top_k, len(sorted_indices_in_valid_list))):
        # Get the item from valid_items_for_similarity using the sorted index
        item_index_in_valid_list = sorted_indices_in_valid_list[i]
        selected_item = valid_items_for_similarity[item_index_in_valid_list]
        
        relevant_chunks.append({
            "text": selected_item['text'],
            "source": selected_item['source'],
            "similarity": similarities[item_index_in_valid_list] # Similarity score for this chunk
        })
        print(f"  📚 Retrieved chunk from '{selected_item['source']}' (Similarity: {similarities[item_index_in_valid_list]:.4f})")
        
    return relevant_chunks

def generate_answer_with_llm(query, relevant_chunks, generative_model_name=GENERATIVE_MODEL):
    """Generates an answer using an LLM based on the query and relevant chunks."""
    if not relevant_chunks:
        return "I couldn't find any relevant information in the provided documents to answer your query."

    context = "\n\n---\n\n".join([chunk['text'] for chunk in relevant_chunks])
    sources = sorted(list(set(chunk['source'] for chunk in relevant_chunks))) # Get unique source filenames, sorted

    prompt = f"""You are a helpful AI assistant. Answer the following query based *only* on the provided context from PDF documents.
If the information is not in the context, say that you cannot answer based on the provided documents.
Be concise and directly answer the question. List the source PDF filenames if you use information from them.

Query: {query}

Context from documents:
---
{context}
---

Answer (mention source PDF filenames, e.g., [document1.pdf, document2.pdf]):
"""

    print(f"\n💬 Sending prompt to {generative_model_name}...")
    try:
        model = genai.GenerativeModel(generative_model_name)
        response = model.generate_content(prompt)
        # Add source attribution to the response
        answer_text = response.text
        if sources:
            answer_text += f"\n\nSources: [{', '.join(sources)}]"
        return answer_text
    except Exception as e:
        print(f"🔴 Error generating answer with LLM: {e}")
        if "400" in str(e) and "API key not valid" in str(e): # More specific error check
             print("🔴 Please ensure your GOOGLE_API_KEY is correct, valid, and has permissions for the selected model.")
        elif "Model gemini-2.5-pro-preview-05-06 not found" in str(e): # Example of specific model error
            print(f"🔴 The model {generative_model_name} was not found. It might be unavailable in your region or require specific access.")
        return "Sorry, I encountered an error while trying to generate an answer."

# --- Main Execution ---
if __name__ == "__main__":
    print("🚀 Starting PDF RAG Pipeline...")

    # Ensure PDF directory exists
    if not os.path.exists(PDF_DIRECTORY):
        print(f"⚠️ PDF directory '{PDF_DIRECTORY}' not found. Please create it and add your PDFs.")
        # Optionally, create the directory if it doesn't exist
        # os.makedirs(PDF_DIRECTORY, exist_ok=True)
        # print(f"Created directory '{PDF_DIRECTORY}'. Please add your PDFs there.")
        exit()

    # Step 1: Build or load the vector store (PDF parsing, chunking, embedding)
    print("\n--- Step 1: Building/Loading Vector Store ---")
    vector_store = build_or_load_vector_store(PDF_DIRECTORY, EMBEDDING_MODEL)

    if not vector_store:
        print("🔴 No embeddings were generated or loaded. Exiting.")
        exit()
    print(f"✅ Vector store ready with {len(vector_store)} chunks.")

    # Step 2: User Query and Retrieval
    print("\n--- Step 2: Querying ---")
    
    while True: # Loop to allow multiple queries
            user_input = input("❓ Enter your query or the name of a PDF file to use as the prompt (or type 'exit' to quit): ")
            if not user_input:
                print("🟡 No input entered. Please try again or type 'exit'.")
                continue
            if user_input.lower() == 'exit':
                break

            prompt_text = ""
            if user_input.lower().endswith(".pdf") and os.path.exists(user_input):
                print(f"📖 Using content of '{user_input}' as the prompt.")
                prompt_text = extract_text_from_pdf(user_input)
                if not prompt_text:
                    print("⚠️ Could not extract text from the specified PDF. Please enter a query manually.")
                    continue
            else:
                prompt_text = user_input
                print(f"\nProcessing query: '{prompt_text}'")

            if prompt_text:
                relevant_chunks = retrieve_relevant_chunks(prompt_text, vector_store, EMBEDDING_MODEL, top_k=TOP_K_RESULTS)

                if not relevant_chunks:
                    print("😔 No relevant information found for your query in the documents.")
                else:
                    # Step 3: Generate Answer
                    print("\n--- Step 3: Generating Answer ---")
                    answer = generate_answer_with_llm(prompt_text, relevant_chunks, GENERATIVE_MODEL)
                    print("\n💡 LLM Answer:")
                    print("--------------------------------------------------")
                    print(answer)
                    print("--------------------------------------------------")
            print("-" * 50) # Separator for next query

    print("\n✅ RAG Pipeline finished.")