import streamlit as st
import ollama

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

st.set_page_config(page_title="Cat Fact AI Explorer", page_icon="🐱")
st.title("🐱 Cat Fact AI Explorer")
st.caption("A simple RAG application powered by Ollama and Streamlit")

# ==========================================
# 2. HELPER FUNCTIONS & CACHING
# ==========================================
def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

# We use st.cache_resource so this database is only built ONCE when the app starts
@st.cache_resource
def initialize_vector_db():
    vector_db = []
    try:
        with open('cat-facts.txt', 'r', encoding='utf-8', errors='replace') as file:
            dataset = file.readlines()
        
        # Display a temporary loading spinner in the UI
        with st.spinner(f"Loading and embedding {len(dataset)} cat facts..."):
            for chunk in dataset:
                chunk = chunk.strip()
                if chunk:  # skip empty lines
                    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
                    vector_db.append((chunk, embedding))
        return vector_db
    except FileNotFoundError:
        st.error("Could not find 'cat-facts.txt'. Please make sure it's in the same folder.")
        return []

# Initialize the database
VECTOR_DB = initialize_vector_db()

def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# ==========================================
# 3. CHAT INTERFACE (STREAMLIT)
# ==========================================

# Initialize chat history in session state so it persists across refreshes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_query := st.chat_input("Ask me anything about cats!"):
    
    # 1. Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Run the RAG Retrieval
    retrieved_knowledge = retrieve(user_query)

    # 3. Build the LLM prompt with context
    instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
'''

    # 4. Display assistant response in chat message container with streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Call Ollama stream
        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': user_query},
            ],
            stream=True,
        )
        
        # Render chunks onto the UI in real-time
        for chunk in stream:
            full_response += chunk['message']['content']
            response_placeholder.markdown(full_response + "▌")
        
        # Remove the cursor block at the end
        response_placeholder.markdown(full_response)
        
        # Sidebar feature: show what knowledge was retrieved (Optional, but cool!)
        with st.sidebar:
            st.write("### 🧠 Retrieved Context for Last Query:")
            for chunk, sim in retrieved_knowledge:
                st.write(f"**Score: {sim:.2f}**\n{chunk}")
                st.divider()

    # Save assistant history
    st.session_state.messages.append({"role": "assistant", "content": full_response})