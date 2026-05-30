# Step 1: - Simple Document
document = '''
# Introduction to Agentic AI

Agentic AI is a type of artificial intelligence that can make decisions and take actions to achieve specific goals. Unlike traditional AI systems that only respond to direct user inputs, Agentic AI can plan, reason, and act autonomously. It is designed to perform tasks with minimal human intervention. Agentic AI combines large language models, memory, reasoning, and tool usage to solve complex problems.

An Agentic AI system starts by understanding a user's objective. It then breaks the objective into smaller tasks. The system creates a plan to accomplish those tasks efficiently. It can choose the best sequence of actions based on the available information. Agentic AI continuously evaluates its progress toward the goal.

One of the key features of Agentic AI is its ability to use external tools. These tools may include web search engines, databases, calculators, APIs, and software applications. By using tools, the AI can gather real-time information and perform actions beyond simple text generation. This makes Agentic AI more powerful and practical in real-world situations.

Memory is another important component of Agentic AI. The system can store information from previous interactions and use it later when needed. This helps the AI maintain context and provide more personalized responses. Long-term memory allows the agent to improve its performance over time.

Agentic AI also relies on reasoning capabilities. It can analyze information, compare alternatives, and make informed decisions. The AI can identify obstacles and adjust its strategy when circumstances change. This adaptability enables it to handle dynamic and unpredictable environments.

In business applications, Agentic AI can automate workflows and improve productivity. It can schedule meetings, generate reports, analyze data, and manage customer interactions. Companies use Agentic AI to reduce repetitive work and increase efficiency. It helps employees focus on higher-value tasks.

In healthcare, Agentic AI can assist doctors by analyzing medical records and suggesting possible diagnoses. In education, it can act as a personalized tutor that adapts to each student's learning style. In finance, it can monitor market trends and support decision-making processes. These applications demonstrate the versatility of Agentic AI.

Despite its advantages, Agentic AI also presents challenges. Ensuring safety and reliability is essential. Developers must prevent the system from taking harmful or unintended actions. Transparency and accountability are important when AI systems make decisions that affect people.

The future of Agentic AI is promising. Researchers are working on improving reasoning, memory, and planning capabilities. As technology advances, Agentic AI systems will become more capable and autonomous. They are expected to play a significant role in various industries and everyday life. Agentic AI represents an important step toward intelligent systems that can understand goals, make decisions, and take meaningful actions to achieve desired outcomes.
'''

print(document)

# Step 2: Chunking
text = '''
# Introduction to Agentic AI

Agentic AI is a type of artificial intelligence that can make decisions and take actions to achieve specific goals. Unlike traditional AI systems that only respond to direct user inputs, Agentic AI can plan, reason, and act autonomously. It is designed to perform tasks with minimal human intervention. Agentic AI combines large language models, memory, reasoning, and tool usage to solve complex problems.

An Agentic AI system starts by understanding a user's objective. It then breaks the objective into smaller tasks. The system creates a plan to accomplish those tasks efficiently. It can choose the best sequence of actions based on the available information. Agentic AI continuously evaluates its progress toward the goal.

One of the key features of Agentic AI is its ability to use external tools. These tools may include web search engines, databases, calculators, APIs, and software applications. By using tools, the AI can gather real-time information and perform actions beyond simple text generation. This makes Agentic AI more powerful and practical in real-world situations.

Memory is another important component of Agentic AI. The system can store information from previous interactions and use it later when needed. This helps the AI maintain context and provide more personalized responses. Long-term memory allows the agent to improve its performance over time.

Agentic AI also relies on reasoning capabilities. It can analyze information, compare alternatives, and make informed decisions. The AI can identify obstacles and adjust its strategy when circumstances change. This adaptability enables it to handle dynamic and unpredictable environments.

In business applications, Agentic AI can automate workflows and improve productivity. It can schedule meetings, generate reports, analyze data, and manage customer interactions. Companies use Agentic AI to reduce repetitive work and increase efficiency. It helps employees focus on higher-value tasks.

In healthcare, Agentic AI can assist doctors by analyzing medical records and suggesting possible diagnoses. In education, it can act as a personalized tutor that adapts to each student's learning style. In finance, it can monitor market trends and support decision-making processes. These applications demonstrate the versatility of Agentic AI.

Despite its advantages, Agentic AI also presents challenges. Ensuring safety and reliability is essential. Developers must prevent the system from taking harmful or unintended actions. Transparency and accountability are important when AI systems make decisions that affect people.

The future of Agentic AI is promising. Researchers are working on improving reasoning, memory, and planning capabilities. As technology advances, Agentic AI systems will become more capable and autonomous. They are expected to play a significant role in various industries and everyday life. Agentic AI represents an important step toward intelligent systems that can understand goals, make decisions, and take meaningful actions to achieve desired outcomes.
'''

chunks = text.split('.')
print(chunks)

# Step 3: Remove empty chunks
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
print(chunks)

# Simple retrieval (TF-IDF + cosine similarity)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()
chunk_matrix = vectorizer.fit_transform(chunks)

# User Query Loop
while True:

    query = input("\nEnter your question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Program Ended")
        break

    print(query)

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, chunk_matrix)
    print(similarities)

    best_match_index = similarities.argmax()
    retrieved_chunk = chunks[best_match_index]

    print(chunks[best_match_index])

    prompt = f'''\
Context:
{retrieved_chunk}

Question:
{query}
'''
    print(prompt)

    answer = f'''\
Based on the Context,
RAG helps reduce hallucination
(Example retrieved evidence: {retrieved_chunk[:160]}...)
'''
    print(answer)