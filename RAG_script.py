#Imports
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
#========================================================
#Retrieval component
#Retrieve most relevant document chunks for a query
def retrieve_relevant_chunks(query,vectorstore,k=3):
#Perform similarity search with scores

    print(f"\n[RETRIEVAL] searching for: '{query}'")
    results = vectorstore.similarity_search_with_score(query,k=k)
#format results
    
    seen = set()
    retrieved_chunks = []
    for doc, score in results:
        content = doc.page_content.strip()

        if content in seen:
            continue
        seen.add(content)


        retrieved_chunks.append({
            "content": content,
            "score": score,
            "metadata": doc.metadata

        })
    print(f" Retrieved {len(retrieved_chunks)} relevant chunks")
    return retrieved_chunks


#========================================================
#Prompt engineering
def build_rag_prompt(query,retrieved_chunks):
#combine all retrieved chunks into context
    context = "\n\n---\n\n".join([chunk['content'] for chunk in retrieved_chunks])

    prompt = f"""You are a helpful assistant that answers questions based on the provided context.
INSTRUCTIONS:
- Use ONLY the information from the context provided to answer the question
- If the answer is not in the context, say "I dont have enough information to answer that question."
- Be consise and accurate
- Cite specific parts of the context when relevant

CONTEXT: {context}

QUESTION {query}

ANSWER:
"""

    return prompt



#========================================================
#Generation component
def generate_response(prompt,retrieved_chunks ,model = "gpt-3.5-turbo",temperature = 0.3):

    print(f"\n[GENERATION] Generating response with {model}")

#Initialise LLM
    llm = ChatOpenAI(
        model = model,
        temperature = temperature,
        openai_api_key = os.getenv("OPENAI_API_KEY")
        )


#Generate response
    response = llm.invoke(prompt)
    if retrieved_chunks:
        confidence = sum(chunk["score"] for chunk in retrieved_chunks)/len(retrieved_chunks)
    else:
        confidence = 0.0
    print("Response generated")
    return {
        "answer":response.content,
        "confidence": round(confidence, 4)
    }

#========================================================
#RAG query function
def query_rag_system(question, vectorstore, k=3, verbose = True):
    print(f"\n" + "+"*70)
    print("RAG QUERY PIPELINE")
    print("="*70)
#Retrieve relevant chunks
    retrieved_chunks = retrieve_relevant_chunks(question, vectorstore,k=k)
    if verbose:
        print("\n[RETRIEVED CHUNKS]")
        for i, chunk in enumerate(retrieved_chunks,1):
            print(f"\nChunk {i} (Score: {chunk['score']:.4f}):")
            print(chunk['content'][:200]+"...")
#Build Prompt
    rag_prompt = build_rag_prompt(question, retrieved_chunks)
    if verbose:
        print(f"\n[PROMPT] Length: {len(rag_prompt)} characters")
#Generate answer

#Return results
    result = generate_response(rag_prompt, retrieved_chunks)
    return{
        "question" : question,
        "answer" : result["answer"],
        "confidence": result["confidence"],
        "retrieved_chunks": retrieved_chunks,
        "num_chunks_used": len(retrieved_chunks)
    }
#========================================================
#Usage
def load_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    return vectorstore