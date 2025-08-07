from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
you are an expert in answering questions about a pizza restaurant

here are some relavent reviews: {reviews}

here are some questions: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\nEnter question for the chatbot:(type q to exit)")
    question = input()
    if question == "q":
        break
    
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews,"question":question})
    print(result)
