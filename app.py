import streamlit as st
from rag_system import get_answer  # Import the function from your RAG system

# Streamlit interface
def main():
    st.title("RAG System Interface")
    st.write("Ask a question and get an answer from the document database.")
    
    query = st.text_input("Enter your query here:")
    
    if st.button("Get Answer"):
        if query:
            with st.spinner("Fetching answer..."):
                answer = get_answer(query)
                st.write("### Answer:")
                st.write(answer)
        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    main()
