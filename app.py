# import streamlit as st
# from langchain_openai import OpenAIEmbeddings  # Changed to OpenAI embeddings
# from langchain_chroma import Chroma 
# from langchain_groq import ChatGroq
# import os
# from dotenv import load_dotenv

# # Page config
# st.set_page_config(
#     page_title="UK Construction Regulations Assistant",
#     page_icon="🏗️",
#     layout="wide"
# )

# # Load environment variables
# load_dotenv()

# # Initialize RAG components
# @st.cache_resource
# def init_rag():
#     """Initialize RAG components with caching"""
#     try:
#         # Check if main_chroma_data exists
#         if not os.path.exists("./main_chroma_data"):
#             st.error("Error: main_chroma_data directory not found. Please check the directory path.")
#             return None, None

#         # Initialize embeddings
#         try:
#             embeddings = OpenAIEmbeddings(
#                 api_key=os.getenv("OPENAI_API_KEY")
#             )
#         except Exception as e:
#             st.error(f"Error initializing embeddings: {str(e)}")
#             return None, None

#         # Initialize vector store
#         try:
#             vectorstore = Chroma(
#                 collection_name="main_construction_rag",
#                 embedding_function=embeddings,
#                 persist_directory="./main_chroma_data"
#             )
#         except Exception as e:
#             st.error(f"Error initializing vector store: {str(e)}")
#             return None, None

#         # Check if GROQ API key is set
#         groq_api_key = os.getenv("GROQ_API_KEY")
#         if not groq_api_key:
#             st.error("Error: GROQ_API_KEY not found in environment variables")
#             return None, None

#         # Initialize LLM
#         try:
#             llm = ChatGroq(
#                 api_key=groq_api_key,
#                 model_name="llama-3.3-70b-versatile",
#                 temperature=0
#             )
#         except Exception as e:
#             st.error(f"Error initializing LLM: {str(e)}")
#             return None, None

#         return vectorstore, llm
#     except Exception as e:
#         st.error(f"Error initializing RAG system: {str(e)}")
#         return None, None

# # Initialize
# vectorstore, llm = init_rag()

# # Sidebar for feedback
# with st.sidebar:
#     st.title("📝 Feedback")
#     feedback = st.text_area("Share your feedback on the answers:", height=100)
#     if st.button("Submit Feedback"):
#         st.success("Thank you for your feedback!")

# # Main interface
# st.title("🏗️ UK Construction Regulations Assistant")
# st.markdown("""
# This AI assistant helps answer questions about UK construction regulations using:
# - Official Building Regulations documents
# - Expert YouTube content from LABC, RICS, and other authorities
# - Technical documentation and guidance
# """)

# # User input
# question = st.text_input("Enter your question about UK construction regulations:")

# if st.button("Get Answer"):
#     if not question:
#         st.warning("Please enter a question.")
#     elif vectorstore is None or llm is None:
#         st.error("RAG system not properly initialized. Please check the errors above.")
#     else:
#         with st.spinner("Searching regulations and generating answer..."):
#             try:
#                 # Get relevant documents
#                 docs = vectorstore.similarity_search(question, k=4)
#                 contexts = [doc.page_content for doc in docs]
                
#                 # Generate answer
#                 context_text = "\n\n".join(contexts)
#                 prompt = f"""Based on the following context from UK Building Regulations, provide a clear and detailed answer to the question.
#                 Include specific references to regulations where available.
                
#                 Question: {question}
                
#                 Context: {context_text}
                
#                 Answer:"""
                
#                 response = llm.invoke(prompt)
                
#                 # Display answer
#                 st.markdown("### Answer")
#                 st.write(response.content)
                
#                 # Display sources
#                 with st.expander("View Source Documents"):
#                     for i, context in enumerate(contexts, 1):
#                         st.markdown(f"**Source {i}:**")
#                         st.markdown(context)
#                         st.divider()
                        
#                 # Add thumbs up/down for answer quality
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     if st.button("👍 Helpful"):
#                         st.success("Thank you for your feedback!")
#                 with col2:
#                     if st.button("👎 Not Helpful"):
#                         st.info("Thank you for your feedback. Please let us know how we can improve in the sidebar.")
                        
#             except Exception as e:
#                 st.error(f"Error generating answer: {str(e)}")

# # Footer
# st.markdown("---")
# st.markdown("*This is a research project. Always verify information with official sources.*")

# Fix SQLite version issue on Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # Updated from langchain_community.vectorstores
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from cloud_storage import download_vectorstore

# Page config
st.set_page_config(
    page_title="UK Building Regulations Assistant",
    page_icon="🏗️",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Initialize RAG components
@st.cache_resource
def init_rag():
    """Initialize RAG components with caching"""
    try:
        # Check if main_chroma_data exists
        if not os.path.exists("./main_chroma_data"):
            #download_vectorstore()
            st.error("Error: main_chroma_data directory not found. Please check the directory path.")
            return None, None

        # Initialize embeddings
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                encode_kwargs={'normalize_embeddings': True}  # Added for stability
            )
        except Exception as e:
            st.error(f"Error initializing embeddings: {str(e)}")
            return None, None

        # Initialize vector store
        try:
            vectorstore = Chroma(
                collection_name="main_construction_rag",
                embedding_function=embeddings,
                persist_directory="./main_chroma_data"
            )
        except Exception as e:
            st.error(f"Error initializing vector store: {str(e)}")
            return None, None

        # Check if GROQ API key is set
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("Error: GROQ_API_KEY not found in environment variables")
            return None, None

        # Initialize LLM
        try:
            llm = ChatGroq(
                api_key=groq_api_key,
                model_name="llama-3.3-70b-versatile",
                temperature=0
            )
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            return None, None

        return vectorstore, llm
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None, None



# Initialize
vectorstore, llm = init_rag()

# Sidebar for feedback
with st.sidebar:
    st.title("📝 Feedback")
    feedback = st.text_area("Share your feedback on the answers:", height=100)
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")

# Main interface
st.title("🏗️ UK Building Regulations Assistant")
st.markdown("""
This AI assistant helps answer questions about UK construction regulations using:
- Official Building Regulations documents
- Expert YouTube content from LABC, RICS, and other authorities
- Technical documentation and guidance
""")

# User input
question = st.text_input("Enter your question about UK construction regulations:")

if st.button("Get Answer"):
    if not question:
        st.warning("Please enter a question.")
    elif vectorstore is None or llm is None:
        st.error("RAG system not properly initialized. Please check the errors above.")
    else:
        with st.spinner("Searching regulations and generating answer..."):
            try:
                # Get relevant documents
                docs = vectorstore.similarity_search(question, k=4)
                contexts = [doc.page_content for doc in docs]
                
                # Generate answer
                context_text = "\n\n".join(contexts)
                prompt = f"""Based on the following context from UK Building Regulations, provide a clear and detailed answer to the question.
                Include specific references to regulations where available.
                
                Question: {question}
                
                Context: {context_text}
                
                Answer:"""
                
                response = llm.invoke(prompt)
                
                # Display answer
                st.markdown("### Answer")
                st.write(response.content)
                
                # Display sources
                with st.expander("View Source Documents"):
                    for i, context in enumerate(contexts, 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(context)
                        st.divider()
                        
                # Add thumbs up/down for answer quality
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Helpful"):
                        st.success("Thank you for your feedback!")
                with col2:
                    if st.button("👎 Not Helpful"):
                        st.info("Thank you for your feedback. Please let us know how we can improve in the sidebar.")
                        
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*This is a research project. Always verify information with official sources.*")