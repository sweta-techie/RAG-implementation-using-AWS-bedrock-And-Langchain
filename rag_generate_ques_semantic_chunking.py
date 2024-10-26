# rag_generate_ques_gpt4.py

import os
import logging
import re
import requests
from typing import List
from tempfile import NamedTemporaryFile

import streamlit as st
import fitz  # PyMuPDF for PDF processing

import openai
from openai import AuthenticationError, RateLimitError, OpenAIError

# -----------------------------
# Streamlit Configuration
# -----------------------------
st.set_page_config(page_title="Enhanced RAG PDF Q&A with GPT-4", layout="wide")
st.title("Enhanced Retrieval-Augmented Generation (RAG) PDF Q&A Application with GPT-4")

# -----------------------------
# Configure Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# OpenAI API Configuration
# -----------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# -----------------------------
# Utility Functions
# -----------------------------

def clean_text(text: str) -> str:
    """
    Cleans the extracted text by removing unwanted artifacts.

    Args:
        text (str): The text to clean.

    Returns:
        str: Cleaned text.
    """
    # Remove extra whitespace and line breaks
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def download_pdf_from_url(url: str) -> NamedTemporaryFile:
    """
    Downloads a PDF from the given URL and saves it to a temporary file.

    Args:
        url (str): The URL pointing to the PDF.

    Returns:
        NamedTemporaryFile: The temporary file containing the downloaded PDF.
    """
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        temp_pdf = NamedTemporaryFile(delete=False, suffix=".pdf")
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_pdf.write(chunk)
        temp_pdf.flush()
        logger.info(f"Downloaded PDF from URL: {url}")
        return temp_pdf
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download PDF from URL '{url}': {e}")
        st.error(f"Failed to download PDF from URL: {e}")
        return None

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        cleaned_text = clean_text(text)
        logger.info(f"Extracted text from PDF: {pdf_path}")
        return cleaned_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF '{pdf_path}': {e}")
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def summarize_text(text: str) -> str:
    """
    Summarizes the given text using GPT-4.

    Args:
        text (str): The text to summarize.

    Returns:
        str: Summarized text.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes academic documents."},
                {"role": "user", "content": f"Summarize the following text in a concise manner:\n\n{text}"}
            ],
            max_tokens=300,
            temperature=0.5,
        )
        # Corrected line: Use attribute access
        summary = response.choices[0].message.content.strip()
        logger.info("Generated summary using GPT-4.")
        return summary
    except OpenAIError as e:
        logger.error(f"OpenAI API error during summarization: {e}")
        st.error(f"Error during summarization: {e}")
        return text  # Return original text if summarization fails

def extract_topics(text: str) -> List[str]:
    """
    Extracts key topics from the text using GPT-4.

    Args:
        text (str): The text from which to extract topics.

    Returns:
        List[str]: List of extracted topics.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in extracting key topics from academic texts."},
                {"role": "user", "content": f"Extract the top 5 key topics from the following text:\n\n{text}"}
            ],
            max_tokens=150,
            temperature=0.5,
        )
        # Corrected line: Use attribute access
        topics_text = response.choices[0].message.content.strip()
        # Assume GPT-4 returns topics separated by commas or a numbered list
        topics = re.split(r'\n|\r|\d+\.', topics_text)
        topics = [topic.strip().strip('-').strip() for topic in topics if topic.strip()]
        logger.info("Extracted topics using GPT-4.")
        return topics[:5]  # Limit to top 5 topics
    except OpenAIError as e:
        logger.error(f"OpenAI API error during topic extraction: {e}")
        st.error(f"Error during topic extraction: {e}")
        return []

def generate_questions(text: str, topics: List[str]) -> List[str]:
    """
    Generates questions based on the provided text and topics using GPT-4.

    Args:
        text (str): The text to base questions on.
        topics (List[str]): List of topics to generate questions about.

    Returns:
        List[str]: List of generated questions.
    """
    questions = []
    for topic in topics:
        try:
            prompt = (
                f"You are an expert educator tasked with creating insightful and thought-provoking questions to test comprehension of complex academic materials.\n\n"
                f"Given the following text and a highlighted topic, generate a clear and relevant question that requires an in-depth understanding of the material to answer.\n\n"
                f"Text:\n\"\"\"\n{text}\n\"\"\"\n\n"
                f"Highlighted Topic: {topic}\n\n"
                f"Question:"
            )
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates academic questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7,
            )
            # Corrected line: Use attribute access
            question = response.choices[0].message.content.strip()
            # Ensure the question ends with a question mark
            if not question.endswith('?'):
                question += '?'
            questions.append(question)
            logger.info(f"Generated question for topic '{topic}': {question}")
        except OpenAIError as e:
            logger.error(f"OpenAI API error during question generation for topic '{topic}': {e}")
            st.error(f"Error during question generation: {e}")
    return questions

def fetch_answer(question: str, text: str) -> str:
    """
    Fetches an answer to the question based on the provided text using GPT-4.

    Args:
        question (str): The question to answer.
        text (str): The context text to base the answer on.

    Returns:
        str: The generated answer.
    """
    try:
        prompt = (
            f"You are an expert in the subject matter of the following text.\n\n"
            f"Text:\n\"\"\"\n{text}\n\"\"\"\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a knowledgeable and helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.5,
        )
        # Corrected line: Use attribute access
        answer = response.choices[0].message.content.strip()
        logger.info(f"Fetched answer for question '{question}'.")
        return answer
    except OpenAIError as e:
        logger.error(f"OpenAI API error during answer fetching for question '{question}': {e}")
        st.error(f"Error during answer fetching: {e}")
        return "An error occurred while generating the answer."

# -----------------------------
# Streamlit Application Logic
# -----------------------------

def initialize_session_state():
    """
    Initializes necessary session state variables.
    """
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'answers' not in st.session_state:
        st.session_state.answers = {}
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""
    if 'pdf_summary' not in st.session_state:
        st.session_state.pdf_summary = ""
    if 'topics' not in st.session_state:
        st.session_state.topics = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

def reset_application():
    """
    Resets the application by clearing session state.
    """
    if st.button("Reset Application", key="reset_application_button"):
        logger.info("Manual reset triggered by user.")
        st.session_state.questions = []
        st.session_state.answers = {}
        st.session_state.pdf_text = ""
        st.session_state.pdf_summary = ""
        st.session_state.topics = []
        st.session_state.uploaded_files = []
        st.success("Application has been reset. Please upload a new PDF to generate questions.")
        st.experimental_rerun()

def main():
    initialize_session_state()

    # Sidebar for Inputs
    st.sidebar.header("Input Options")

    # Option to upload PDF files
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(uploaded_file)
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_file.flush()
                        text = extract_text_from_pdf(temp_file.name)
                        st.session_state.pdf_text += f"\n\n{text}"
                    os.remove(temp_file.name)
        st.success("Uploaded PDFs processed successfully!")

    # Option to input a PDF URL
    pdf_url = st.sidebar.text_input("Or enter a PDF URL:", placeholder="https://example.com/paper.pdf")
    if pdf_url:
        with st.spinner("Downloading and processing PDF from URL..."):
            temp_pdf = download_pdf_from_url(pdf_url)
            if temp_pdf:
                text = extract_text_from_pdf(temp_pdf.name)
                st.session_state.pdf_text += f"\n\n{text}"
                os.remove(temp_pdf.name)
                st.success("PDF downloaded and processed successfully!")

    # Checkbox for summarization
    summarize = st.sidebar.checkbox("Summarize Document", value=True)

    # Proceed only if PDF text is available
    if st.session_state.pdf_text:
        if summarize and not st.session_state.pdf_summary:
            with st.spinner("Summarizing the document..."):
                summary = summarize_text(st.session_state.pdf_text)
                st.session_state.pdf_summary = summary
            st.write("**Summary of the Document:**")
            st.write(st.session_state.pdf_summary)
            text_for_topics = st.session_state.pdf_summary
        else:
            text_for_topics = st.session_state.pdf_text

        # Extract Topics
        if not st.session_state.topics:
            with st.spinner("Extracting key topics from the document..."):
                topics = extract_topics(text_for_topics)
                st.session_state.topics = topics
            st.write("**Extracted Topics:**")
            for idx, topic in enumerate(st.session_state.topics, 1):
                st.write(f"{idx}. {topic}")

        # Generate Questions
        if not st.session_state.questions:
            with st.spinner("Generating questions based on extracted topics..."):
                questions = generate_questions(text_for_topics, st.session_state.topics)
                st.session_state.questions = questions
            if st.session_state.questions:
                st.success("Questions generated successfully!")
            else:
                st.warning("No questions were generated. Please try with a different PDF or adjust the settings.")

        # Display Generated Questions
        if st.session_state.questions:
            st.subheader("Generated Questions:")
            for idx, question in enumerate(st.session_state.questions, 1):
                if st.button(f"Q{idx}: {question}", key=f"question_{idx}"):
                    # Fetch or display the answer
                    if question not in st.session_state.answers:
                        with st.spinner("Fetching answer..."):
                            answer = fetch_answer(question, st.session_state.pdf_text)
                            st.session_state.answers[question] = answer
                    st.write(f"**Q{idx}:** {question}")
                    st.write(f"**Answer:** {st.session_state.answers[question]}")
            st.write("---")

    else:
        st.info("Please upload a PDF file or enter a PDF URL to generate questions.")

    # Reset Button
    reset_application()

if __name__ == "__main__":
    main()
