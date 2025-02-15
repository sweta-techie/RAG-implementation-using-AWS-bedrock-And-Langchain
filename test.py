# rag_generate_ques_semantic_chunking_corrected.py

import os
import re
import requests
from typing import List
from tempfile import NamedTemporaryFile
import time
import random
import uuid

import streamlit as st
import fitz  # PyMuPDF for PDF processing

import openai
from openai import RateLimitError, OpenAIError

from dotenv import load_dotenv
from security import safe_requests

load_dotenv()
import sys
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# -----------------------------
# Streamlit Configuration
# -----------------------------
st.set_page_config(page_title="RAG PDF Q&A with GPT-4", layout="wide")
st.title("Retrieval-Augmented Generation (RAG) PDF Q&A Application with GPT-4")
# st.write(f"Python version: {sys.version}")
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
        response = safe_requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        temp_pdf = NamedTemporaryFile(delete=False, suffix=".pdf")
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_pdf.write(chunk)
        temp_pdf.flush()
        return temp_pdf
    except requests.exceptions.RequestException as e:
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
        return cleaned_text
    except Exception as e:
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
        prompt = (
            "You are a helpful assistant that summarizes academic documents.\n\n"
            f"Summarize the following text in a concise manner:\n\n{text}"
        )
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes academic documents."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,  # Adjust as needed
            temperature=0.5,
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except OpenAIError as e:
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
        prompt = (
            "You are an expert in extracting key topics from academic texts.\n\n"
            f"Extract the top 5 key topics from the following text:\n\n{text}"
        )
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in extracting key topics from academic texts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.5,
        )
        topics_text = response.choices[0].message.content.strip()
        # Split topics assuming they're separated by commas or listed
        topics = re.split(r',|\n|\d+\.', topics_text)
        topics = [topic.strip().strip('-').strip() for topic in topics if topic.strip()]
        return topics[:5]  # Limit to top 5 topics
    except OpenAIError as e:
        st.error(f"Error during topic extraction: {e}")
        return []

def generate_questions_batch(text: str, topics: List[str], num_questions: int) -> List[str]:
    """
    Generates multiple questions in a single API call based on provided topics.

    Args:
        text (str): The text to base questions on.
        topics (List[str]): List of topics to generate questions about.
        num_questions (int): Total number of questions to generate.

    Returns:
        List[str]: List of generated questions.
    """
    try:
        prompt = (
            f"You are an expert educator tasked with creating {num_questions} insightful and thought-provoking questions "
            f"to test comprehension of the following academic material. Each question should relate to one of the provided topics "
            f"and require an in-depth understanding of the material to answer.\n\n"
            f"Text:\n\"\"\"\n{text}\n\"\"\"\n\n"
            f"Topics:\n" + "\n".join([f"- {topic}" for topic in topics]) + "\n\n"
            f"Questions:\n1."
        )

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates academic questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,  # Adjust based on expected response length
            temperature=0.7,
        )

        questions_text = response.choices[0].message.content.strip()
        # Split the questions by numbering
        questions = re.split(r'\n\d+\.', questions_text)
        # Clean and format questions
        questions = [q.strip().lstrip('. ').rstrip('?') + '?' for q in questions if q.strip()]
        return questions[:num_questions]
    except OpenAIError as e:
        st.error(f"Error during question generation: {e}")
        return []

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
            max_tokens=300,  # Adjust as needed
            temperature=0.5,
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except OpenAIError as e:
        st.error(f"Error during answer fetching: {e}")
        return "An error occurred while generating the answer."

def fetch_answer_with_backoff(question: str, text: str, max_retries: int = 3) -> str:
    """
    Fetches an answer with exponential backoff in case of rate limits.

    Args:
        question (str): The question to answer.
        text (str): The context text to base the answer on.
        max_retries (int): Maximum number of retries.

    Returns:
        str: The generated answer or error message.
    """
    for attempt in range(max_retries):
        try:
            return fetch_answer(question, text)
        except RateLimitError:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            st.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
        except OpenAIError as e:
            st.error(f"Error during answer fetching: {e}")
            break
    st.error("Max retries exceeded. Unable to fetch answer at this time.")
    return "An error occurred while generating the answer."

def split_text_into_chunks(text: str, max_tokens: int) -> List[str]:
    """
    Splits the text into chunks that do not exceed the max_tokens limit.

    Args:
        text (str): The text to split.
        max_tokens (int): Maximum number of tokens per chunk.

    Returns:
        List[str]: List of text chunks.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if current_tokens + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
                current_tokens = 0
        current_chunk += " " + sentence
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def estimate_tokens(text: str) -> int:
    """
    Estimates the number of tokens in the given text.
    This is a rough estimation using the ratio of characters to tokens.

    Args:
        text (str): The text to estimate.

    Returns:
        int: Estimated number of tokens.
    """
    # Average English word is ~4 characters, 1 token ~= 4 characters
    return len(text) // 4

# -----------------------------
# Session State Initialization
# -----------------------------

def initialize_session_state():
    """
    Initializes necessary session state variables.
    """
    if 'uploaded_pdfs' not in st.session_state:
        st.session_state.uploaded_pdfs = []  # List of dicts: {'key', 'name', 'text'}
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'answers' not in st.session_state:
        st.session_state.answers = {}
    if 'pdf_summary' not in st.session_state:
        st.session_state.pdf_summary = ""
    if 'topics' not in st.session_state:
        st.session_state.topics = []
    if 'previous_num_questions' not in st.session_state:
        st.session_state.previous_num_questions = 5  # Default slider value
    if 'custom_question_input' not in st.session_state:
        st.session_state.custom_question_input = ""

initialize_session_state()

# -----------------------------
# Sidebar for Inputs
# -----------------------------

st.sidebar.header("Input Options")

# Optional: Show Debug Info
# show_debug = st.sidebar.checkbox("Show Debug Info", value=False)

# Function to generate unique key for each file
def generate_file_key(file) -> str:
    return f"{file.name}_{file.size}"

# Option to upload PDF files
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
if uploaded_files is not None:
    # Generate keys for currently uploaded files
    current_file_keys = set(generate_file_key(file) for file in uploaded_files)
    # Generate keys for previously uploaded files
    previous_file_keys = set(pdf['key'] for pdf in st.session_state.uploaded_pdfs)

    # Identify removed files
    removed_keys = previous_file_keys - current_file_keys
    if removed_keys:
        st.session_state.uploaded_pdfs = [pdf for pdf in st.session_state.uploaded_pdfs if pdf['key'] not in removed_keys]
        st.success(f"Removed {len(removed_keys)} PDF(s).")
        # Clear previous questions and answers since PDFs have changed
        st.session_state.questions = []
        st.session_state.answers = {}
        st.session_state.pdf_summary = ""
        st.session_state.topics = []

    # Identify new uploads
    new_keys = current_file_keys - previous_file_keys
    if new_keys:
        for file in uploaded_files:
            key = generate_file_key(file)
            if key in new_keys:
                with st.spinner(f"Processing {file.name}..."):
                    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(file.read())
                        temp_file.flush()
                        text = extract_text_from_pdf(temp_file.name)
                os.remove(temp_file.name)
                # Add the new PDF to session_state
                st.session_state.uploaded_pdfs.append({'key': key, 'name': file.name, 'text': text})
        st.success(f"Processed {len(new_keys)} new PDF(s).")
        # Clear previous questions and answers since PDFs have changed
        st.session_state.questions = []
        st.session_state.answers = {}
        st.session_state.pdf_summary = ""
        st.session_state.topics = []
else:
    # Handle the case when all PDFs are removed
    if st.session_state.uploaded_pdfs:
        st.session_state.uploaded_pdfs = []
        st.session_state.questions = []
        st.session_state.answers = {}
        st.session_state.pdf_summary = ""
        st.session_state.topics = []
        st.info("All PDFs have been removed.")

# Option to input a PDF URL
pdf_url = st.sidebar.text_input("Or enter a PDF URL:", placeholder="https://example.com/paper.pdf")
if pdf_url:
    with st.spinner("Downloading and processing PDF from URL..."):
        temp_pdf = download_pdf_from_url(pdf_url)
        if temp_pdf:
            file_name = os.path.basename(temp_pdf.name)
            file_size = os.path.getsize(temp_pdf.name)
            key = f"{file_name}_{file_size}"
            text = extract_text_from_pdf(temp_pdf.name)
            # Add the new PDF to session_state
            st.session_state.uploaded_pdfs.append({'key': key, 'name': file_name, 'text': text})
            os.remove(temp_pdf.name)
            st.success("PDF downloaded and processed successfully!")
            # Clear previous questions and answers since PDFs have changed
            st.session_state.questions = []
            st.session_state.answers = {}
            st.session_state.pdf_summary = ""
            st.session_state.topics = []
    st.write("---")

# Slider for number of questions to generate
num_questions = st.sidebar.slider(
    "Number of questions to generate:",
    min_value=1,
    max_value=15,
    value=5,
    step=1,
    help="Select the number of questions you want to generate based on the document."
)

# Checkbox for summarization (unchecked by default)
summarize = st.sidebar.checkbox("Summarize Document", value=False)
st.session_state.summarize = summarize

# Detect changes in slider value to regenerate questions
if num_questions != st.session_state.previous_num_questions:
    st.session_state.questions = []
    st.session_state.answers = {}
    st.session_state.previous_num_questions = num_questions
    st.session_state.pdf_summary = ""  # Optionally reset summary
    st.session_state.topics = []

# -----------------------------
# Combine All PDF Texts
# -----------------------------

combined_pdf_text = "\n\n".join([pdf['text'] for pdf in st.session_state.uploaded_pdfs])

# Optional Debug Info: Display number of PDFs and a snippet of combined text
# if show_debug:
#     st.sidebar.write(f"Number of uploaded PDFs: {len(st.session_state.uploaded_pdfs)}")
#     st.sidebar.write(f"Combined PDF text length: {len(combined_pdf_text)} characters")
#     st.sidebar.write(f"Combined PDF text snippet:\n{combined_pdf_text[:500]}...")

# -----------------------------
# Main Content
# -----------------------------

if combined_pdf_text:
    # Define maximum tokens for input
    MAX_TOKENS = 7000  # Reserve some tokens for the completion

    text_token_count = estimate_tokens(combined_pdf_text)
    if text_token_count > MAX_TOKENS:
        if not st.session_state.summarize:
            st.warning(
                "The combined text from the uploaded PDFs is too long to process without summarization. "
                "Please enable the 'Summarize Document' checkbox to reduce the text size."
            )
        else:
            # Split the text into manageable chunks
            chunks = split_text_into_chunks(combined_pdf_text, max_tokens=7000)
            all_questions = []
            for chunk in chunks:
                with st.spinner("Generating questions for a text chunk..."):
                    topics = extract_topics(chunk)
                    questions = generate_questions_batch(chunk, topics, num_questions)
                    all_questions.extend(questions)
            st.session_state.questions = all_questions
            st.success("Questions generated successfully from all chunks!")
    else:
        # Summarize if checkbox is selected and summary not already generated
        if st.session_state.summarize and not st.session_state.pdf_summary:
            with st.spinner("Summarizing the document..."):
                summary = summarize_text(combined_pdf_text)
                st.session_state.pdf_summary = summary
            st.write("**Summary of the Document:**")
            st.write(st.session_state.pdf_summary)
            text_for_topics = st.session_state.pdf_summary
        else:
            text_for_topics = combined_pdf_text

        # Extract Topics
        if not st.session_state.topics:
            with st.spinner("Extracting key topics from the document..."):
                topics = extract_topics(text_for_topics)
                st.session_state.topics = topics
            # Optionally display topics for debugging
            # if show_debug:
            #     st.write("**Extracted Topics:**")
            #     for idx, topic in enumerate(st.session_state.topics, 1):
            #         st.write(f"{idx}. {topic}")
            # Removed the display of topics on UI as per the request

        # Generate Questions
        if not st.session_state.questions:
            with st.spinner("Generating questions based on extracted topics..."):
                questions = generate_questions_batch(text_for_topics, st.session_state.topics, num_questions)
                st.session_state.questions = questions
            if st.session_state.questions:
                st.success("Questions generated successfully!")
            else:
                st.warning("No questions were generated. Please try with a different PDF or adjust the settings.")

    # Display Generated Questions as Clickable Buttons
    if st.session_state.questions:
        st.subheader("Generated Questions:")
        for idx, question in enumerate(st.session_state.questions, 1):
            # Each button has a unique key
            if st.button(f"Q{idx}: {question}", key=f"question_{idx}"):
                # Fetch the answer and store it in session state
                if question not in st.session_state.answers:
                    with st.spinner("Fetching answer..."):
                        answer = fetch_answer_with_backoff(question, combined_pdf_text)
                        st.session_state.answers[question] = answer
                # Display the answer below the button
                st.write(f"**Answer:** {st.session_state.answers[question]}")
        st.write("---")

    # Custom Query Input Field using a Form
    st.subheader("Ask a Custom Question")

    with st.form(key='custom_query_form', clear_on_submit=True):
        user_question = st.text_input(
            "Enter your custom question here:",
            placeholder="Type your question and press Enter",
            key='custom_question_input'
        )
        submit_button = st.form_submit_button(label='Submit Custom Query')

    if submit_button:
        if not user_question.strip():
            st.error("Please enter a valid question.")
        else:
            with st.spinner("Fetching answer..."):
                answer = fetch_answer_with_backoff(user_question, combined_pdf_text)
                st.session_state.answers[user_question] = answer
        st.write(f"**Your Question:** {user_question}")
        st.write(f"**Answer:** {st.session_state.answers[user_question]}")
        st.write("---")

else:
    st.info("Please upload a PDF file or enter a PDF URL to generate questions.")
