import sys
import os
import logging
import hashlib
import re
from typing import List, Dict, Any, Tuple
from tempfile import NamedTemporaryFile
import io
import uuid

import streamlit as st
from PIL import Image
import fitz  # PyMuPDF

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Annoy

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import openai
from openai import AuthenticationError, RateLimitError, OpenAIError

from langchain.schema import Document
from keybert import KeyBERT
import nltk  # For sentence tokenization

import spacy  # For Named Entity Recognition
from difflib import SequenceMatcher  # For fuzzy matching
import secrets

# -----------------------------
# Streamlit Configuration
# -----------------------------
st.set_page_config(page_title="Enhanced RAG PDF Q&A", layout="wide")
st.header("Enhanced Retrieval-Augmented Generation (RAG) PDF Q&A Application")

# -----------------------------
# Configure Logging
# -----------------------------
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logs
logger = logging.getLogger(__name__)

# -----------------------------
# Download NLTK Data
# -----------------------------
nltk.download('punkt')  # Download the Punkt tokenizer for sentence tokenization

# -----------------------------
# Initialize Models
# -----------------------------
kw_model = KeyBERT()

# Load spaCy model for NER
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Download the model if not present
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

nlp = load_spacy_model()

# -----------------------------
# Helper Functions
# -----------------------------

def compute_file_hash(file) -> str:
    """
    Computes the SHA256 hash of the given file's content.

    Args:
        file (UploadedFile): The uploaded file.

    Returns:
        str: The hexadecimal SHA256 hash of the file content.
    """
    file_content = file.read()
    file_hash = hashlib.sha256(file_content).hexdigest()
    file.seek(0)  # Reset file pointer after reading
    return file_hash

@st.cache_resource
def load_retrieval_model():
    """
    Loads the embedding model for retrieval.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_model

@st.cache_resource
def load_bm25(texts: List[str]):
    """
    Initializes the BM25 model with the provided texts.

    Args:
        texts (List[str]): List of document texts.

    Returns:
        BM25Okapi: Initialized BM25 model.
    """
    tokenized_corpus = [doc.lower().split(" ") for doc in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

@st.cache_resource
def load_qg_model():
    """
    Loads the question generation model.
    """
    model_name = 'valhalla/t5-base-qg-hl'
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def load_vlm_model():
    """
    Configures OpenAI's GPT-4 model.

    Returns:
        str: Model identifier for GPT-4.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    return "gpt-4"  # Ensure you have access to GPT-4

def clean_text(text: str) -> str:
    """
    Cleans the extracted text by removing unwanted artifacts.

    Args:
        text (str): The text to clean.

    Returns:
        str: Cleaned text.
    """
    # Remove variations of 'swipe right' with flexible spacing and word boundaries
    text = re.sub(r'\bswipe\s+right\b', '', text, flags=re.IGNORECASE)
    # Remove other known artifacts if any (add more patterns as needed)
    # Example: text = re.sub(r'\bartifact_pattern\b', '', text, flags=re.IGNORECASE)
    # Remove extra whitespace and line breaks
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_chunks(text: str, max_length: int = 500) -> List[str]:
    """
    Splits the text into smaller chunks using sentence tokenization.

    Args:
        text (str): The text to split.
        max_length (int): Maximum length of each chunk.

    Returns:
        List[str]: List of text chunks.
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def extract_answer_spans(text: str, top_n: int = 5) -> List[str]:
    """
    Extracts main topics and subtopics from the text using KeyBERT.

    Args:
        text (str): The text from which to extract answer spans.
        top_n (int): The number of top answer spans to extract.

    Returns:
        List[str]: List of extracted answer spans (main topics/subtopics).
    """
    # Extract keywords/phrases using KeyBERT with corrected parameters
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),  # Limit to unigrams and bigrams
        stop_words='english',
        use_maxsum=False,               # Disable MaxSum similarity
        use_mmr=True,                   # Enable Maximal Marginal Relevance
        diversity=0.7,                  # Adjust diversity for better coverage
        nr_candidates=20,
        top_n=top_n
    )
    # Extract only the phrases
    answer_spans = [phrase for phrase, score in keywords]
    
    # Filter out generic terms if necessary
    GENERIC_TERMS = {'machine learning', 'artificial intelligence', 'ai'}
    answer_spans = [phrase for phrase in answer_spans if phrase.lower() not in GENERIC_TERMS]
    
    logger.debug(f"Filtered Answer Spans: {answer_spans}")
    
    return answer_spans

def is_answer_in_chunk(answer: str, chunk: str, threshold: float = 0.55) -> bool:
    """
    Checks if the answer is approximately in the chunk using token overlap.

    Args:
        answer (str): The answer span to check.
        chunk (str): The text chunk.
        threshold (float): Similarity threshold.

    Returns:
        bool: True if answer is in chunk, False otherwise.
    """
    answer_tokens = set(nltk.word_tokenize(answer.lower()))
    chunk_tokens = set(nltk.word_tokenize(chunk.lower()))
    common_tokens = answer_tokens.intersection(chunk_tokens)
    if len(answer_tokens) == 0:
        return False
    similarity = len(common_tokens) / len(answer_tokens)
    logger.debug(f"Similarity between answer and chunk: {similarity}")
    return similarity >= threshold

def process_uploaded_pdfs(uploaded_files) -> List[Document]:
    """
    Processes uploaded PDF files, extracts text and images, and stores documents.

    Args:
        uploaded_files (List[UploadedFile]): List of uploaded PDF files.

    Returns:
        List[Document]: List of Document objects with text and image data.
    """
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file.flush()
                loader = PyMuPDFLoader(temp_file.name)
                docs = loader.load()
                if docs:
                    for doc in docs:
                        content = doc.page_content.strip()
                        if content:
                            # Clean the content
                            content = clean_text(content)
                            doc.page_content = content  # Update the document content
                            documents.append(doc)
                    logger.info(f"Processed file: {uploaded_file.name}, Pages extracted: {len(docs)}")
                else:
                    logger.warning(f"No pages extracted from file: {uploaded_file.name}")
                    st.warning(f"No content was extracted from {uploaded_file.name}. Please check the PDF.")
            os.remove(temp_file.name)
    st.session_state.docs = documents  # Store in session state
    return documents

def extract_images_from_page(pdf_path: str, page_number: int) -> List[Image.Image]:
    """
    Extracts images from a specific page of a PDF.

    Args:
        pdf_path (str): Path to the PDF file.
        page_number (int): Page number to extract images from.

    Returns:
        List[Image.Image]: List of extracted images.
    """
    images = []
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_number - 1)  # zero-based index
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    except Exception as e:
        logger.error(f"Error extracting images from page {page_number}: {e}")
    finally:
        doc.close()
    return images

def setup_retriever(documents: List[Document]) -> Tuple[BM25Okapi, HuggingFaceEmbeddings, List[str]]:
    """
    Sets up the retriever using BM25 and embedding-based retrieval.

    Args:
        documents (List[Document]): List of Document objects with text and images.

    Returns:
        Tuple[BM25Okapi, HuggingFaceEmbeddings, List[str]]: BM25 model, embedding model, and texts.
    """
    texts = [doc.page_content for doc in documents]
    embedding_model = load_retrieval_model()
    bm25 = load_bm25(texts)
    return bm25, embedding_model, texts

def is_question_about_topic(question: str, topic: str, model: SentenceTransformer, threshold: float = 0.55) -> bool:
    """
    Checks if the question is semantically related to the main topic.

    Args:
        question (str): The generated question.
        topic (str): The main topic.
        model (SentenceTransformer): The sentence transformer model for embeddings.
        threshold (float): The similarity threshold.

    Returns:
        bool: True if related, False otherwise.
    """
    question_embedding = model.encode(question, convert_to_tensor=True)
    topic_embedding = model.encode(topic, convert_to_tensor=True)
    similarity = util.cos_sim(question_embedding, topic_embedding).item()
    logger.debug(f"Similarity between question and topic ('{topic}'): {similarity}")
    return similarity >= threshold

def generate_questions(tokenizer, model, chunk: str, answer: str, similarity_model: SentenceTransformer, threshold: float, num_return_sequences: int =1) -> Tuple[List[str], List[Tuple[str, float]]]:
    """
    Generates high-quality, contextually relevant questions based on a text chunk and a highlighted answer.

    Args:
        tokenizer: The tokenizer for the question generation model.
        model: The pre-trained question generation model.
        chunk (str): A paragraph or chunk of text from the document.
        answer (str): The highlighted main topic or subtopic.
        similarity_model (SentenceTransformer): Model to compute similarity between question and topic.
        threshold (float): Similarity threshold.
        num_return_sequences (int): Number of questions to generate.

    Returns:
        Tuple[List[str], List[Tuple[str, float]]]: A tuple containing a list of accepted questions and a list of tuples with rejected questions and their similarity scores.
    """
    # Optimized Prompt focusing on main topics
    prompt = (
        f"You are an expert educator specializing in creating challenging and insightful questions to test comprehension of complex materials.\n"
        f"Given the following paragraph extracted from a document and a highlighted main topic or subtopic within that paragraph, generate a clear, contextually relevant, and thought-provoking question that requires an understanding of the material to answer.\n\n"
        f"Instructions:\n"
        f"- The question should be directly related to the provided paragraph and the highlighted main topic.\n"
        f"- Focus on the significance, implications, or mechanisms of the main topic within the context.\n"
        f"- Ensure the question is open-ended and promotes critical thinking.\n"
        f"- Avoid yes/no questions or questions that can be answered with a single word.\n"
        f"- Use clear and precise language appropriate for a graduate-level audience.\n"
        f"- Do not introduce or reference topics outside of the provided paragraph and highlighted main topic.\n\n"
        f"Example:\n"
        f"Paragraph:\n"
        f'"""\n'
        f"Artificial Intelligence (AI) has transformed various industries by enabling machines to learn from data and make decisions. One of the key aspects of AI is machine learning, which focuses on the development of algorithms that can improve over time without being explicitly programmed.\n"
        f'"""\n'
        f"Highlighted Main Topic: machine learning\n"
        f"Question: How does machine learning contribute to the ability of AI systems to improve their performance over time without explicit programming?\n\n"
        f"Now, generate the question based on the following:\n\n"
        f"Paragraph:\n"
        f'"""\n{chunk}\n"""\n'
        f"Highlighted Main Topic: {answer}\n"
        f"Question:"
    )

    # Debug: Display the prompt
    logger.debug(f"Generated Prompt:\n{prompt}")

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)

    # Generate questions using optimized parameters
    try:
        outputs = model.generate(
            input_ids,
            max_length=128,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
    except Exception as e:
        logger.error(f"Error during question generation: {e}")
        return [], []

    accepted_questions = []
    rejected_questions = []
    for output in outputs:
        q = tokenizer.decode(output, skip_special_tokens=True).strip()

        # Clean up the generated question
        q = re.sub(r'^(Question:)\s*', '', q, flags=re.IGNORECASE)
        q = q.strip()

        # Ensure the question ends with a question mark
        if not q.endswith('?'):
            q += '?'

        # Validate the question length and content
        if len(q.split()) > 5 and not q.lower().startswith('answer'):
            # Check if the question is about the main topic
            if is_question_about_topic(q, answer, similarity_model, threshold):
                accepted_questions.append(q)
            else:
                # Calculate similarity for rejection logging
                similarity = util.cos_sim(
                    similarity_model.encode(q, convert_to_tensor=True),
                    similarity_model.encode(answer, convert_to_tensor=True)
                ).item()
                if similarity < 0.3:
                    rejected_questions.append((q, similarity))
                else:
                    logger.warning(f"Ignored somewhat relevant question: '{q}' with similarity {similarity}")
        else:
            logger.warning(f"Ignored short or incomplete question: '{q}'")

    # Remove duplicates while preserving order
    unique_accepted_questions = list(dict.fromkeys(accepted_questions))
    unique_rejected_questions = list(dict.fromkeys(rejected_questions))

    # Log the generated questions before filtering
    logger.debug(f"Accepted Questions: {unique_accepted_questions}")
    logger.debug(f"Rejected Questions: {unique_rejected_questions}")

    return unique_accepted_questions, unique_rejected_questions

def filter_semantic_duplicates(questions: List[str], model: SentenceTransformer, threshold: float = 0.555) -> List[str]:
    """
    Filters out semantically similar questions based on cosine similarity.

    Args:
        questions (List[str]): List of questions to filter.
        model (SentenceTransformer): Sentence transformer model for embeddings.
        threshold (float): Cosine similarity threshold above which questions are considered duplicates.

    Returns:
        List[str]: Filtered list of unique questions.
    """
    if not questions:
        return []
    embeddings = model.encode(questions, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)
    unique_questions = []
    seen = set()
    for idx, question in enumerate(questions):
        if idx in seen:
            continue
        unique_questions.append(question)
        similar_indices = (cosine_scores[idx] >= threshold).nonzero(as_tuple=True)[0].tolist()
        for sim_idx in similar_indices:
            seen.add(sim_idx)
    return unique_questions

def extract_named_entities(text: str, nlp_model) -> List[Dict[str, Any]]:
    """
    Extracts named entities from the text using spaCy.

    Args:
        text (str): The text from which to extract named entities.
        nlp_model: The spaCy NLP model.

    Returns:
        List[Dict[str, Any]]: List of extracted named entities with their labels.
    """
    doc = nlp_model(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    logger.debug(f"Extracted Named Entities: {entities}")
    return entities

def generate_ner_questions(entities: List[Dict[str, Any]]) -> List[str]:
    """
    Generates questions based on extracted named entities.

    Args:
        entities (List[Dict[str, Any]]): List of named entities with their labels.

    Returns:
        List[str]: List of generated questions based on named entities.
    """
    ner_questions = []
    for ent in entities:
        entity = ent["text"]
        label = ent["label"]  # Corrected from 'label_' to 'label'
        if label in ["ORG", "GPE", "LOC", "EVENT", "PRODUCT", "WORK_OF_ART", "LAW"]:
            question = f"What is {entity}?"
            ner_questions.append(question)
            question = f"Describe {entity}."
            ner_questions.append(question)
            question = f"Explain the significance of {entity}."
            ner_questions.append(question)
    # Remove duplicates
    ner_questions = list(dict.fromkeys(ner_questions))
    logger.debug(f"NER-based Questions: {ner_questions}")
    return ner_questions

def get_answer_vlm(vlm_model, query: str, retrieved_docs: List[Document]) -> str:
    """
    Generates an answer to the user's query using OpenAI's GPT-4 model.

    Args:
        vlm_model (str): Model identifier for GPT-4.
        query (str): User's query.
        retrieved_docs (List[Document]): List of retrieved Document objects.

    Returns:
        str: Generated answer.
    """
    # Concatenate relevant text from retrieved documents
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Construct the prompt for the model
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    try:
        response = openai.ChatCompletion.create(
            model=vlm_model,
            messages=[
                {"role": "system", "content": "You are a knowledgeable and helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.55,
        )
        answer = response.choices[0].message['content'].strip()
        logger.debug(f"Generated Answer: {answer}")
        return answer
    except AuthenticationError:
        logger.error("Authentication with OpenAI API failed. Check your API key.")
        return "Authentication with OpenAI API failed. Please check your API key."
    except RateLimitError:
        logger.error("Rate limit exceeded for OpenAI API.")
        return "Rate limit exceeded. Please try again later."
    except OpenAIError as e:
        logger.error(f"An OpenAI API error occurred: {e}")
        return "An error occurred while generating the answer."
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "An unexpected error occurred while generating the answer."

def summarize_text(text: str, summarizer, max_length: int = 150) -> str:
    """
    Summarizes the given text using the provided summarization pipeline.

    Args:
        text (str): The text to summarize.
        summarizer: The HuggingFace summarization pipeline.
        max_length (int): Maximum length of the summary.

    Returns:
        str: Summarized text.
    """
    try:
        summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return text  # Return original text if summarization fails

@st.cache_resource
def load_summarization_model():
    """
    Loads the summarization model.

    Returns:
        pipeline: HuggingFace summarization pipeline.
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

def initialize_session_state():
    """
    Initializes necessary session state variables.
    """
    if 'generated_questions' not in st.session_state:
        st.session_state.generated_questions = []
    if 'custom_generated_questions' not in st.session_state:
        st.session_state.custom_generated_questions = []
    if 'additional_generated_questions_from_initial' not in st.session_state:
        st.session_state.additional_generated_questions_from_initial = []
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = None
    if 'answers' not in st.session_state:
        st.session_state.answers = {}
    if 'questions_generated' not in st.session_state:
        st.session_state.questions_generated = False
    if 'prev_num_questions' not in st.session_state:
        st.session_state.prev_num_questions = 3
    if 'uploaded_files_hashes' not in st.session_state:
        st.session_state.uploaded_files_hashes = []
    if 'bm25' not in st.session_state:
        st.session_state.bm25 = None
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = None
    if 'texts' not in st.session_state:
        st.session_state.texts = []
    if 'docs' not in st.session_state:
        st.session_state.docs = []
    if 'bm25_scores' not in st.session_state:
        st.session_state.bm25_scores = []
    if 'selected_documents' not in st.session_state:
        st.session_state.selected_documents = []
    if 'named_entities' not in st.session_state:
        st.session_state.named_entities = []
    if 'ner_questions' not in st.session_state:
        st.session_state.ner_questions = []
    # Initialize similarity_threshold and previous value
    if 'similarity_threshold' not in st.session_state:
        st.session_state.similarity_threshold = 0.70
    if 'prev_similarity_threshold' not in st.session_state:
        st.session_state.prev_similarity_threshold = 0.70

def cleanup_old_annoy_indexes(current_index_path: str):
    """
    Deletes old Annoy index files to save storage and prevent conflicts.

    Args:
        current_index_path (str): The current Annoy index file path to retain.
    """
    existing_files = [f for f in os.listdir('.') if f.startswith('annoy_index_') and f.endswith('.ann')]
    for file_name in existing_files:
        if file_name != current_index_path:
            try:
                os.remove(file_name)
                logger.info(f"Deleted Annoy index file: {file_name}")
            except Exception as e:
                logger.error(f"Error deleting Annoy index file '{file_name}': {e}")
                st.error(f"An error occurred while deleting Annoy index file '{file_name}'.")

# -----------------------------
# Main Streamlit App
# -----------------------------

def main():
    initialize_session_state()

    # PDF uploader
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    # Slider for number of questions to generate
    num_questions = st.slider("Number of questions to generate:", min_value=3, max_value=20, value=3)

    # Slider for similarity_threshold
    similarity_threshold = st.slider(
        "Similarity Threshold:",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.similarity_threshold,
        step=0.05,
        help="Adjust the similarity threshold for question relevance."
    )

    # Display current similarity threshold
    st.write(f"**Current Similarity Threshold:** {similarity_threshold}")

    # Check if the similarity_threshold slider value has changed
    if similarity_threshold != st.session_state.prev_similarity_threshold:
        logger.info(f"Similarity threshold changed from {st.session_state.prev_similarity_threshold} to {similarity_threshold}. Regenerating questions.")
        st.session_state.questions_generated = False  # Reset the flag
        st.session_state.prev_similarity_threshold = similarity_threshold  # Update the stored value

    # Compute current list of uploaded file hashes
    current_uploaded_files_hashes = [compute_file_hash(file) for file in uploaded_files]
    logger.debug(f"Previous uploaded file hashes: {st.session_state.uploaded_files_hashes}")
    logger.debug(f"Current uploaded file hashes: {current_uploaded_files_hashes}")

    # Display uploaded files and their number of pages for verification
    if uploaded_files:
        st.write("**Uploaded Files and Number of Pages:**")
        for file in uploaded_files:
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                num_pages = doc.page_count
            st.write(f"- **{file.name}** (Pages: {num_pages})")
        # Reset the file pointer after reading
        for file in uploaded_files:
            file.seek(0)

    # Check if the list of uploaded file hashes has changed (addition, deletion, modification)
    if current_uploaded_files_hashes != st.session_state.uploaded_files_hashes:
        logger.info("Detected change in uploaded files. Resetting session state and Annoy index.")
        st.session_state.questions_generated = False  # Reset the flag
        st.session_state.uploaded_files_hashes = current_uploaded_files_hashes  # Update the list

        # Clear previous questions and answers
        st.session_state.generated_questions = []
        st.session_state.custom_generated_questions = []
        st.session_state.additional_generated_questions_from_initial = []
        st.session_state.selected_question = None
        st.session_state.answers = {}
        st.session_state.named_entities = []
        st.session_state.ner_questions = []

        # Delete the existing Annoy index if it exists
        cleanup_old_annoy_indexes("annoy_index.ann")
        if os.path.exists("annoy_index.ann"):
            st.success("Previous data cleared. Processing new uploads...")
        else:
            logger.info("No existing Annoy index found. Skipping deletion.")

    # Check if PDFs are uploaded and questions need to be generated
    if uploaded_files and not st.session_state.questions_generated:
        with st.spinner("Processing uploaded PDFs..."):
            # Process uploaded PDFs
            docs = process_uploaded_pdfs(uploaded_files)
            if not docs:
                st.error("No content was found in the uploaded PDFs.")
                logger.warning("No documents extracted from uploaded PDFs.")
                return

            # Setup retriever
            bm25, embedding_model, texts = setup_retriever(docs)
            st.session_state.bm25 = bm25
            st.session_state.embedding_model = embedding_model
            st.session_state.texts = texts

            # Create Annoy index
            index_path = "annoy_index.ann"
            vectorstore = Annoy.from_texts(
                texts,
                embedding_model,
                index_path=index_path,
                n_trees=10
            )
            st.session_state.vectorstore_annoy = vectorstore

            st.success("Vector store created from uploaded PDFs!")

            # Extract Named Entities
            content = " ".join(texts)
            logger.debug(f"Content for NER: {content[:500]}...")  # Log the first 500 characters
            named_entities = extract_named_entities(content, nlp)
            st.session_state.named_entities = named_entities

            # Generate NER-based questions
            ner_questions = generate_ner_questions(named_entities)
            st.session_state.ner_questions = ner_questions

            # Generate questions using the content from the uploaded PDFs
            if content:
                tokenizer, qg_model = load_qg_model()
                similarity_model = SentenceTransformer('all-MiniLM-L6-v2')  # Initialize once
                generated_questions = []
                # Split content into chunks for question generation
                chunks = split_into_chunks(content, max_length=500)
                logger.debug(f"Number of chunks created: {len(chunks)}")

                # Shuffle chunks to ensure diverse topics
                secrets.SystemRandom().shuffle(chunks)

                for idx, chunk in enumerate(chunks):
                    if len(generated_questions) >= num_questions:
                        break  # Stop if desired number of questions is reached
                    logger.debug(f"Processing chunk {idx+1}/{len(chunks)}: {chunk[:100]}...")  # Log first 100 characters of the chunk
                    # Extract answer spans from the chunk
                    answer_spans = extract_answer_spans(chunk, top_n=5)
                    logger.debug(f"Extracted answer spans from chunk {idx+1}: {answer_spans}")
                    if not answer_spans:
                        logger.warning("No answer spans found in the current chunk.")
                        continue  # Skip if no answer spans are found

                    # Select the top topic for question generation
                    top_topic = answer_spans[0]
                    logger.debug(f"Selected Top Topic: {top_topic}")

                    # Check if the answer is approximately in the chunk
                    if not is_answer_in_chunk(top_topic, chunk, threshold=similarity_threshold):
                        logger.warning(f"Answer '{top_topic}' not found in the chunk. Skipping.")
                        continue

                    # Generate questions
                    questions, rejected = generate_questions(
                        tokenizer,
                        qg_model,
                        chunk,
                        top_topic,
                        similarity_model,
                        threshold=similarity_threshold,
                        num_return_sequences=1
                    )
                    if questions:
                        # Append the question
                        generated_questions.extend(questions)
                        logger.info(f"Generated Question from Chunk {idx+1}: {questions[0]}")
                    else:
                        logger.warning(f"No valid question generated from chunk {idx+1}.")

                # Deduplicate generated questions
                unique_accepted_questions = list(dict.fromkeys(generated_questions))
                logger.debug(f"Number of unique accepted questions after deduplication: {len(unique_accepted_questions)}")

                # Combine NER-based questions and generated questions
                all_questions = unique_accepted_questions + ner_questions
                unique_all_questions = list(dict.fromkeys(all_questions))
                logger.debug(f"Total combined questions before semantic deduplication: {len(unique_all_questions)}")

                # Deduplicate semantic duplicates based on the current similarity_threshold
                unique_all_questions = filter_semantic_duplicates(unique_all_questions, similarity_model, threshold=similarity_threshold)
                logger.debug(f"Number of unique all questions after semantic deduplication: {len(unique_all_questions)}")

                # Ensure at least the requested number of questions
                if len(unique_all_questions) >= num_questions:
                    limited_questions = unique_all_questions[:num_questions]
                    logger.info(f"Number of questions limited to: {len(limited_questions)}")
                    st.session_state.generated_questions = limited_questions
                    st.session_state.questions_generated = True  # Set the flag to True
                else:
                    if unique_all_questions:
                        st.session_state.generated_questions = unique_all_questions
                        st.session_state.questions_generated = True
                        logger.info(f"Generated {len(unique_all_questions)} questions out of requested {num_questions}.")
                        st.warning(f"Only {len(unique_all_questions)} questions were generated. Some chunks may not have yielded meaningful questions.")
                    else:
                        st.warning("No questions were generated. Please try uploading a different PDF or increasing the number of questions.")
                        return

                # Display a snippet of the content for verification
                st.write("**Content Snippet Used for Question Generation:**")
                st.write(content[:500] + "..." if len(content) > 500 else content)

    # Display generated questions as clickable buttons
    if 'generated_questions' in st.session_state and st.session_state.generated_questions:
        st.subheader("Generated Questions:")

        # Add the "Refresh Questions" button with a unique key
        if st.button("Refresh Questions", key="refresh_questions_button"):
            logger.info("Refresh Questions button clicked.")
            st.session_state.questions_generated = False  # Reset the flag to trigger regeneration

        for idx, question in enumerate(st.session_state.generated_questions):
            # Use a unique prefix to namespace the keys
            button_key = f"generated_question_{idx}"
            if st.button(question, key=button_key):
                logger.info(f"Question clicked: {question}")
                # Check if the answer already exists to avoid redundant calls
                if question not in st.session_state.answers:
                    with st.spinner("Fetching answer..."):
                        try:
                            vlm_model = load_vlm_model()  # Load your VLM here
                            retrieved_docs = st.session_state.vectorstore_annoy.similarity_search(question, k=3)
                            answer = get_answer_vlm(vlm_model, question, retrieved_docs)
                            st.session_state.answers[question] = answer
                            logger.info(f"Answer fetched for question: {question}")
                        except Exception as e:
                            logger.error(f"Error fetching answer for question '{question}': {e}")
                            st.error(f"An error occurred while fetching the answer for the question: {question}")

                st.session_state.selected_question = question

        # Display the answer if available
        if st.session_state.selected_question and st.session_state.selected_question in st.session_state.answers:
            st.markdown(f"**Answer to:** {st.session_state.selected_question}")
            st.write(st.session_state.answers[st.session_state.selected_question])
            
            # Feedback Mechanism
            rating = st.radio(
                "How relevant is this question?", 
                ("Highly Relevant", "Somewhat Relevant", "Not Relevant"), 
                key=f"rating_generated_{st.session_state.selected_question}"
            )
            if rating == "Highly Relevant":
                st.success("Thank you for your feedback!")
            elif rating == "Somewhat Relevant":
                st.info("Thank you! We'll strive to improve.")
            elif rating == "Not Relevant":
                st.warning("We're sorry. We'll work on generating better questions.")

            # Generate two additional questions from the answer
            answer_text = st.session_state.answers[st.session_state.selected_question]
            tokenizer, qg_model = load_qg_model()
            similarity_model = SentenceTransformer('all-MiniLM-L6-v2')  # Initialize once
            additional_questions = []
            answer_chunks = split_into_chunks(answer_text, max_length=500)
            for chunk in answer_chunks[:1]:  # Process only the first chunk for simplicity
                answer_spans = extract_answer_spans(chunk, top_n=2)
                for span in answer_spans:
                    questions, _ = generate_questions(
                        tokenizer,
                        qg_model,
                        chunk,
                        span,
                        similarity_model,
                        threshold=similarity_threshold,
                        num_return_sequences=1
                    )
                    if questions:
                        additional_questions.extend(questions)
                        if len(additional_questions) >=2:
                            break
                if len(additional_questions) >=2:
                    break

            # Deduplicate and limit to two questions
            unique_additional_questions = list(dict.fromkeys(additional_questions))[:2]
            if unique_additional_questions:
                st.session_state.additional_generated_questions_from_initial.extend(unique_additional_questions)
                st.success("Additional Questions Generated from Answer:")
                for idx, question in enumerate(unique_additional_questions):
                    # Use a unique prefix to namespace the keys
                    button_key = f"additional_initial_question_{idx}"
                    if st.button(question, key=button_key):
                        logger.info(f"Additional Question clicked: {question}")
                        # Check if the answer already exists to avoid redundant calls
                        if question not in st.session_state.answers:
                            with st.spinner("Fetching answer..."):
                                try:
                                    vlm_model = load_vlm_model()  # Load your VLM here
                                    retrieved_docs = st.session_state.vectorstore_annoy.similarity_search(question, k=3)
                                    answer_new = get_answer_vlm(vlm_model, question, retrieved_docs)
                                    st.session_state.answers[question] = answer_new
                                    logger.info(f"Answer fetched for additional question: {question}")
                                except Exception as e:
                                    logger.error(f"Error fetching answer for additional question '{question}': {e}")
                                    st.error(f"An error occurred while fetching the answer for the question: {question}")

                        st.session_state.selected_question = question

                # Display the answer if available
                if st.session_state.selected_question and st.session_state.selected_question in st.session_state.answers:
                    st.markdown(f"**Answer to:** {st.session_state.selected_question}")
                    st.write(st.session_state.answers[st.session_state.selected_question])
                    
                    # Feedback Mechanism
                    rating = st.radio(
                        "How relevant is this question?", 
                        ("Highly Relevant", "Somewhat Relevant", "Not Relevant"), 
                        key=f"rating_additional_initial_{st.session_state.selected_question}"
                    )
                    if rating == "Highly Relevant":
                        st.success("Thank you for your feedback!")
                    elif rating == "Somewhat Relevant":
                        st.info("Thank you! We'll strive to improve.")
                    elif rating == "Not Relevant":
                        st.warning("We're sorry. We'll work on generating better questions.")

    # Generate and display two clickable questions from custom query responses
    if 'custom_generated_questions' in st.session_state and st.session_state.custom_generated_questions:
        st.subheader("Additional Questions from Your Query:")
        for idx, (question, similarity) in enumerate(st.session_state.custom_generated_questions):
            # Use a unique prefix to namespace the keys
            button_key = f"custom_question_{idx}"
            if st.button(question, key=button_key):
                logger.info(f"Custom-generated Question clicked: {question}")
                # Check if the answer already exists to avoid redundant calls
                if question not in st.session_state.answers:
                    with st.spinner("Fetching answer..."):
                        try:
                            vlm_model = load_vlm_model()  # Load your VLM here
                            retrieved_docs = st.session_state.vectorstore_annoy.similarity_search(question, k=3)
                            answer = get_answer_vlm(vlm_model, question, retrieved_docs)
                            st.session_state.answers[question] = answer
                            logger.info(f"Answer fetched for custom-generated question: {question}")
                        except Exception as e:
                            logger.error(f"Error fetching answer for custom-generated question '{question}': {e}")
                            st.error(f"An error occurred while fetching the answer for the question: {question}")

                st.session_state.selected_question = question

        # Display the answer if available
        if st.session_state.selected_question and st.session_state.selected_question in st.session_state.answers:
            st.markdown(f"**Answer to:** {st.session_state.selected_question}")
            st.write(st.session_state.answers[st.session_state.selected_question])
            
            # Feedback Mechanism
            rating = st.radio(
                "How relevant is this question?", 
                ("Highly Relevant", "Somewhat Relevant", "Not Relevant"), 
                key=f"rating_custom_{st.session_state.selected_question}"
            )
            if rating == "Highly Relevant":
                st.success("Thank you for your feedback!")
            elif rating == "Somewhat Relevant":
                st.info("Thank you! We'll strive to improve.")
            elif rating == "Not Relevant":
                st.warning("We're sorry. We'll work on generating better questions.")

            # Generate two additional questions from the answer
            # (Optional: Can be enabled if needed)
            # ...

    # Custom query from user
    st.subheader("Ask a Custom Question")
    user_question = st.text_input("Enter your question:")

    if st.button("Submit Custom Query", key="submit_custom_query"):
        if not user_question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Processing..."):
                try:
                    vlm_model = load_vlm_model()  # Load your VLM here
                    # Retrieve relevant documents
                    retrieved_docs = st.session_state.vectorstore_annoy.similarity_search(user_question, k=3)
                    # Generate answer using VLM
                    answer = get_answer_vlm(vlm_model, user_question, retrieved_docs)
                    if answer:
                        st.write("**Answer:**")
                        st.write(answer)
                        logger.info(f"Custom query answered: {user_question}")
                        
                        # Generate two questions from the answer
                        tokenizer, qg_model = load_qg_model()
                        similarity_model = SentenceTransformer('all-MiniLM-L6-v2')  # Initialize once
                        additional_questions = []
                        answer_chunks = split_into_chunks(answer, max_length=500)
                        for chunk in answer_chunks[:1]:  # Process only the first chunk for simplicity
                            answer_spans = extract_answer_spans(chunk, top_n=2)
                            for span in answer_spans:
                                questions, _ = generate_questions(
                                    tokenizer,
                                    qg_model,
                                    chunk,
                                    span,
                                    similarity_model,
                                    threshold=similarity_threshold,
                                    num_return_sequences=1
                                )
                                if questions:
                                    additional_questions.extend(questions)
                                    if len(additional_questions) >=2:
                                        break
                            if len(additional_questions) >=2:
                                break

                        # Deduplicate and limit to two questions
                        unique_additional_questions = list(dict.fromkeys(additional_questions))[:2]
                        if unique_additional_questions:
                            st.session_state.custom_generated_questions.extend([(q, similarity_threshold) for q in unique_additional_questions])
                            st.success("Additional Questions Generated from Answer:")
                            for idx, question in enumerate(unique_additional_questions):
                                # Use a unique prefix to namespace the keys
                                button_key = f"additional_custom_query_question_{idx}"
                                if st.button(question, key=button_key):
                                    logger.info(f"Additional Custom Question clicked: {question}")
                                    # Check if the answer already exists to avoid redundant calls
                                    if question not in st.session_state.answers:
                                        with st.spinner("Fetching answer..."):
                                            try:
                                                vlm_model = load_vlm_model()  # Load your VLM here
                                                retrieved_docs = st.session_state.vectorstore_annoy.similarity_search(question, k=3)
                                                answer_new = get_answer_vlm(vlm_model, question, retrieved_docs)
                                                st.session_state.answers[question] = answer_new
                                                logger.info(f"Answer fetched for additional custom question: {question}")
                                            except Exception as e:
                                                logger.error(f"Error fetching answer for additional custom question '{question}': {e}")
                                                st.error(f"An error occurred while fetching the answer for the question: {question}")

                                    st.session_state.selected_question = question

                            # Display the answer if available
                            if st.session_state.selected_question and st.session_state.selected_question in st.session_state.answers:
                                st.markdown(f"**Answer to:** {st.session_state.selected_question}")
                                st.write(st.session_state.answers[st.session_state.selected_question])
                                
                                # Feedback Mechanism
                                rating = st.radio(
                                    "How relevant is this question?", 
                                    ("Highly Relevant", "Somewhat Relevant", "Not Relevant"), 
                                    key=f"rating_additional_custom_query_{st.session_state.selected_question}"
                                )
                                if rating == "Highly Relevant":
                                    st.success("Thank you for your feedback!")
                                elif rating == "Somewhat Relevant":
                                    st.info("Thank you! We'll strive to improve.")
                                elif rating == "Not Relevant":
                                    st.warning("We're sorry. We'll work on generating better questions.")
                    else:
                        st.warning("No answer found for your custom query.")
                except Exception as e:
                    logger.error(f"Error fetching answer for custom query '{user_question}': {e}")
                    st.error("An error occurred while fetching the answer for your custom query.")

    # Reset Application Button
    if st.button("Reset Application", key="reset_application_button"):
        logger.info("Manual reset triggered by user.")
        # Clear all relevant session state variables
        st.session_state.generated_questions = []
        st.session_state.custom_generated_questions = []
        st.session_state.additional_generated_questions_from_initial = []
        st.session_state.selected_question = None
        st.session_state.answers = {}
        st.session_state.questions_generated = False
        st.session_state.uploaded_files_hashes = []
        st.session_state.prev_num_questions = 3
        st.session_state.bm25 = None
        st.session_state.embedding_model = None
        st.session_state.texts = []
        st.session_state.docs = []
        st.session_state.bm25_scores = []
        st.session_state.selected_documents = []
        st.session_state.named_entities = []
        st.session_state.ner_questions = []
        st.session_state.similarity_threshold = 0.70  # Reset similarity_threshold
        st.session_state.prev_similarity_threshold = 0.70  # Reset previous similarity_threshold

        # Delete all Annoy index files
        cleanup_old_annoy_indexes("annoy_index.ann")
        st.success("Application has been reset. Please upload a new PDF to generate questions.")

        # Halt script execution and wait for user interaction
        st.stop()

if __name__ == "__main__":
    main()
