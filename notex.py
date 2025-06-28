import streamlit as st
import pdfplumber
import docx
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from bs4 import BeautifulSoup
import re
import random
from transformers import pipeline,AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering


# Load models
summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").cpu()
summarizer = pipeline("summarization", model=summarizer_model, tokenizer=summarizer_tokenizer, device=-1)

# Q&A model
qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad").cpu()
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer, device=-1)
#sidebar


st.sidebar.header("🎯 My Daily Learning Goal", divider='rainbow')  # Optional for visual break
goal = st.sidebar.slider("Articles to summarize today:", 1, 10, 2)

if 'progress' not in st.session_state:
    st.session_state['progress'] = 0

if st.sidebar.button("🎉 Mark This Article as Learned"):
    st.session_state['progress'] += 1
    if st.session_state['progress'] >= goal:
        st.sidebar.success("🌟 You’ve achieved your goal for today!")  # subtle glow
    else:
        st.sidebar.info(f"📝 {goal - st.session_state['progress']} article(s) left to go!")

# Custom CSS for cleaner look

st.markdown("""
    <style>
        /* Font styling */
        h1, h2, h3 {
            font-family: 'Poppins', sans-serif;
        }

        /* Dark Background Gradient */
        body {
            background: linear-gradient(135deg, #1a001f, #2a003f);
            background-size: 400% 400%;
            animation: gradientBackground 20s ease infinite;
            margin: 0;
            height: 100vh;
            color: #EAD8FF; /* Pale lavender text */
        }

        @keyframes gradientBackground {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Main content container */
        .block-container {
            padding: 1.5rem 2rem;
            border-radius: 12px;
            background-color: #1b1b2f;
            animation: slideUp 1s ease-out;
            box-shadow: 0 0 12px rgba(191, 0, 255, 0.25); /* Electric purple glow */
        }

        @keyframes slideUp {
            0% { transform: translateY(50px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }

        /* Text hover */
        h1:hover, h2:hover, h3:hover {
            color: #FF00FF;
            transform: scale(1.08);
            transition: all 0.3s ease-in-out;
        }

        /* Button styling */
        button[kind="primary"] {
            background-color: #BF00FF;
            color: white;
            border-radius: 8px;
        }

        button:hover {
            background-color: #FF00FF;
            transition: 0.3s ease-in-out;
        }

        /* Custom scroll bar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-thumb {
            background: #BF00FF;
            border-radius: 10px;
        }

        ::-webkit-scrollbar-track {
            background: #1a001f;
        }
            
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <h1 style='text-align: center; color:#BF00FF;'>NoteX</h1>
    <h4 style='text-align: center; color:#EAD8FF;'>“Why Read It All? Let <span style="color:#FF00FF;">NoteX</span> Break It Down.”</h4>
    <hr style='border:1px solid #555;'/>
""", unsafe_allow_html=True)


# --- Extraction Functions ---
def extract_text_from_pdf(uploaded_file):
    text = ''
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

def get_youtube_transcript(url):
    try:
        video_id = url.split("v=")[-1] if "youtube.com" in url else url.split("/")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([i['text'] for i in transcript])
    except Exception as e:
        return str(e)

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all('p')
        return "\n".join([p.get_text() for p in paragraphs if p.get_text().strip()])
    except Exception as e:
        return f"Error: {str(e)}"

# --- Summarization ---
def summarize_by_sections(text):
    sections = re.split(r'\n(?=\d+\.\s|Chapter\s+\d+|Section\s+\d+)', text)
    summaries = {}

    for i, sec in enumerate(sections):
        sec = sec.strip()
        word_count = len(sec.split())
        if word_count > 20:
            short_text = sec[:1000]
            max_len = min(150, max(13, int(word_count * 0.5)))
            min_len = max(10, int(max_len * 0.6))
            try:
                adjusted_max = min(max_len, len(short_text.split()))
                adjusted_min = min(min_len, adjusted_max - 1) if adjusted_max > 1 else 1
                summary = summarizer(short_text, max_length=adjusted_max, min_length=adjusted_min, do_sample=False)
                summaries[f"Section {i+1}"] = summary[0]['summary_text']
            except Exception as e:
                summaries[f"Section {i+1}"] = f"Error: {str(e)}"
        else:
            summaries[f"Section {i+1}"] = "Section too short to summarize."
    return summaries

def answer_question(context, question):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# --- Tabs for Inputs ---
tab1, tab2, tab3, tab4 = st.tabs([" PDF", " DOCX", " YouTube", " Website"])

text = ""

with tab1:
    uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])
    if uploaded_pdf is not None:
        text = extract_text_from_pdf(uploaded_pdf)
        st.success("✅ PDF content extracted!")

with tab2:
    uploaded_docx = st.file_uploader("Upload Word Document", type=["docx"])
    if uploaded_docx is not None:
        text = extract_text_from_docx(uploaded_docx)
        st.success("✅ DOCX content extracted!")

with tab3:
    youtube_link = st.text_input("Paste the YouTube link:")
    if youtube_link:
        text = get_youtube_transcript(youtube_link)
        st.success("✅ YouTube transcript extracted!")

with tab4:
    website_url = st.text_input("Paste the website article link:")
    if website_url:
        text = extract_text_from_url(website_url)
        st.success("✅ Website article extracted!")

# --- View Extracted Text ---
if text:
    with st.expander("View Extracted Text"):
        st.write(text[:1500] + "..." if len(text) > 1500 else text)

# --- Summarization Section ---
if st.button("🪄 Summarize"):
    if text:
        with st.spinner("Summarizing..."):
            section_summaries = summarize_by_sections(text)
            st.subheader("Section-wise Summary")
            for title, summary in section_summaries.items():
                st.markdown(f"#### 🔹 {title}")
                st.info(summary)
            st.session_state['summary'] = " ".join(section_summaries.values())
    else:
        st.error("Please upload or paste something first!")

# --- Q&A Section ---
if 'summary' in st.session_state:
    st.subheader("Ask a Doubt")
    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if question:
            answer = answer_question(st.session_state['summary'], question)
            st.success(f"Answer: {answer}")
        else:
            st.warning("Please type your question.")

import random

def generate_quiz_from_summary(summary_text):
    sentences = summary_text.split('. ')
    if len(sentences) < 3:
        return None, [], None
    question = f"What is a key idea from this summary?"
    correct_answer = random.choice(sentences).strip()
    distractors = random.sample([s for s in sentences if s != correct_answer], k=min(2, len(sentences)-1))
    options = [correct_answer] + distractors
    random.shuffle(options)
    return question, options, correct_answer

if 'summary' in st.session_state:
    st.subheader(" Try a Quiz")

    if st.button("🎯 Generate Quiz"):
        quiz_q, quiz_options, quiz_ans = generate_quiz_from_summary(st.session_state['summary'])
        if quiz_q:
            st.session_state['quiz_q'] = quiz_q
            st.session_state['quiz_options'] = quiz_options
            st.session_state['quiz_ans'] = quiz_ans
            st.session_state['quiz_selected'] = None  # Reset selection
        else:
            st.warning("Not enough content to generate a quiz.")

    if 'quiz_q' in st.session_state:
        st.markdown(f"**Q: {st.session_state['quiz_q']}**")
        st.session_state['quiz_selected'] = st.radio(
            "Choose the correct answer:",
            st.session_state['quiz_options'],
            key="quiz_radio"
        )

        if st.button("Submit Answer"):
            if st.session_state['quiz_selected'] == st.session_state['quiz_ans']:
                st.success(" Correct!")
            else:
                st.error(f" Incorrect. Correct answer was: {st.session_state['quiz_ans']}")


# Show session-based history
if 'history' in st.session_state and st.session_state['history']:
    if st.checkbox(" Show My Learning History"):
        for idx, item in enumerate(st.session_state['history']):
            st.markdown(f"**Q{idx+1}:** {item['question']}")
            st.markdown(f"**Answer:** {item['answer']}")
            st.markdown("---")


# --- Footer ---
st.markdown("""
    <hr>
    <p style='text-align: center; font-size: 11px;'>Done by Varsha | © 2025 EduSummarizer</p>
""", unsafe_allow_html=True)

