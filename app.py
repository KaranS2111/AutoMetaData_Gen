import streamlit as st
import os
import tempfile
import json
from datetime import datetime
import re
from groq import Groq
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import docx
from langdetect import detect
import spacy
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import subprocess
import importlib.util


# nlp = spacy.load("en_core_web_sm")
st.set_page_config(
    page_title="Automated Metadata Generator", 
    page_icon="📄", 
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metadata-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .keyword-tag {
    background-color: #e0efff; 
    color: #1a1a1a;            
    padding: 0.3rem 0.7rem;
    border-radius: 0.4rem;
    margin: 0.2rem;
    display: inline-block;
    font-size: 0.9rem;
    font-weight: 500;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="main-header">🤖 Automated Metadata Generator</h1>', unsafe_allow_html=True)
st.markdown("Upload a document and get intelligent metadata extraction powered by Groq")


st.sidebar.header("⚙️ Configuration")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", 
                                     help="Get your free API key from https://console.groq.com/keys")


@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy English model not found. Please install with: python -m spacy download en_core_web_sm")
        st.stop()


if groq_api_key:
    client = Groq(api_key=groq_api_key)
    nlp = load_spacy_model()
else:
    st.warning("Please enter your Groq API key in the sidebar to continue")
    st.stop()


def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        doc = fitz.open(tmp_path)
        for page in doc:
            text += page.get_text()
        doc.close()
        
        if len(text.strip()) < 50:
            with st.spinner("Running OCR on PDF..."):
                images = convert_from_path(tmp_path)
                for image in images:
                    text += pytesseract.image_to_string(image, lang='eng')
        
        os.unlink(tmp_path)
        
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""
    
    return text.strip()

def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

def extract_text_from_txt(txt_file):
    try:
        raw = txt_file.read()
        for encoding in ['utf-8', 'latin-1', 'utf-16']:
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        st.error("Could not decode .txt file. Unsupported encoding.")
    except Exception as e:
        st.error(f"Error reading .txt file: {e}")
    return ""


def extract_keywords(text, max_keywords=15):
    text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
    
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform([text_clean])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    
    tfidf_keywords = [(feature_names[i], tfidf_scores[i]) for i in tfidf_scores.argsort()[-20:][::-1]]
    
    doc = nlp(text)
    entities = [(ent.text.lower(), ent.label_) for ent in doc.ents if len(ent.text) > 2]
    
    all_keywords = []
    for kw, score in tfidf_keywords:
        if len(kw) > 2 and score > 0.01:
            all_keywords.append(kw)
    
    for ent, label in entities:
        if label in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT'] and ent not in all_keywords:
            all_keywords.append(ent)
    
    return all_keywords[:max_keywords]

def create_summary(text, max_length=300):
    if len(text.strip()) < 100:
        return text.strip()
    
    chunk_size = 8000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    summaries = []
    
    for i, chunk in enumerate(chunks[:3]):
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating concise, informative summaries. Extract the key points and main ideas from the text."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this text in 2-3 sentences, focusing on the main points:\n\n{chunk}"
                    }
                ],
                model="llama-3.1-8b-instant",
                max_tokens=150,
                temperature=0.3
            )
            summaries.append(response.choices[0].message.content.strip())
        except Exception as e:
            st.warning(f"Error summarizing chunk {i+1}: {e}")
            sentences = chunk.split('. ')[:3]
            summaries.append('. '.join(sentences) + '.')
    
    combined_summary = ' '.join(summaries)
    
    if len(combined_summary) > max_length and len(summaries) > 1:
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Create a final concise summary from these partial summaries."
                    },
                    {
                        "role": "user",
                        "content": f"Combine these summaries into one coherent summary (max 2-3 sentences):\n\n{combined_summary}"
                    }
                ],
                model="llama-3.1-8b-instant",
                max_tokens=100,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except:
            return combined_summary[:max_length] + "..."
    
    return combined_summary

def extract_topics(text, keywords):
    try:
        kw_text = ", ".join(keywords[:10])
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at identifying main topics and themes in documents. Based on the keywords and content, identify 3-5 main topics/themes."
                },
                {
                    "role": "user",
                    "content": f"Based on these keywords: {kw_text}\n\nAnd this text sample: {text[:1000]}...\n\nIdentify the main topics/themes (return as a simple list):"
                }
            ],
            model="llama-3.1-8b-instant",
            max_tokens=100,
            temperature=0.4
        )
        topics_text = response.choices[0].message.content.strip()
        topics = [topic.strip('- ').strip() for topic in topics_text.split('\n') if topic.strip()]
        return topics[:5]
    except Exception as e:
        st.warning(f"Error extracting topics: {e}")
        return keywords[:5]


uploaded_file = st.file_uploader(
    "Choose a file", 
    type=['pdf', 'docx', 'txt'],
    help="Upload PDF, DOCX, or TXT files for metadata extraction"
)

if uploaded_file is not None:
    st.success(f"📎 File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
    
    with st.spinner("🔍 Extracting text from document..."):
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif file_type == 'docx':
            text = extract_text_from_docx(uploaded_file)
        elif file_type == 'txt':
            text = extract_text_from_txt(uploaded_file)
        else:
            st.error("Unsupported file type")
            st.stop()
    
    if not text:
        st.error("Could not extract text from the document")
        st.stop()
    
    st.success(f"✅ Extracted {len(text)} characters from document")
    MAX_CHARS = 500000
    full_text = text  # Keep full version for preview
    if len(text) > MAX_CHARS:
        st.warning(f"Text too long ({len(text)} chars). Using first {MAX_CHARS} characters for metadata.")
        text = full_text[:MAX_CHARS]
    

    with st.expander("📖 Document Preview"):
        st.text_area("Extracted Text (first 1000 characters)", text[:1000], height=200)
    

    with st.spinner("🧠 Generating intelligent metadata..."):
        title = "Untitled"
        lines = text.split('\n')
        for line in lines[:10]:
            line = line.strip()
            if 10 < len(line) < 150 and not line.isdigit():
                title = line
                break

        try:
            language = detect(text)
        except:
            language = "unknown"
        
  
        keywords = extract_keywords(text)

        summary = create_summary(text)

        topics = extract_topics(text, keywords)

        doc = nlp(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        
        for label in entities:
            entities[label] = list(set(entities[label]))[:10]

        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        avg_words_per_sentence = words / max(sentences, 1)

        metadata = {
            "filename": uploaded_file.name,
            "title": title,
            "summary": summary,
            "topics_and_themes": topics,
            "keywords": keywords,
            "language": language,
            "statistics": {
                "word_count": words,
                "character_count": len(text),
                "sentence_count": sentences,
                "avg_words_per_sentence": round(avg_words_per_sentence, 2)
            },
            "named_entities": entities,
            "created_at": datetime.now().isoformat(),
            "extraction_method": "Enhanced with Groq LLM"
        }

    st.markdown("## 📊 Generated Metadata")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="metadata-section">', unsafe_allow_html=True)
        st.markdown("### 📋 Document Overview")
        st.markdown(f"**Title:** {metadata['title']}")
        st.markdown(f"**Language:** {metadata['language']}")
        st.markdown(f"**Summary:**")
        st.write(metadata['summary'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metadata-section">', unsafe_allow_html=True)
        st.markdown("### 🏷️ Topics & Keywords")
        st.markdown("**Main Topics:**")
        for topic in metadata['topics_and_themes']:
            st.markdown(f"• {topic}")
        
        st.markdown("**Keywords:**")
        keyword_html = ""
        for kw in metadata['keywords'][:12]:
            keyword_html += f'<span class="keyword-tag">{kw}</span> '
        st.markdown(keyword_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metadata-section">', unsafe_allow_html=True)
        st.markdown("### 📈 Statistics")
        stats_df = pd.DataFrame(list(metadata['statistics'].items()), columns=['Metric', 'Value'])
        st.dataframe(stats_df, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if metadata['named_entities']:
            st.markdown('<div class="metadata-section">', unsafe_allow_html=True)
            st.markdown("### 🏛️ Named Entities")
            for entity_type, entity_list in list(metadata['named_entities'].items())[:5]:
                if entity_list:
                    st.markdown(f"**{entity_type}:** {', '.join(entity_list[:3])}")
            st.markdown('</div>', unsafe_allow_html=True)
    
 
    st.markdown("## 💾 Download Results")
    
    json_str = json.dumps(metadata, indent=2, ensure_ascii=False)
    st.download_button(
        label="📄 Download Complete Metadata (JSON)",
        data=json_str,
        file_name=f"{uploaded_file.name}_metadata.json",
        mime="application/json",
        use_container_width=True
    )
    
    with st.expander("🔍 View Complete Metadata"):
        st.json(metadata)


st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Groq by Karan Sardar, IITR | Get your API key at [console.groq.com](https://console.groq.com/keys)")
