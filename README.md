# 🤖 Automated Metadata Generator

**Intelligent document metadata generation powered by LLama 3.1**

![image](https://github.com/user-attachments/assets/969a6f57-f94e-49b5-9dbe-a3a055631580)


## ✨ Features

- **Smart Text Extraction** - PDF, DOCX, TXT with OCR fallback
- **AI-Powered Summarization** - Llama 3.1 generates intelligent summaries
- **Advanced Keywords** - TF-IDF + Named Entity Recognition
- **Topic Detection** - Automatic theme identification
- **Language Detection** - Multi-language support
- **Document Stats** - Word count, readability metrics
- **Named Entities** - People, organizations, locations extraction
- **JSON Export** - Complete metadata download

## 🚀 Live Demo

**Try it now:** [autometadatagen.streamlit.app](https://autometadatagen.streamlit.app/)

## 🛠️ Quick Start

### Prerequisites
```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler

# Windows
Download Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
Download Poppler: https://blog.alivate.com.au/poppler-windows/
```

### Installation
```bash
git clone https://github.com/KaranS2111/AutoMetaData_Gen.git
cd AutoMetaData_Gen
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Run Locally
```bash
streamlit run app.py
```

## 🔑 Setup

1. Get your free Groq API key: [console.groq.com](https://console.groq.com/keys)
2. Enter API key in the app sidebar
3. Upload your document and extract metadata!

## 📄 Supported Formats

- **PDF** - Including scanned documents (OCR)
- **DOCX** - Microsoft Word documents  
- **TXT** - Plain text files

## 🔄 Pipeline

- **Document Upload** - User uploads PDF/DOCX/TXT file via Streamlit interface
- **Text Extraction** - PyMuPDF extracts text, falls back to OCR for scanned documents
- **AI Processing** - Groq's Llama 3.1 generates summaries and identifies topics
- **Feature Extraction** - spaCy + scikit-learn extract keywords, entities, and statistics
- **JSON Export** - Complete metadata packaged and downloaded as structured JSON

## 🧠 AI Processing

- **Model**: Llama 3.1 8B Instant via Groq
- **Text Limit**: 700,000 characters (truncated if longer)
- **Features**: Summarization, topic extraction, keyword analysis

## 📊 Output Metadata

```json
{
  "filename": "document.pdf",
  "title": "Auto-extracted title",
  "summary": "AI-generated summary",
  "topics_and_themes": ["Topic 1", "Topic 2"],
  "keywords": ["keyword1", "keyword2"],
  "language": "en",
  "statistics": {
    "word_count": 1234,
    "character_count": 5678,
    "sentence_count": 89
  },
  "named_entities": {
    "PERSON": ["John Doe"],
    "ORG": ["Company Name"]
  }
  "created_at" : "2025-06-24T18:39:16.327586"
}
```

## 🏗️ Built With

- **Streamlit** - Web interface
- **Groq** - LLM API (Llama 3.1)
- **spaCy** - NLP processing
- **PyMuPDF** - PDF extraction
- **scikit-learn** - TF-IDF keywords
- **Tesseract** - OCR support

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

**⭐ Star this repo if you find it useful!**
