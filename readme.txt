# EduNLP – AI Study Assistant

## Overview
EduNLP is an AI-powered study assistant that helps students
understand their lecture PDFs.

## Features
- PDF text extraction
- Automatic document summarisation
- Topic explanations
- Question answering
- Source-based answers with page references
- Revision assistance

## Technologies Used
- Python
- Streamlit
- Sentence Transformers
- PyMuPDF
- NumPy
- Ollama

## How to Run

1. Install the required libraries:

   pip install -r requirements.txt

2. Install Ollama and download the model:

   ollama pull qwen2.5:3b

3. Start the application:

   streamlit run app.py