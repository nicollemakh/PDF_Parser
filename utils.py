import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from gtts import gTTS
import tempfile
import os 
import fitz
from nltk.tokenize import sent_tokenize
import torch
import nltk

nltk.download('punkt')  # download sentence tokenizer once

model_name = "facebook/bart-large-cnn"
grammar_model_name = "vennify/t5-base-grammar-correction"

grammar_tokenizer = AutoTokenizer.from_pretrained(grammar_model_name)
grammar_model = AutoModelForSeq2SeqLM.from_pretrained(grammar_model_name)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Force CPU usage to avoid MPS errors on Mac
device = -1  # CPU device index for pipeline
model.to("cpu")

# Create summarization pipeline using CPU
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

grammar_corrector = pipeline(
    "text2text-generation",
    model=grammar_model,
    tokenizer=grammar_tokenizer,
    device=-1  # CPU usage
)

def grammar_polish(text):
    result = grammar_corrector(text, max_length=512, do_sample=False)
    return result[0]['generated_text']

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"PDF extraction error: {e}")
    return text

def chunk_text(text, chunk_size=3000):
    # Naively split text into chunks of ~3000 characters
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def chunk_text_by_tokens(text, max_tokens=1200):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        tentative = current_chunk + (" " if current_chunk else "") + sentence
        tokens = tokenizer.encode(tentative, add_special_tokens=False)
        if len(tokens) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk = tentative

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def summarize_text(text):
    print("summarize_text called")
    try:
        chunks = chunk_text_by_tokens(text, max_tokens=900)
        print(f"Total chunks: {len(chunks)}")

        partial_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1} length: {len(chunk)} chars, {len(tokenizer.encode(chunk))} tokens")
            if len(chunk.strip()) < 50:
                continue
            result = summarizer(
                chunk,
                max_length=250,
                min_length=100,
                do_sample=False,
                num_beams=4,
                no_repeat_ngram_size=3
            )
            print(f"Chunk {i+1} summary snippet: {result[0]['summary_text'][:100]}")
            partial_summaries.append(result[0]['summary_text'])

        # Combine chunk summaries into one big text
        combined = " ".join(partial_summaries)

        # Re-chunk combined summary to avoid token overflow in final summarization
        combined_chunks = chunk_text_by_tokens(combined, max_tokens=900)

        final_summaries = []
        for chunk in combined_chunks:
            result = summarizer(
                chunk,
                max_length=300,
                min_length=150,
                do_sample=False,
                num_beams=4,
                no_repeat_ngram_size=3
            )
            final_summaries.append(result[0]['summary_text'])

        final_summary = " ".join(final_summaries)

        # Grammar & punctuation polish
        polished_summary = grammar_polish(final_summary)

        return polished_summary

    except Exception as e:
        print(f"Summarization error: {e}")
        return "Summarization failed."


def generate_quiz(text):
    import random
    sentences = [s.strip() for s in text.split('.') if len(s.split()) > 8]
    keywords = set()
    for sent in sentences:
        words = [w.strip('.,') for w in sent.split() if w.istitle()]
        keywords.update(words)
    keywords = list(keywords)
    quiz = []
    for i, sent in enumerate(sentences[:5]):
        words = sent.split()
        answer = None
        for w in words[::-1]:
            if w.istitle():
                answer = w.strip('.,')
                break
        if not answer:
            answer = words[-1].strip('.,')
        wrong_options = random.sample([k for k in keywords if k != answer], k=min(3, len(keywords)-1)) if len(keywords) > 3 else ["OptionA", "OptionB", "OptionC"]
        options = wrong_options + [answer]
        random.shuffle(options)
        question = sent.replace(answer, "_____")
        quiz.append({
            'question': question,
            'options': options,
            'answer': answer
        })
    return quiz

def text_to_speech(text):
    try:
        tts = gTTS(text)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        print(f"TTS error: {e}")
        return None
    
import fitz  # PyMuPDF

def highlight_pdf(pdf_path, query, output_path):
    try:
        doc = fitz.open(pdf_path)
        query_lower = query.lower()

        found = False
        for page in doc:
            # Remove hit_max
            text_instances = page.search_for(query)
            if text_instances:
                found = True
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                highlight.update()

        if not found:
            print(f"No instances of '{query}' found in PDF.")
        
        doc.save(output_path, garbage=4, deflate=True)
        doc.close()
        return output_path
    except Exception as e:
        print(f"Highlighting error: {e}")
        return None