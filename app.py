import streamlit as st
import spacy
import language_tool_python
from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import re

# --- Configuration & Setup ---
st.set_page_config(page_title="AI Communication Coach", layout="wide")

@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    tool = language_tool_python.LanguageTool('en-US')
    analyzer = SentimentIntensityAnalyzer()
    return nlp, st_model, tool, analyzer

try:
    nlp, st_model, tool, analyzer = load_models()
except Exception as e:
    st.error(f"Error loading models. Please ensure dependencies are installed and Java is available for LanguageTool. Error: {e}")
    st.stop()

# --- Scoring Logic ---

def analyze_content(text, st_model):
    score = 0
    breakdown = {}
    
    # 1. Salutation (5 pts)
    text_lower = text.lower()
    if any(phrase in text_lower for phrase in ["excited to introduce", "feeling great"]):
        salutation_score = 5
    elif any(phrase in text_lower for phrase in ["good morning", "good afternoon", "hello everyone"]):
        salutation_score = 4
    elif any(word in text_lower for word in ["hi", "hello"]):
        salutation_score = 2
    else:
        salutation_score = 0
    score += salutation_score
    breakdown['Salutation'] = salutation_score

    # 2. Keyword Presence (30 pts)
    # Mandatory (4 pts each, max 20)
    mandatory_keywords = {
        "Name": ["My name is", "I am"],
        "Age": ["years old", "age is"],
        "School/Class": ["study at", "student of", "class", "grade", "school"],
        "Family": ["family", "parents", "brother", "sister", "father", "mother"],
        "Hobbies": ["hobby", "interest", "like to", "love to", "playing", "reading"]
    }
    
    # Optional (2 pts each, max 10)
    optional_keywords = {
        "Origin": ["from", "live in"],
        "Ambition": ["goal", "ambition", "want to become", "future"],
        "Fun Facts": ["fun fact", "strength", "good at"]
    }

    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if not sentences:
        sentences = [text]
        
    embeddings = st_model.encode(sentences, convert_to_tensor=True)
    
    keyword_score = 0
    
    # Check Mandatory
    for category, phrases in mandatory_keywords.items():
        # Simple string matching first for speed/accuracy on exact phrases
        found = False
        for phrase in phrases:
            if phrase.lower() in text_lower:
                found = True
                break
        
        # If not found, try semantic similarity (threshold 0.4)
        if not found:
            phrase_embeddings = st_model.encode(phrases, convert_to_tensor=True)
            cosine_scores = util.cos_sim(embeddings, phrase_embeddings)
            if (cosine_scores > 0.4).any():
                found = True
        
        if found:
            keyword_score += 4
            
    # Cap Mandatory at 20
    keyword_score = min(keyword_score, 20)
    breakdown['Mandatory Keywords'] = keyword_score

    # Check Optional
    opt_score = 0
    for category, phrases in optional_keywords.items():
        found = False
        for phrase in phrases:
            if phrase.lower() in text_lower:
                found = True
                break
        
        if not found:
            phrase_embeddings = st_model.encode(phrases, convert_to_tensor=True)
            cosine_scores = util.cos_sim(embeddings, phrase_embeddings)
            if (cosine_scores > 0.4).any():
                found = True
        
        if found:
            opt_score += 2
            
    # Cap Optional at 10
    opt_score = min(opt_score, 10)
    keyword_score += opt_score
    score += keyword_score
    breakdown['Optional Keywords'] = opt_score

    # 3. Flow (5 pts)
    # Rough check: Salutation at start, Name early, Closing at end?
    # Simplified logic: If Salutation detected AND Name detected, give 5.
    # A strict order check is hard with just text, but let's try index based.
    
    flow_score = 0
    # Find indices of key elements
    salutation_idx = -1
    name_idx = -1
    
    # Heuristic for indices
    for i, sent in enumerate(sentences):
        s_lower = sent.lower()
        if salutation_idx == -1 and any(x in s_lower for x in ["hi", "hello", "good morning", "good afternoon"]):
            salutation_idx = i
        if name_idx == -1 and any(x in s_lower for x in ["my name", "i am"]):
            name_idx = i
            
    if salutation_idx != -1 and name_idx != -1:
        if salutation_idx <= name_idx:
             flow_score = 5
    elif salutation_idx != -1: # If only salutation found at start
        if salutation_idx == 0:
            flow_score = 2 # Partial credit
    
    # Fallback: if score is high enough, assume flow is okay
    if score >= 15 and flow_score == 0:
        flow_score = 5

    score += flow_score
    breakdown['Flow'] = flow_score
    
    return min(score, 40), breakdown

def analyze_speech_rate(word_count, duration_sec):
    if duration_sec <= 0:
        return 0, "N/A"
    
    minutes = duration_sec / 60
    wpm = word_count / minutes
    
    if 111 <= wpm <= 140:
        score = 10
        status = "Ideal"
    elif 141 <= wpm <= 160:
        score = 6
        status = "Fast"
    elif 81 <= wpm <= 110:
        score = 6
        status = "Slow"
    elif wpm > 161:
        score = 2
        status = "Too Fast"
    else: # < 80
        score = 2
        status = "Too Slow"
        
    return score, f"{int(wpm)} WPM ({status})"

def analyze_grammar(text, tool):
    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return 0, 0, 0
        
    # Grammar (10 pts)
    matches = tool.check(text)
    error_count = len(matches)
    errors_per_100 = (error_count / word_count) * 100
    
    grammar_metric = 1 - min(errors_per_100 / 10, 1)
    
    if grammar_metric > 0.9:
        grammar_score = 10
    elif 0.7 <= grammar_metric <= 0.89:
        grammar_score = 8
    elif 0.5 <= grammar_metric <= 0.69:
        grammar_score = 6
    else:
        grammar_score = 4
        
    # Vocabulary (10 pts)
    # TTR
    unique_words = len(set([w.lower() for w in words]))
    ttr = unique_words / word_count
    
    if ttr > 0.9:
        vocab_score = 10
    elif 0.7 <= ttr <= 0.89:
        vocab_score = 8
    elif 0.5 <= ttr <= 0.69:
        vocab_score = 6
    else:
        vocab_score = 2
        
    return grammar_score, vocab_score, errors_per_100

def analyze_clarity(text):
    filler_words = ["um", "uh", "like", "you know", "so", "actually", "basically", "right", "i mean", "well", "kinda", "sort of", "okay", "hmm", "ah"]
    words = text.lower().split()
    word_count = len(words)
    if word_count == 0:
        return 0, 0
        
    filler_count = 0
    for w in words:
        # Simple check, can be improved with regex for phrases like "you know"
        # For phrases, we need to check the text string
        pass

    # Better approach for phrases
    filler_count = 0
    text_lower = text.lower()
    for filler in filler_words:
        # Use regex to find whole words/phrases
        count = len(re.findall(r'\b' + re.escape(filler) + r'\b', text_lower))
        filler_count += count
        
    rate = (filler_count / word_count) * 100
    
    if rate <= 3:
        score = 15
    elif 4 <= rate <= 6:
        score = 12
    elif 7 <= rate <= 9:
        score = 9
    elif 10 <= rate <= 12:
        score = 6
    else:
        score = 3
        
    return score, rate

def analyze_engagement(text, analyzer):
    vs = analyzer.polarity_scores(text)
    pos_prob = vs['compound'] # VADER gives compound -1 to 1. Rubric asks for probability.
    # Let's normalize compound to 0-1 or use 'pos' score?
    # "Positivity probability" usually implies 0-1.
    # VADER 'pos' is the proportion of text that falls in positive category.
    # Let's use 'pos' from VADER as it maps 0-1.
    # OR, maybe the prompt implies a custom model. But VADER is specified.
    # Let's use the 'pos' component.
    
    # Re-reading rubric: "Positivity probability".
    # If using TextBlob, sentiment.polarity is -1 to 1.
    # If using VADER, 'pos' is 0 to 1. Let's use 'pos'.
    
    pos_val = vs['pos']
    # However, 'pos' in VADER is often low for normal sentences (e.g. 0.2).
    # Maybe they mean the compound score normalized?
    # Compound: >= 0.05 is positive.
    # Let's map compound (-1 to 1) to 0-1 roughly: (compound + 1) / 2
    
    prob = (vs['compound'] + 1) / 2
    
    if prob >= 0.9:
        score = 15
    elif 0.7 <= prob <= 0.89:
        score = 12
    elif 0.5 <= prob <= 0.69:
        score = 9
    else:
        score = 3
        
    return score, prob

# --- UI ---

st.title("AI Communication Coach 🎤")
st.markdown("Analyze your self-introduction transcript and get a detailed score.")

col1, col2 = st.columns(2)

with col1:
    transcript = st.text_area("Introduction Transcript", height=300, placeholder="Paste your text here...")

with col2:
    if "duration" not in st.session_state:
        st.session_state.duration = 60

    duration = st.number_input("Audio Duration (seconds)", min_value=1, key="duration")
    if st.button("Generate Score", type="primary"):
        if not transcript.strip():
            st.warning("Please enter a transcript.")
        else:
            with st.spinner("Analyzing..."):
                # 1. Content
                content_score, content_breakdown = analyze_content(transcript, st_model)
                
                # 2. Speech Rate
                word_count = len(transcript.split())
                rate_score, rate_status = analyze_speech_rate(word_count, duration)
                
                # 3. Language
                grammar_score, vocab_score, error_rate = analyze_grammar(transcript, tool)
                lang_score = grammar_score + vocab_score
                
                # 4. Clarity
                clarity_score, filler_rate = analyze_clarity(transcript)
                
                # 5. Engagement
                eng_score, sentiment_prob = analyze_engagement(transcript, analyzer)
                
                total_score = content_score + rate_score + lang_score + clarity_score + eng_score
                
                # Display
                st.success("Analysis Complete!")
                
                st.metric(label="Total Score", value=f"{total_score}/100")
                
                st.subheader("Detailed Breakdown")
                
                # Create a DataFrame for the breakdown
                data = {
                    "Category": ["Content & Structure", "Speech Rate", "Language & Grammar", "Clarity", "Engagement"],
                    "Score": [f"{content_score}/40", f"{rate_score}/10", f"{lang_score}/20", f"{clarity_score}/15", f"{eng_score}/15"],
                    "Details": [
                        f"Salutation: {content_breakdown['Salutation']}, Keywords: {content_breakdown['Mandatory Keywords'] + content_breakdown['Optional Keywords']}, Flow: {content_breakdown['Flow']}",
                        f"{rate_status}",
                        f"Grammar: {grammar_score} (Errors: {error_rate:.1f}%), Vocab: {vocab_score}",
                        f"Filler Rate: {filler_rate:.1f}%",
                        f"Sentiment Score: {sentiment_prob:.2f}"
                    ]
                }
                df = pd.DataFrame(data)
                st.table(df)
                
                # Visual feedback
                if total_score >= 80:
                    st.balloons()
