import gradio as gr
import os
import re
import gc
import math
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import hdbscan
import nltk
from huggingface_hub import hf_hub_download, InferenceClient
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Dynamic Hardware Acceleration
if torch.cuda.is_available():
    hf_device = 0
    torch_device = "cuda"
elif torch.backends.mps.is_available():
    hf_device = "mps"
    torch_device = "mps"
else:
    hf_device = -1
    torch_device = "cpu"
    torch.set_num_threads(2) # HF Free Tier CPU optimization

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)


try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)
from nltk.corpus import words

ENGLISH_STOPWORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 'he', 'him', 'she', 'it', 'they', 'them', 'what', 'which', 'who', 'this', 'that', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'a', 'an', 'the', 'if', 'or', 'as', 'of', 'at', 'by', 'for', 'with', 'about', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'then', 'here', 'there', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'}

ENGLISH_VOCAB = set(words.words()).union(ENGLISH_STOPWORDS)

# Generic severe issues to bypass "Neutral" sentiment bias
GENERIC_SEVERE_ISSUES = {
    'worst', 'terrible', 'awful', 'disgusting', 'horrible', 'scam', 'fraud', 
    'fake', 'broken', 'damaged', 'ruined', 'poison', 'cockroach', 'bug', 
    'insect', 'hair', 'rude', 'unprofessional', 'pathetic', 'useless', 
    'garbage', 'trash', 'stale', 'rotten', 'dirty', 'filthy', 'late', 'delay'
}

# Model Loading
# Hindi to English
mdl = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", dtype=torch.float32, low_cpu_mem_usage=True).to(torch_device)
tknzr = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# Hinglish to English
qwen_path = hf_hub_download(repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF", filename="qwen2.5-3b-instruct-q4_k_m.gguf")
llm_hinglish = Llama(model_path=qwen_path, n_ctx=1024, n_threads=2)

# Sentiment & Embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=torch_device)
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=hf_device)


# Helper functions
def route_language(text, threshold=0.85):
    """
    Routes text based on script and English word threshold.
    If >= 85% of the words are valid English, it goes to the English pool.
    """
    if re.search(r'[\u0900-\u097F]', text): 
        return 'hindi'
    
    words_in_text = re.sub(r'[^\w\s]', '', text.lower()).split()
    if not words_in_text:
        return 'english' # Fallback for empty/punctuation-only
        
    english_word_count = sum(1 for w in words_in_text if w in ENGLISH_VOCAB)
    
    # If 85% or more of the words are English, treat as English
    if (english_word_count / len(words_in_text)) >= threshold:
        return 'english'
        
    return 'hinglish'

def get_caveman(text):
    """Strips grammar and stopwords to compress tokens for the API."""
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    clean = [w for w in words if w not in ENGLISH_STOPWORDS]
    return " ".join(clean) if clean else "empty"

def dynamic_cluster(embeddings):
    """Calculates UMAP/HDBSCAN parameters dynamically based strictly on the POOL size."""
    num_samples = len(embeddings)
    if num_samples < 5: 
        return [0] * num_samples 
    
    n_components = max(3, min(15, int(math.log10(num_samples) * 2)))
    n_neighbors = min(15, max(2, num_samples - 1)) 
    min_cluster_size = max(3, min(500, int(num_samples * 0.02)))
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine', random_state=42)
    reduced = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom')
    return clusterer.fit_predict(reduced)

def get_centric_caveman_samples(cluster_indices, embeddings, full_texts, n=5):
    
    if len(cluster_indices) <= n:
        selected_indices = cluster_indices
    else:
        cluster_embs = embeddings[cluster_indices]
        centroid = np.mean(cluster_embs, axis=0)
        distances = np.linalg.norm(cluster_embs - centroid, axis=1)
        closest_local_indices = np.argsort(distances)[:n]
        selected_indices = [cluster_indices[i] for i in closest_local_indices]
    
    centric_caveman_list = [get_caveman(full_texts[i]) for i in selected_indices]
    return " | ".join(centric_caveman_list)

def call_qwen_api_with_retry(prompt, retries=2):

    for attempt in range(retries):
        try:
            res = client.chat_completion(
                model="Qwen/Qwen2.5-72B-Instruct", 
                messages=[{"role": "user", "content": prompt}], 
                max_tokens=100
            )
            return res.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return f"API Error: {str(e)}"


# Main pipeline function
def execute_pipeline(file_path):
    if not file_path: return "No file provided.", None, "", ""
    
    try:
        df = pd.read_csv(file_path, header=None)
        raw_reviews = df.iloc[:, 0].dropna().astype(str).tolist()[:120] # Limit for PoC speed
    except Exception as e:
        return f"Error reading CSV: {str(e)}", None, "", ""
    
    baseline_tokens = 0
    optimized_tokens = 0
    
    # Language Pooling & Batching
    hindi_pool, hinglish_pool, english_pool = [], [], []
    
    for idx, text in enumerate(raw_reviews):
        baseline_tokens += len(text.split()) + 50
        lang = route_language(text, threshold=0.85)
        if lang == 'hindi': hindi_pool.append((idx, text))
        elif lang == 'hinglish': hinglish_pool.append((idx, text))
        else: english_pool.append((idx, text))

    full_english_texts = [""] * len(raw_reviews)
    
    # Process English
    for idx, text in english_pool:
        full_english_texts[idx] = text

    # Process Hindi
    if hindi_pool:
        batch_size = 16
        tknzr.src_lang = "hin_Deva"
        output_lang = tknzr.convert_tokens_to_ids("eng_Latn")
        
        for i in range(0, len(hindi_pool), batch_size):
            batch = hindi_pool[i:i+batch_size]
            batch_texts = [text for _, text in batch]
            
            inputs = tknzr(batch_texts, return_tensors="pt", padding=True, truncation=True).to(torch_device)
            out = mdl.generate(**inputs, forced_bos_token_id=output_lang)
            translated_batch = tknzr.batch_decode(out, skip_special_tokens=True)
            
            for (idx, _), trans_text in zip(batch, translated_batch):
                full_english_texts[idx] = trans_text

    # Process Hinglish
    if hinglish_pool:
        for idx, text in hinglish_pool:
            prompt = f"<|im_start|>system\nTranslate Hinglish to English.<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            out = llm_hinglish(prompt, max_tokens=100, stop=["<|im_end|>"], echo=False)
            full_english_texts[idx] = out['choices'][0]['text'].strip()

    # 5. Sentiment Pooling
    # Optimization: batch_size=16 speeds up transformer pipelines significantly
    
    sent_results = sentiment_analyzer(full_english_texts, batch_size=16, truncation=True, max_length=512)
    buckets = {"Positive": [], "Neutral": [], "Negative": []}
    
    for i, (text, res) in enumerate(zip(full_english_texts, sent_results)):
        label = res['label'].lower()
        text_lower = text.lower()
        
        # Generic Severe Issue Trap
        if any(k in text_lower for k in GENERIC_SEVERE_ISSUES):
            cat = "Negative"
        elif "negative" in label:
            cat = "Negative"
        elif "positive" in label:
            cat = "Positive"
        else:
            cat = "Neutral"
            
        buckets[cat].append(i)

    insight_markdown = ""
    
    # Process Each Pool Independently
    for sentiment, global_indices in buckets.items():
        if not global_indices: continue
        
        insight_markdown += f"## {sentiment} Feedback Analysis\n"
        
        pool_texts = [full_english_texts[i] for i in global_indices]
        
        # Optimization: normalize_embeddings=True improves UMAP cosine distance speed/accuracy
        pool_embs = embedder.encode(pool_texts, batch_size=32, normalize_embeddings=True)
        
        pool_labels = dynamic_cluster(pool_embs)
        
        for c_id in set(pool_labels):
            if c_id == -1: continue 
            
            local_c_indices = [j for j, lab in enumerate(pool_labels) if lab == c_id]
            
            compressed_input = get_centric_caveman_samples(
                np.array(local_c_indices), 
                pool_embs, 
                pool_texts, 
                n=5
            )
            
            prompt = (
                f"Analyze these representative keyword-sets from a {sentiment} cluster: [{compressed_input}]. "
                "Infer the specific business issue or strength. Output exactly:\n"
                "1. Insight: [1 sentence]\n2. Action: [1 sentence]"
            )
            
            optimized_tokens += len(prompt.split())
            summary = call_qwen_api_with_retry(prompt)

            insight_markdown += f"**Cluster {c_id+1} ({len(local_c_indices)} reviews)**\n{summary}\n\n"

    # Metrics & Visualization
    savings = max(0, 100 - ((optimized_tokens / baseline_tokens) * 100)) if baseline_tokens > 0 else 0
    viability_report = f"""
    ### Scalability & Cost Metrics
    * **Baseline API Tokens (Full Text):** ~{baseline_tokens}
    * **Optimized API Tokens (Centroid-Caveman):** ~{optimized_tokens}
    * **Net Token Savings:** {savings:.1f}%
    * **Architecture:** Stratified Pool Clustering (HDBSCAN)
    """

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.pie([len(buckets[k]) for k in buckets], labels=buckets.keys(), autopct='%1.1f%%', colors=['#99ff99','#66b3ff','#ff9999'])
    
    # Memory Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    return "Analysis Complete", fig, insight_markdown, viability_report

# UI THEME & LAUNCH
dark_theme = gr.themes.Base(
    primary_hue="blue",             
    secondary_hue="zinc",           
    neutral_hue="zinc",             
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#09090b",            
    block_background_fill="#18181b",          
    block_border_width="1px",    
    block_border_color="#27272a",                 
    body_text_color="#f4f4f5",                
    body_text_color_subdued="#a1a1aa",         
    block_title_text_color="#ffffff",
    input_background_fill="#09090b",
    input_border_color="#27272a",
    input_border_color_focus="#3f3f46",        
    button_primary_background_fill="#2563eb",            
    button_primary_background_fill_hover="#3b82f6",      
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#27272a",
    button_secondary_background_fill_hover="#3f3f46",
    button_secondary_text_color="#e4e4e7",        
    block_radius="6px",                   
    button_large_radius="6px"            
)

with gr.Blocks(theme=dark_theme) as demo:    
    gr.Markdown("# Efficient Multilingual Review Analyser")    
    gr.Markdown("Architecture: Sorting based on language -> Translation to english and pooling by sentiment -> Embedding -> Dynamic Clustering -> Centroid Caveman Compression -> Qwen 72B API Inference -> Insights & Metrics")        
    
    with gr.Row():        
        file_input = gr.File(label="Upload CSV")        
        with gr.Column():            
            process_btn = gr.Button("Generate Analysis", variant="primary")            
            status = gr.Textbox(label="System Status")                
            
    with gr.Row():        
        plot_output = gr.Plot(label="Sentiment Distribution")        
        metrics_output = gr.Markdown()            
        
    insights_output = gr.Markdown()    
    
    process_btn.click(
        execute_pipeline, 
        inputs=file_input, 
        outputs=[status, plot_output, insights_output, metrics_output]
    )

if __name__ == "__main__":    
    demo.launch()