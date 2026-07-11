import gradio as gr
import os
import re
import gc
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import hdbscan
from huggingface_hub import hf_hub_download, InferenceClient
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

torch.set_num_threads(2)

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)

ENGLISH_STOPWORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 'he', 'him', 'she', 'it', 'they', 'them', 'what', 'which', 'who', 'this', 'that', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'a', 'an', 'the', 'if', 'or', 'as', 'of', 'at', 'by', 'for', 'with', 'about', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'then', 'here', 'there', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'}
SOFT_HINGLISH_STOPWORDS = {'hai', 'hain', 'tha', 'thi', 'the', 'ho', 'kar', 'karta', 'liye', 'diya', 'gaya', 'aap', 'yeh', 'ye', 'woh', 'mai', 'main', 'me', 'ki', 'ke', 'ka', 'ko', 'se', 'aur', 'ya', 'toh', 'to', 'bhi', 'hi', 'kya', 'bas'}

# Hindi to English (Facebook's nllb-200-distilled-600M)
mdl = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M",torch_dtype=torch.float32,low_cpu_mem_usage=True,)

tknzr = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",src_lang="hin_Deva",tgt_lang="eng_Latn")
translator = pipeline(task="translation",model=mdl,tokenizer=tknzr)
# Hinglish to English (Qwen 3B GGUF)
qwen_path = hf_hub_download(repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF", filename="qwen2.5-3b-instruct-q4_k_m.gguf")
llm_hinglish = Llama(model_path=qwen_path, n_ctx=1024, n_threads=2) # n_threads=2 matches HF Free Tier

# Sentiment & Embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", device=-1)



# Helper functions

def route_language(text):
    """Routes text to the appropriate translation model based on script and slang."""
    if re.search(r'[\u0900-\u097F]', text): return 'hindi'
    if any(w in text.lower().split() for w in set(SOFT_HINGLISH_STOPWORDS)): return 'hinglish'
    return 'english'

def get_caveman(text):
    """Strips grammar and stopwords to compress tokens for the API."""
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    clean = [w for w in words if w not in ENGLISH_STOPWORDS and w not in SOFT_HINGLISH_STOPWORDS]
    return " ".join(clean) if clean else "empty"

def dynamic_cluster(embeddings):
    """Calculates UMAP/HDBSCAN parameters dynamically based strictly on the POOL size."""
    num_samples = len(embeddings)
    if num_samples < 5: 
        return [0] * num_samples # Too small to cluster, group as a single cluster
    
    n_components = max(3, min(15, int(math.log10(num_samples) * 2)))
    n_neighbors = min(15, max(2, num_samples - 1)) # max(2) prevents UMAP crash on tiny pools
    min_cluster_size = max(3, min(500, int(num_samples * 0.02)))
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine', random_state=42)
    reduced = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom')
    return clusterer.fit_predict(reduced)

def get_centric_caveman_samples(cluster_indices, embeddings, full_texts, n=5):
    """Finds the mathematical center of a cluster, picks the top N reviews, and Caveman-compresses them."""
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
    full_english_texts = []
    
    # Translation (Keep Full Sentences)
    for text in raw_reviews:
        baseline_tokens += len(text.split()) + 50 # Estimate if we sent full text to API
        lang = route_language(text)
        
        if lang == 'hindi':
            inputs = tknzr([text], return_tensors="pt")
            out = mdl.generate(**inputs)
            full_english_texts.append(tknzr.batch_decode(out, skip_special_tokens=True)[0])
        elif lang == 'hinglish':
            prompt = f"<|im_start|>system\nTranslate Hinglish to English.<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            out = llm_hinglish(prompt, max_tokens=100, stop=["<|im_end|>"])
            full_english_texts.append(out['choices'][0]['text'].strip())
        else:
            full_english_texts.append(text)

    # Sentiment Pooling (Stratification)
    sent_results = sentiment_analyzer(full_english_texts)
    buckets = {"Positive": [], "Neutral": [], "Negative": []}
    for i, res in enumerate(sent_results):
        label = res['label']
        cat = "Negative" if "0" in label else ("Positive" if "2" in label else "Neutral")
        buckets[cat].append(i)

    insight_markdown = ""
    
    # Process Each Pool Independently
    for sentiment, global_indices in buckets.items():
        if not global_indices: continue
        
        insight_markdown += f"## {sentiment} Feedback Analysis\n"
        
        # Isolate texts and embeddings for THIS pool only
        pool_texts = [full_english_texts[i] for i in global_indices]
        pool_embs = embedder.encode(pool_texts)
        
        # Dynamic Clustering based on POOL SIZE
        pool_labels = dynamic_cluster(pool_embs)
        
        # Analyze Clusters
        for c_id in set(pool_labels):
            if c_id == -1: continue # Skip HDBSCAN noise
            
            # Find indices of items in this specific cluster
            local_c_indices = [j for j, lab in enumerate(pool_labels) if lab == c_id]
            
            # Extract Centroid Caveman samples
            compressed_input = get_centric_caveman_samples(
                np.array(local_c_indices), 
                pool_embs, 
                pool_texts, 
                n=5
            )
            
            # Qwen 72B API Inference
            prompt = (
                f"Analyze these representative keyword-sets from a {sentiment} cluster: [{compressed_input}]. "
                "Infer the specific business issue or strength. Output exactly:\n"
                "1. Insight: [1 sentence]\n2. Action: [1 sentence]"
            )
            
            optimized_tokens += len(prompt.split())
            
            try:
                res = client.chat_completion(
                    model="Qwen/Qwen2.5-72B-Instruct", 
                    messages=[{"role": "user", "content": prompt}], 
                    max_tokens=100
                )
                summary = res.choices[0].message.content
            except Exception:
                summary = f"Centric Keywords: {compressed_input}"

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
    
    gc.collect()
    return "Analysis Complete", fig, insight_markdown, viability_report

# UI THEME & LAUNCH
dark_theme = gr.themes.Base(
    primary_hue="blue",         
    secondary_hue="zinc",       
    neutral_hue="zinc",         
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    # Body & Backgrounds (Pure Slate/Zinc Dark Mode)
    body_background_fill="#09090b",        
    block_background_fill="#18181b",      
    block_border_width="1px",
    block_border_color="#27272a",         
    
    # Typography
    body_text_color="#f4f4f5",            
    body_text_color_subdued="#a1a1aa",     
    heading_large_text_color="#ffffff",
    
    # Inputs & File Upload Area
    input_background_fill="#09090b",
    input_border_color="#27272a",
    input_border_color_focus="#3f3f46",
    
    # Buttons (Tailored Steel Blue Accent)
    button_primary_background_fill="#2563eb",        
    button_primary_background_fill_hover="#3b82f6",  
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#27272a",
    button_secondary_background_fill_hover="#3f3f46",
    button_secondary_text_color="#e4e4e7",
    
    # Structure & Edges
    block_radius="6px",               
    button_large_radius="6px",
    container_blurs=False,             
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

    process_btn.click(execute_pipeline, inputs=file_input, outputs=[status, plot_output, insights_output, metrics_output])

if __name__ == "__main__":
    demo.launch(theme=dark_theme)