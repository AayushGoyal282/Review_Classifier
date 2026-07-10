import gradio as gr
import re
import gc
import math
import torch
import pandas as pd
import matplotlib.pyplot as plt
import umap
import hdbscan
from collections import Counter
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

torch.set_num_threads(2)

# --- STOPWORDS DEFINITIONS ---
ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
    'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'what', 'which', 'who', 
    'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'do', 'does', 'did', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 
    'will', 'just', 'don', 'should', 'now', 'not', 'never', 'no', 'didn', "didn't", 'empty'
}

SOFT_HINGLISH_STOPWORDS = {
    'hai', 'hain', 'tha', 'thi', 'the', 'thhe', 'ho', 'hua', 'hui', 'hue', 'raha', 'rahi', 'rahe',
    'kar', 'karta', 'karte', 'karti', 'karo', 'kare', 'karna', 'liye', 'liya', 'diya', 'di', 'do',
    'gaya', 'gayi', 'gaye', 'jaa', 'ja', 'apna', 'apne', 'apni', 'mera', 'mere', 'meri', 'tera',
    'tere', 'teri', 'iska', 'iske', 'iski', 'uska', 'uske', 'uski', 'in', 'un', 'hum', 'hamara',
    'hamare', 'hamari', 'aap', 'aapka', 'aapke', 'aapki', 'yeh', 'ye', 'woh', 'voh', 'mai', 'main',
    'me', 'tu', 'tum', 'tumhara', 'ki', 'ke', 'ka', 'ko', 'se', 'pe', 'par', 'mein', 'aur', 'ya',
    'toh', 'to', 'lekin', 'agar', 'magar', 'jab', 'tab', 'tak', 'bhi', 'hi', 'kya', 'kyun', 'kyu',
    'kaha', 'kahan', 'kaise', 'kab', 'kaun', 'konsa', 'bhai', 'yaar', 'bro', 'sir', 'madam', 'maam',
    'plz', 'please', 'pls', 'ji', 'haan', 'ha', 'yes', 'wala', 'wale', 'wali', 'matlab', 'mtlb',
    'kuch', 'koi', 'ab', 'aaj', 'kal', 'thik', 'theek', 'sirf', 'bas'
}

# --- MODEL LOADING ---
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 1. Transformer Sentiment Model (Restored to original RoBERTa)
sentiment_analyzer = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment-latest", 
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True, 
    max_length=128
)

# 2. NLLB
translation_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="hin_Deva")
translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-600M", 
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

# 3. Qwen
qwen_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(qwen_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

slm_model = AutoModelForCausalLM.from_pretrained(
    qwen_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
)

@torch.inference_mode()
def batch_ask_qwen_optimized(system_prompt: str, user_texts: list, max_tokens: int) -> list:
    if not user_texts:
        return []

    formatted_prompts = []
    for text in user_texts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        formatted_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        
    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True)
    outputs = slm_model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.1, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return [resp.strip() for resp in responses]

def router(text: str) -> str:
    text_lower = text.lower()
    if re.search(r'[\u0900-\u097F]', text):
        return 'hindi'
    hinglish_markers = {'kya', 'hai', 'bhai', 'matlab', 'thik', 'kyun', 'nahi', 'aur', 'bohot', 'toh'}
    words = set(text_lower.split())
    if words.intersection(hinglish_markers):
        return 'hinglish'
    return 'english'

def get_top_keywords(caveman_list, top_k=12):
    all_words = " ".join(caveman_list).split()
    most_common = Counter(all_words).most_common(top_k)
    return ", ".join([word for word, count in most_common])

def dynamic_cluster(embeddings):
    num_samples = len(embeddings)
    if num_samples < 5:
        return [0] * num_samples
    n_components = max(3, min(15, int(math.log10(num_samples) * 2)))
    n_neighbors = min(15, max(3, num_samples - 1))
    min_cluster_size = max(3, min(500, int(num_samples * 0.02)))
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine', random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom')
    return clusterer.fit_predict(reduced_embeddings)

def generate_tri_pie_chart(pos_sizes, neu_sizes, neg_sizes):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#c2c2f0', '#E5CCFF', '#D3D3D3']

    def plot_pie(ax, sizes, title):
        if sizes:
            ax.pie(sizes.values(), labels=sizes.keys(), autopct='%1.1f%%', startangle=90, colors=colors)
            ax.set_title(title)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_title(title)
            ax.axis('off')

    plot_pie(ax1, pos_sizes, "⭐ Strengths (Positive)")
    plot_pie(ax2, neu_sizes, "📊 Operations (Neutral)")
    plot_pie(ax3, neg_sizes, "🚨 Issues (Negative)")
    plt.tight_layout()
    return fig

# --- MAIN PIPELINE ---
@torch.inference_mode()
def execute_pipeline(file_path):
    if not file_path:
        return "No file provided.", None, "Error", "Upload a CSV first."

    try:
        df = pd.read_csv(file_path, header=None) 
        if df.empty or df.shape[1] == 0:
            return "CSV is empty or unreadable.", None, "Error", "Error"
        raw_reviews = df.iloc[:, 0].dropna().astype(str).tolist()
        if len(raw_reviews) > 120:
            raw_reviews = raw_reviews[:120] 
    except Exception as e:
        return f"Error reading CSV: {str(e)}", None, "Error", "Error"
        
    grouped_data = {'english': [], 'hindi': [], 'hinglish': []}
    baseline_tokens_cost = 0
    optimized_tokens_cost = 0

    for idx, review in enumerate(raw_reviews):
        baseline_tokens_cost += len(tokenizer.tokenize(review)) + 60
        lang = router(review)
        grouped_data[lang].append((idx, review))
        
    full_english_texts = {i: "" for i in range(len(raw_reviews))}

    for idx, review in grouped_data['english']:
        full_english_texts[idx] = review.lower().strip()
        
    if grouped_data['hindi']:            
        hindi_indices = [item[0] for item in grouped_data['hindi']]
        raw_hindi_texts = [item[1] for item in grouped_data['hindi']]
        inputs = translation_tokenizer(raw_hindi_texts, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = translation_model.generate(**inputs, forced_bos_token_id=translation_tokenizer.convert_tokens_to_ids("eng_Latn"), max_length=60, num_beams=2, early_stopping=True)
        translations = translation_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        for i, translated_text in enumerate(translations):
            full_english_texts[hindi_indices[i]] = translated_text.strip().lower()
            
    # 3. Process Hinglish (Strictly Caveman Pooling)
    if grouped_data['hinglish']:
        hinglish_indices = [item[0] for item in grouped_data['hinglish']]
        caveman_texts = []
        
        # Strip Hinglish stopwords BEFORE hitting LLM
        for idx, review in grouped_data['hinglish']:
            clean_hinglish = re.sub(r'[^\w\s]', '', review.lower())
            words = [w for w in clean_hinglish.split() if w not in SOFT_HINGLISH_STOPWORDS]
            caveman_texts.append(" ".join(words) if words else "empty")
            
        # Robust Caveman Prompt (No hardcoding, forces slang inference)
        sys_prompt = (
            "You are an expert multilingual AI. Translate this numbered list of broken Hinglish (Hindi-English) keywords into English keywords. "
            "These are raw keywords without grammar. Use your knowledge of Indian internet slang to infer meaning across any domain. "
            "Keep the exact same numbering. Output ONLY the English translations.\n"
            "Example Input:\n1. khana mehenga bassi\n2. kida baal bc\n3. app crash ganda\n4. delivery guy badtameezi\n5. product quality kharab\n6. service slow\n7. refund process lamba\n8. app UI confusing\n9. payment gateway fail\n10. customer support rude\n"
            "Example Output:\n1. food expensive stale\n2. bug hair bad\n3. app crash bad\n4. delivery guy rude\n5. product quality poor\n6. service slow\n7. refund process long\n8. app UI confusing\n9. payment gateway fail\n10. customer support rude"
        )
        
        CHUNK_SIZE = 20
        
        for i in range(0, len(caveman_texts), CHUNK_SIZE):
            chunk_indices = hinglish_indices[i : i + CHUNK_SIZE]
            chunk_texts = caveman_texts[i : i + CHUNK_SIZE]
            
            pooled_input = "\n".join([f"{j+1}. {text}" for j, text in enumerate(chunk_texts)])
            optimized_tokens_cost += len(tokenizer.tokenize(sys_prompt + pooled_input))
            
            llm_response = batch_ask_qwen_optimized(sys_prompt, [pooled_input], max_tokens=CHUNK_SIZE * 15)[0]
            parsed_lines = re.findall(r'\d+\.\s*(.*)', llm_response)
            
            for j, original_idx in enumerate(chunk_indices):
                if j < len(parsed_lines):
                    full_english_texts[original_idx] = parsed_lines[j].strip().lower()
                else:
                    full_english_texts[original_idx] = chunk_texts[j].lower()
            
    # 4. Sentiment Split (No Triggers, pure model inference)
    positives = {"raw": [], "caveman": [], "emb": []}
    neutrals = {"raw": [], "caveman": [], "emb": []}
    negatives = {"raw": [], "caveman": [], "emb": []}

    texts_to_analyze = [full_english_texts[i] for i in range(len(raw_reviews))]
    sentiment_results = sentiment_analyzer(texts_to_analyze, batch_size=8)

    for i, (full_text, sentiment) in enumerate(zip(texts_to_analyze, sentiment_results)):
        label = sentiment['label'].lower() 
        
        no_punct = re.sub(r'[^\w\s]', '', full_text.lower())
        caveman_words = [w for w in no_punct.split() if w not in ENGLISH_STOPWORDS]
        caveman_text = " ".join(caveman_words)
        
        if label == 'negative':
            negatives["raw"].append(raw_reviews[i])
            negatives["caveman"].append(caveman_text)
        elif label == 'positive':
            positives["raw"].append(raw_reviews[i])
            positives["caveman"].append(caveman_text)
        else:
            neutrals["raw"].append(raw_reviews[i])
            neutrals["caveman"].append(caveman_text)

    # 5. Embeddings & Clustering
    def process_bucket(bucket):
        if bucket["caveman"]:
            bucket["emb"] = embedder.encode(bucket["caveman"], batch_size=32)
            return dynamic_cluster(bucket["emb"])
        return []

    neg_labels = process_bucket(negatives)
    neu_labels = process_bucket(neutrals)
    pos_labels = process_bucket(positives)

    def group_clusters(labels, data_dict):
        clusters = {}
        for raw, caveman, label in zip(data_dict["raw"], data_dict["caveman"], labels):
            if label not in clusters:
                clusters[label] = {"raw": [], "caveman": []}
            clusters[label]["raw"].append(raw)
            clusters[label]["caveman"].append(caveman)
        return clusters

    neg_clusters = group_clusters(neg_labels, negatives)
    neu_clusters = group_clusters(neu_labels, neutrals)
    pos_clusters = group_clusters(pos_labels, positives)

    neg_sizes = {(f"Issue {k+1}" if k != -1 else "Misc"): len(v["raw"]) for k, v in neg_clusters.items()}
    neu_sizes = {(f"Topic {k+1}" if k != -1 else "Misc"): len(v["raw"]) for k, v in neu_clusters.items()}
    pos_sizes = {(f"Strength {k+1}" if k != -1 else "Misc"): len(v["raw"]) for k, v in pos_clusters.items()}
    pie_chart = generate_tri_pie_chart(pos_sizes, neu_sizes, neg_sizes)
        
    insight_markdown = ""

    def summarize_cluster(sys_prompt, clusters, title):
        markdown = f"### {title}\n\n"
        for cid, data in clusters.items():
            if cid == -1: continue 
            top_keywords = get_top_keywords(data["caveman"], top_k=12)
            nonlocal optimized_tokens_cost 
            optimized_tokens_cost += len(tokenizer.tokenize(sys_prompt + top_keywords)) + 40
            summary = batch_ask_qwen_optimized(sys_prompt, [top_keywords], max_tokens=80)[0]
            markdown += f"**Cluster {cid + 1} ({len(data['raw'])} reviews)**\n{summary}\n*(Keywords: {top_keywords})*\n\n---\n\n"
        return markdown

    # Domain-Agnostic Prompts
    if neg_clusters:
        neg_prompt = (
            "You are a strict business consultant. Analyze these extracted customer complaint keywords to determine the industry/context. "
            "Do NOT comment on the language. Output exactly two lines:\n"
            "1. Issue: [1-sentence summary of the core problem]\n"
            "2. Action: [1 short, actionable recommendation to fix it]"
        )
        insight_markdown += summarize_cluster(neg_prompt, neg_clusters, "🚨 Critical Issues (Requires Attention)")

    if neu_clusters:
        neu_prompt = (
            "You are a business operations analyst. Analyze these mixed customer keywords to determine the context. "
            "Do NOT comment on the language or text format. Output exactly two lines:\n"
            "1. Observation: [1-sentence summary of customer behavior or neutral feedback]\n"
            "2. Suggestion: [1 short operational tweak to improve the experience]"
        )
        insight_markdown += summarize_cluster(neu_prompt, neu_clusters, "📊 Operational Observations (Factual / Mixed)")

    if pos_clusters:
        pos_prompt = (
            "You are a marketing analyst. Analyze these positive customer keywords to determine the context. "
            "Do NOT comment on the language. Output exactly two lines:\n"
            "1. Strength: [1-sentence summary of what customers love]\n"
            "2. Highlight: [1-sentence suggestion on how to use this in marketing/advertising]"
        )
        insight_markdown += summarize_cluster(pos_prompt, pos_clusters, "⭐ Core Strengths (What to keep doing)")

    token_savings_percentage = max(0, 100 - ((optimized_tokens_cost / baseline_tokens_cost) * 100)) if baseline_tokens_cost > 0 else 0

    viability_markdown = f"""
    ### Scalability & Cost Savings Analysis
    * **Direct API Processing Cost:** ~{baseline_tokens_cost} tokens required.
    * **Our Local Pipeline Cost:** ~{optimized_tokens_cost} tokens required.
    #### **Net LLM Token Savings: {token_savings_percentage:.1f}%**
    """

    gc.collect()
    return f"Processed {len(raw_reviews)} rows successfully.", pie_chart, insight_markdown, viability_markdown

# --- GRADIO UI ---
with gr.Blocks(title="Review Analyzer") as demo: 
    gr.Markdown("# Multilingual(hindi/hinglish/english) Review Analyzer") 
    gr.Markdown("Upload a `.csv` file. The tool utilizes Caveman Input Pooling, RoBERTa Sentiment Analysis, and dynamic clustering to generate Domain-Agnostic Action Items.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload CSV File", file_types=['.csv'])
            process_btn = gr.Button("Generate Analysis", variant="primary")
            status_text = gr.Textbox(label="System Status", interactive=False)
            
    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Sentiment & Topic Distribution")
            visual_output = gr.Plot(label="Review Categorization")
            
        with gr.Column(scale=1):
            gr.Markdown("## Analysis Insights:")
            insights_display = gr.Markdown()
            
    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            savings_display = gr.Markdown(label="Metrics:")

    process_btn.click(
        fn=execute_pipeline, 
        inputs=[file_input], 
        outputs=[status_text, visual_output, insights_display, savings_display],
        concurrency_limit=1
    )

if __name__ == "__main__": 
    demo.launch(theme=gr.themes.Soft())