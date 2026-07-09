import gradio as gr
import re
import gc
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

# Pytorch will use fixed number of threads on CPU to avoid memory spikes
torch.set_num_threads(2)

# --- STOPWORDS DEFINITIONS ---
ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
    'can', 'will', 'just', 'don', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y'
}

HINGLISH_STOPWORDS = {
    'hai', 'hain', 'tha', 'thi', 'the', 'thhe', 'ho', 'hua', 'hui', 'hue', 'raha', 'rahi', 'rahe',
    'kar', 'karta', 'karte', 'karti', 'karo', 'kare', 'karna', 'liye', 'liya', 'diya', 'di', 'do',
    'gaya', 'gayi', 'gaye', 'jaa', 'ja', 'apna', 'apne', 'apni', 'mera', 'mere', 'meri', 'tera', 
    'tere', 'teri', 'iska', 'iske', 'iski', 'uska', 'uske', 'uski', 'in', 'un', 'hum', 'hamara', 
    'hamare', 'hamari', 'aap', 'aapka', 'aapke', 'aapki', 'yeh', 'ye', 'woh', 'voh', 'mai', 'main', 
    'me', 'tu', 'tum', 'tumhara', 'ki', 'ke', 'ka', 'ko', 'se', 'pe', 'par', 'mein', 'aur', 'ya', 
    'toh', 'to', 'lekin', 'agar', 'magar', 'jab', 'tab', 'tak', 'bhi', 'hi', 'kya', 'kyun', 'kyu', 
    'kaha', 'kahan', 'kaise', 'kab', 'kaun', 'konsa', 'bhai', 'yaar', 'bro', 'sir', 'madam', 'maam', 
    'plz', 'please', 'pls', 'ji', 'haan', 'ha', 'yes', 'wala', 'wale', 'wali', 'matlab', 'mtlb', 
    'kuch', 'koi', 'ab', 'aaj', 'kal', 'thik', 'theek', 'bohot', 'bahut', 'bhot', 'jyada', 'zyada', 
    'sirf', 'bas'
}

# --- MODEL LOADING ---
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# NLLB Translation Model for Hindi
translation_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="hin_Deva")
translation_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# Qwen LLM for Hinglish extraction & Final Summarization
qwen_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(qwen_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

slm_model = AutoModelForCausalLM.from_pretrained(
    qwen_id, 
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=True
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
    
    outputs = slm_model.generate(
        **inputs, 
        max_new_tokens=max_tokens, 
        temperature=0.1, 
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
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

def generate_cluster_chart(cluster_sizes):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"Topic {k+1}" for k in cluster_sizes.keys()]
    sizes = list(cluster_sizes.values())
    
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#c2c2f0', '#E5CCFF']
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors[:len(sizes)],
           wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    ax.axis('equal')  
    plt.tight_layout()
    return fig

# --- MAIN PIPELINE ---
@torch.inference_mode()
def execute_pipeline(file_path):
    if not file_path:
        return "No file provided.", None, "Error", "Upload a CSV first."
        
    try:
        df = pd.read_csv(file_path)
        if df.empty or df.shape[1] == 0:
            return "CSV is empty or unreadable.", None, "Error", "Error"
            
        raw_reviews = df.iloc[:, 0].dropna().astype(str).tolist()
        if len(raw_reviews) > 50:
            raw_reviews = raw_reviews[:50]
            
    except Exception as e:
        return f"Error reading CSV: {str(e)}", None, "Error", "Error"
        
    grouped_data = {'english': [], 'hindi': [], 'hinglish': []}
    baseline_tokens_cost = 0
    optimized_tokens_cost = 0
    words_dropped_locally = 0
    
    for idx, review in enumerate(raw_reviews):
        baseline_tokens_cost += len(tokenizer.tokenize(review)) + 60
        lang = router(review)
        grouped_data[lang].append((idx, review))
        
    processed_results = {}
    
    # 1. Process English (Regex/Stopwords)
    for idx, review in grouped_data['english']:
        clean_text = re.sub(r'[^\w\s]', '', review.lower())
        caveman_text = " ".join([w for w in clean_text.split() if w not in ENGLISH_STOPWORDS])
        processed_results[idx] = re.sub(r'[^\w\s]', '', caveman_text).strip()
        
    # 2. Process Hindi (NLLB Translation + Stopword alignment)
    if grouped_data['hindi']:            
        hindi_indices = [item[0] for item in grouped_data['hindi']]
        raw_hindi_texts = []
        
        for idx, review in grouped_data['hindi']:
            raw_hindi_texts.append(review)
            
        inputs = translation_tokenizer(raw_hindi_texts, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = translation_model.generate(
            **inputs, 
            forced_bos_token_id=translation_tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=60
        )
        translations = translation_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        for i, translated_text in enumerate(translations):
            clean_text = re.sub(r'[^\w\s]', '', translated_text.lower())
            caveman_text = " ".join([w for w in clean_text.split() if w not in ENGLISH_STOPWORDS])
            
            # Track the filler words destroyed from the translated English sentence
            words_dropped_locally += (len(clean_text.split()) - len(caveman_text.split()))
            processed_results[hindi_indices[i]] = re.sub(r'[^\w\s]', '', caveman_text).strip()
            
    # 3. Process Hinglish (Qwen Batched Extraction)
    if grouped_data['hinglish']:
        hinglish_indices = [item[0] for item in grouped_data['hinglish']]
        clean_hinglish_texts = []
        # Optimized prompt to prevent sentences
        sys_prompt = "You are a keyword extractor. Read this text and output exactly 2-3 short English keywords describing the core topic (e.g., 'fast delivery', 'bad battery'). Output ONLY the keywords. Do not write sentences."
        
        for idx, review in grouped_data['hinglish']:
            original_word_count = len(review.split())
            clean_hinglish = " ".join([w for w in review.lower().split() if w not in HINGLISH_STOPWORDS])
            words_dropped_locally += (original_word_count - len(clean_hinglish.split()))
            clean_hinglish_texts.append(clean_hinglish if clean_hinglish else "empty")
            optimized_tokens_cost += len(tokenizer.tokenize(sys_prompt + clean_hinglish)) + 12
            
        llm_responses = batch_ask_qwen_optimized(sys_prompt, clean_hinglish_texts, max_tokens=15)
        for i, response in enumerate(llm_responses):
            processed_results[hinglish_indices[i]] = re.sub(r'[^\w\s]', '', response.lower()).strip()
            
    # 4. Reconstruct Order & Extract Embeddings
    processed_texts = [processed_results[i] for i in range(len(raw_reviews))]
    embeddings = embedder.encode(processed_texts, batch_size=32)
    
    # 5. AUTOMATIC K-MEANS CLUSTERING (Silhouette Score)
    num_samples = len(raw_reviews)
    if num_samples < 3:
        best_k = 1
        labels = [0] * num_samples
    else:
        best_k = 2
        best_score = -1
        max_k = min(6, num_samples - 1)
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbls = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, lbls)
            if score > best_score:
                best_score = score
                best_k = k
                
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

    clusters = {i: {"raw": [], "caveman": []} for i in range(best_k)}
    for raw, caveman, label in zip(raw_reviews, processed_texts, labels):
        clusters[label]["raw"].append(raw)
        clusters[label]["caveman"].append(caveman)
        
    cluster_sizes = {i: len(clusters[i]["raw"]) for i in clusters.keys()}
    pie_chart = generate_cluster_chart(cluster_sizes)
        
    # 6. Batched Cluster Insight Generation (Sentiment Aware)
    # Optimized prompt to explicitly identify mixed sentiments
    insight_sys_prompt = "Analyze these grouped customer keywords. Write a single concise sentence summarizing the main features mentioned, explicitly noting if the feedback is positive, negative, or mixed."
    cluster_payloads = []
    
    for cid, data in clusters.items():
        sample_slice = random.sample(data["caveman"], min(6, len(data["caveman"])))
        keyword_payload = ", ".join(sample_slice)
        cluster_payloads.append(keyword_payload)
        optimized_tokens_cost += len(tokenizer.tokenize(insight_sys_prompt + keyword_payload)) + 50
        
    summary_insights = batch_ask_qwen_optimized(insight_sys_prompt, cluster_payloads, max_tokens=40)
    
    insight_markdown = f"*(Automatically detected **{best_k} distinct topics** based on data similarity)*\n\n"
    for cid, data in clusters.items():
        summary_insight = summary_insights[cid]
        insight_markdown += f"### Topic {cid + 1} ({len(data['raw'])} items)\n"
        insight_markdown += f"**Summary:** {summary_insight}\n\n"
        
        sample_slice = random.sample(data["caveman"], min(5, len(data["caveman"])))
        insight_markdown += f"**Core Analyzed Phrases:** `{', '.join(sample_slice)}`\n"
        insight_markdown += "\n---\n\n"

    token_savings_percentage = max(0, 100 - ((optimized_tokens_cost / baseline_tokens_cost) * 100)) if baseline_tokens_cost > 0 else 0
    
    viability_markdown = f"""
    ### Scalability & Cost Savings Analysis
    This architecture intercepts unstructured data *before* it hits expensive LLM APIs.
    
    * **Direct API Processing Cost:** ~{baseline_tokens_cost} tokens required.
    * **Our Local Pipeline Cost:** ~{optimized_tokens_cost} tokens required.
    * **Tokens Avoided Locally:** {words_dropped_locally} useless slang/grammar words eradicated purely on CPU logic.
    
    #### **Net LLM Token Savings: {token_savings_percentage:.1f}%**
    """

    gc.collect()
    return f"Processed {len(raw_reviews)} rows into {best_k} clusters automatically.", pie_chart, insight_markdown, viability_markdown

# --- GRADIO UI ---
with gr.Blocks(title="Review Classifier") as demo:
    gr.Markdown("# Multilingual Feedback Intelligence Tool")
    gr.Markdown("Upload a `.csv` file. The tool will automatically detect the language, route it through an embedded translation/AI pipeline, **determine the ideal number of feedback topics**, and output operational insights. *(Max 50 rows due to cloud limits)*")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload CSV File (First column will be processed)", file_types=['.csv'])
            process_btn = gr.Button("Execute Analysis", variant="primary")
            status_text = gr.Textbox(label="System Status", interactive=False)
            
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Topic Distribution")
            visual_output = gr.Plot(label="Feedback Categorization")
            
        with gr.Column(scale=1):
            gr.Markdown("## Automated Analysis")
            insights_display = gr.Markdown()
            
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            savings_display = gr.Markdown(label="System Metrics")

    process_btn.click(
        fn=execute_pipeline, 
        inputs=[file_input], 
        outputs=[status_text, visual_output, insights_display, savings_display],
        concurrency_limit=1
    )

if __name__ == "__main__":
    demo.launch( theme=gr.themes.Soft())