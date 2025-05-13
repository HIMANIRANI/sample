import os
import json
import torch
import pickle
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

# 1. Detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {device.upper()}")

# 2. Load model manually onto GPU
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)

# 3. Define paths
company_folder = 'updated_company_data'
save_path = 'company_vec'
texts_cache = 'texts.pkl'
embeddings_cache = 'embeddings.pkl'

# 4. Prepare documents
documents = []
indicator_requirements = {
    "SMA": 10,
    "EMA": 12,
    "RSI": 14,
    "MACD": 26,
    "Bollinger Bands": 20
}

for filename in os.listdir(company_folder):
    if filename.endswith('.json'):
        symbol = filename.replace('.json', '')
        file_path = os.path.join(company_folder, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            company_data = json.load(f)

        for date, data in company_data.items():
            price = data.get('price', {})
            indicators = data.get('indicators', {})

            lines = [
                f"Company Symbol: {symbol}",
                f"Date: {date}",
                f"Open Price: {price.get('open', 'N/A')}",
                f"Close Price: {price.get('close', 'N/A')}",
                f"Max Price: {price.get('max', 'N/A')}",
                f"Min Price: {price.get('min', 'N/A')}",
                f"Traded Shares: {data.get('tradedShares', 'N/A')}",
                f"Amount: {data.get('amount', 'N/A')}",
            ]

            for indicator_name, required_days in indicator_requirements.items():
                if indicator_name == "Bollinger Bands":
                    bb_high = indicators.get('BB_High')
                    bb_low = indicators.get('BB_Low')
                    bb_mid = indicators.get('BB_Mid')

                    if bb_high is not None and bb_low is not None and bb_mid is not None:
                        lines.append(f"Bollinger Bands High: {bb_high}")
                        lines.append(f"Bollinger Bands Low: {bb_low}")
                        lines.append(f"Bollinger Bands Mid: {bb_mid}")
                    else:
                        lines.append(f"Bollinger Bands not available: Requires minimum {required_days} days of data.")
                else:
                    value = indicators.get(indicator_name)
                    if value is not None:
                        lines.append(f"{indicator_name}: {value}")
                    else:
                        lines.append(f"{indicator_name} not available: Requires minimum {required_days} days of data.")

            page_content = "\n".join(lines).strip()
            documents.append(Document(page_content=page_content))

print(f"✅ Total documents prepared: {len(documents)}")

# 5. Load existing embeddings if available
texts = []
embeddings = []

if os.path.exists(texts_cache) and os.path.exists(embeddings_cache):
    print("📂 Found previous cache! Resuming...")
    with open(texts_cache, "rb") as f:
        texts = pickle.load(f)
    with open(embeddings_cache, "rb") as f:
        embeddings = pickle.load(f)
else:
    print("🆕 No cache found. Starting fresh...")

# 6. Find where to resume
start_index = len(texts)
print(f"➡️ Resuming embedding from document {start_index + 1} onwards...")

# 7. Embed documents in Batches
batch_size = 32  # You can increase to 64 if GPU VRAM allows (for GTX 1650, 32 is safe)

texts_to_embed = [doc.page_content for doc in documents[start_index:]]

for batch_start in tqdm(range(0, len(texts_to_embed), batch_size), desc="Embedding in batches"):
    batch_texts = texts_to_embed[batch_start:batch_start + batch_size]
    
    try:
        batch_embeddings = model.encode(batch_texts, batch_size=batch_size, device=device, show_progress_bar=False)

        texts.extend(batch_texts)
        embeddings.extend(batch_embeddings)

        # Save cache after every batch
        with open(texts_cache, "wb") as f:
            pickle.dump(texts, f)
        with open(embeddings_cache, "wb") as f:
            pickle.dump(embeddings, f)

        print(f"💾 Saved after {len(texts)} total documents.")
    
    except Exception as e:
        print(f"❌ Error in batch starting at document {batch_start+start_index+1}: {e}")

print("✅ All embeddings completed and cached.")

# 8. Build FAISS
print("📦 Building FAISS vectorstore...")
company_vectorstore = FAISS.from_embeddings(
    list(zip(texts, embeddings)),
    embedding=model
)

# 9. Save FAISS index
if not os.path.exists(save_path):
    os.makedirs(save_path)

company_vectorstore.save_local(save_path)
print(f"✅ Company FAISS vectorstore saved at {save_path}")
