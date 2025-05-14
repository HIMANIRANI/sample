import os
import json
import torch
import pickle
import gzip
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
texts_cache = 'texts.pkl.gz'
embeddings_cache = 'embeddings.pkl.gz'

# 4. Safe pickle load and save (gzip)
def safe_load_pickle(filename):
    recovered = []
    if not os.path.exists(filename):
        return recovered
    
    try:
        with gzip.open(filename, "rb") as f:
            while True:
                try:
                    item = pickle.load(f)
                    recovered.append(item)
                except EOFError:
                    print(f"✅ Reached end of partial pickle file: {filename}")
                    break
                except Exception as e:
                    print(f"❌ Error reading {filename}: {e}")
                    break
    except Exception as e:
        print(f"❌ Failed to open {filename}: {e}")
    
    return recovered

def safe_save_pickle(obj, filename):
    tmp_filename = filename + '.tmp'
    with gzip.open(tmp_filename, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_filename, filename)  # Atomic replace

# 5. Prepare documents
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

# 6. Load existing embeddings if available
texts = safe_load_pickle(texts_cache)
embeddings = safe_load_pickle(embeddings_cache)

if texts and embeddings:
    print(f"📂 Found previous cache! Resuming from document {len(texts) + 1}")
else:
    print("🆕 No previous cache found. Starting fresh...")

# 7. Find where to resume
start_index = len(texts)

# 8. Embed documents in Batches
batch_size = 32  # Still keeping small batch for GPU memory safety
save_every = 100  # Save after every 100 embeddings
new_embeddings_since_last_save = 0

texts_to_embed = [doc.page_content for doc in documents[start_index:]]

for batch_start in tqdm(range(0, len(texts_to_embed), batch_size), desc="Embedding in batches"):
    batch_texts = texts_to_embed[batch_start:batch_start + batch_size]

    try:
        batch_embeddings = model.encode(batch_texts, batch_size=batch_size, device=device, show_progress_bar=False)

        texts.extend(batch_texts)
        embeddings.extend(batch_embeddings)

        new_embeddings_since_last_save += len(batch_texts)

        if new_embeddings_since_last_save >= save_every:
            safe_save_pickle(texts, texts_cache)
            safe_save_pickle(embeddings, embeddings_cache)
            print(f"💾 Saved after {len(texts)} total documents.")
            new_embeddings_since_last_save = 0

    except Exception as e:
        print(f"❌ Error in batch starting at document {batch_start+start_index+1}: {e}")
        break

# Final save if anything left
if new_embeddings_since_last_save > 0:
    safe_save_pickle(texts, texts_cache)
    safe_save_pickle(embeddings, embeddings_cache)
    print(f"💾 Final save after {len(texts)} total documents.")

print("✅ All embeddings completed and cached.")

# 9. Build FAISS
print("📦 Building FAISS vectorstore...")
company_vectorstore = FAISS.from_embeddings(
    list(zip(texts, embeddings)),
    embedding=model
)

# 10. Save FAISS index
if not os.path.exists(save_path):
    os.makedirs(save_path)

company_vectorstore.save_local(save_path)
print(f"✅ Company FAISS vectorstore saved at {save_path}")
