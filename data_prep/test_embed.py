import gzip
import pickle
import numpy as np

# Define your compressed file paths
texts_file = 'texts.pkl.gz'
embeddings_file = 'embeddings.pkl.gz'

# Load saved texts
print("🔄 Loading texts...")
texts = []
with gzip.open(texts_file, 'rb') as f:
    while True:
        try:
            item = pickle.load(f)
            texts.append(item)
        except EOFError:
            print("✅ Loaded texts completely.")
            break
        except Exception as e:
            print(f"❌ Error loading texts: {e}")
            break

# Load saved embeddings
print("🔄 Loading embeddings...")
embeddings = []
with gzip.open(embeddings_file, 'rb') as f:
    while True:
        try:
            item = pickle.load(f)
            embeddings.append(item)
        except EOFError:
            print("✅ Loaded embeddings completely.")
            break
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            break

# Convert embeddings to numpy array
embeddings = np.array(embeddings)

# Basic checks
print("\n🛠 Verifying...")

if len(texts) != len(embeddings):
    print(f"❌ Number mismatch: {len(texts)} texts vs {len(embeddings)} embeddings")
else:
    print(f"✅ Number of texts and embeddings match: {len(texts)} items.")

print(f"✅ Embeddings shape: {embeddings.shape}")

# Show a sample
print("\n🔎 Sample Text:")
print(texts[0])
print("\n🔎 Sample Embedding (first 5 values):")
print(embeddings[0][:5])
