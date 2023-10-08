from datasets import load_dataset
import random
import json

def get_k_shingles(text, k=2):
    """Return a list of k-shingles for the given text."""
    return [text[i:i+k] for i in range(len(text) - k + 1)]

# --- MinHash Class Definition ---
class MinHash:
    def __init__(self, num_hashes):
        self.num_hashes = num_hashes
        self.max_hash = 2**32 - 1
        self.hash_functions = self._generate_hash_functions()

    def _generate_hash_functions(self):
        def _hash(x, a, b):
            return (a * x + b) % self.max_hash

        hash_functions = []
        for _ in range(self.num_hashes):
            a = random.randint(1, self.max_hash - 1)
            b = random.randint(0, self.max_hash - 1)
            hash_functions.append(lambda x, a=a, b=b: _hash(hash(x), a, b))
        return hash_functions

    def compute(self, set_data):
        min_hashes = [self.max_hash] * self.num_hashes
        for item in set_data:
            for i, hash_function in enumerate(self.hash_functions):
                hash_val = hash_function(item)
                min_hashes[i] = min(hash_val, min_hashes[i])
        return tuple(min_hashes)

# --- LSH Class Definition ---
class LSH:
    def __init__(self, minhash, bands, rows):
        self.minhash = minhash
        self.bands = bands
        self.rows = rows
        self.buckets = [{} for _ in range(bands)]

    def _hash_band(self, band):
        return hash(band)

    def insert(self, key, minhash_signature):
        for band_idx in range(self.bands):
            start_idx = band_idx * self.rows
            end_idx = (band_idx + 1) * self.rows
            band = minhash_signature[start_idx:end_idx]
            bucket_idx = self._hash_band(band)
            if bucket_idx not in self.buckets[band_idx]:
                self.buckets[band_idx][bucket_idx] = []
            self.buckets[band_idx][bucket_idx].append(key)

    def detect_all_duplicates(self, data):
        all_duplicates = {}
        for idx, entry in enumerate(data):
            shingle_set = set(entry['instruction'])
            signature = self.minhash.compute(shingle_set)
            similar_items = self.query(signature)
            if idx in similar_items:
                similar_items.remove(idx)
            if similar_items:
                all_duplicates[idx] = similar_items
        return all_duplicates

    def query(self, minhash_signature):
        candidates = set()
        for band_idx in range(self.bands):
            start_idx = band_idx * self.rows
            end_idx = (band_idx + 1) * self.rows
            band = minhash_signature[start_idx:end_idx]
            bucket_idx = self._hash_band(band)
            if bucket_idx in self.buckets[band_idx]:
                for candidate in self.buckets[band_idx][bucket_idx]:
                    candidates.add(candidate)
        return candidates

# --- Load Dataset ---
print("Loading dataset from Hugging Face Hub...")
dataset_name = "jondurbin/airoboros-2.2.1"  # Replace with your dataset name or path
data = load_dataset(dataset_name)
print("Dataset loaded successfully!")

# --- MinHash and LSH Setup ---
NUM_HASHES = 100
BANDS = 10
ROWS = 10

minhash = MinHash(NUM_HASHES)
lsh = LSH(minhash, BANDS, ROWS)

# Populate LSH with data
print("Processing data for LSH...")
for idx, entry in enumerate(data['train']):
    shingles = get_k_shingles(entry['instruction'])
    shingle_hashes = [hash(shingle) for shingle in shingles]  # Convert each shingle to a unique integer
    signature = minhash.compute(shingle_hashes)
    lsh.insert(idx, signature)

print("All entries processed for LSH!")

# Detect duplicates
print("Detecting duplicates...")
duplicates_map = lsh.detect_all_duplicates(data['train'])
print(f"Found potential duplicates for {len(duplicates_map)} entries.")

# Get all duplicate indices
print("Gathering duplicate indices...")
duplicate_indices = set()
for duplicates in duplicates_map.values():
    duplicate_indices.update(duplicates)

# Deduplicate
print("Deduplicating dataset...")
deduplicated_data = [entry for idx, entry in enumerate(data['train']) if idx not in duplicate_indices]

# Save deduplicated data
print("Saving deduplicated data to file...")
with open("deduplicated_data.json", 'w') as f:
    json.dump(deduplicated_data, f)

print("Deduplication complete! Deduplicated data saved to deduplicated_data.json.")
