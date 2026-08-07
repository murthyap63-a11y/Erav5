import os
import json
import hashlib
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Simulate a frozen, deterministic tokenizer 
# In production, replace this with: from transformers import AutoTokenizer
class FrozenTokenizer:
    """A simulated tokenizer with a fixed vocabulary and deterministic encoding."""
    def __init__(self):
        # Explicit vocabulary layout for auditing and immutability
        self.vocab = {"<|endoftext|>": 0, "the": 1, "project": 2, "phoenix": 3, "is": 4, "active": 5}
        self.vocab_hash = self._calculate_vocab_hash()

    def _calculate_vocab_hash(self) -> str:
        """Computes a frozen fingerprint of the vocabulary layout."""
        vocab_string = json.dumps(self.vocab, sort_keys=True)
        return hashlib.sha256(vocab_string.encode("utf-8")).hexdigest()

    def encode(self, text: str) -> List[int]:
        """Encodes text to token IDs. Unknown words drop to standard token ID 1."""
        tokens = []
        for word in text.lower().split():
            tokens.append(self.vocab.get(word, 1))
        return tokens

# ==========================================
# SHARD BUILDER ENGINE
# ==========================================

class ImmutableShardBuilder:
    """Handles creation of append-only binary shards with cryptographic manifests."""
    def __init__(self, output_dir: str, tokenizer: FrozenTokenizer):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer

    def _compute_file_sha256(self, file_path: Path) -> str:
        """Calculates SHA-256 hash of a file to ensure data immutability."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def create_shard(self, shard_id: int, documents: List[str]) -> Tuple[Path, Path]:
        """Tokenizes documents, writes a binary shard, and signs its manifest safely."""
        token_ids: List[int] = []
        
        for doc in documents:
            token_ids.extend(self.tokenizer.encode(doc))
            token_ids.append(self.tokenizer.vocab["<|endoftext|>"])

        bin_filename = self.output_dir / f"shard_{shard_id:04d}.bin"
        manifest_filename = self.output_dir / f"shard_{shard_id:04d}.manifest.json"

        # Safe Unlock: If the files already exist from a previous test run, temporarily unlock them
        if bin_filename.exists():
            os.chmod(bin_filename, 0o644)
        if manifest_filename.exists():
            os.chmod(manifest_filename, 0o644)

        # Write immutable token integers to raw binary
        with open(bin_filename, "wb") as f:
            for token in token_ids:
                f.write(token.to_bytes(2, byteorder="big", signed=False))

        # Enforce Immutability: Lock it back down to Read-Only for safety
        os.chmod(bin_filename, 0o444)

        # Calculate binary payload signature hash
        binary_hash = self._compute_file_sha256(bin_filename)

        manifest_data: Dict[str, Any] = {
            "shard_id": shard_id,
            "total_tokens": len(token_ids),
            "byte_size": bin_filename.stat().st_size,
            "data_type": "uint16",
            "content_sha256": binary_hash,
            "tokenizer_fingerprint": self.tokenizer.vocab_hash
        }

        with open(manifest_filename, "w") as f:
            json.dump(manifest_data, f, indent=4, sort_keys=True)
            
        # Lock manifest file too
        os.chmod(manifest_filename, 0o444) 

        return bin_filename, manifest_filename

# ==========================================
# AUDITING & FIREWALL VERIFICATION
# ==========================================

class DataIntegrityAuditor:
    """Verifies that data remains uncorrupted, unaltered, and completely reproducible."""
    def __init__(self, tokenizer: FrozenTokenizer):
        self.tokenizer = tokenizer

    def verify_shard(self, bin_path: Path, manifest_path: Path) -> bool:
        """Validates real data hashes against manifest entries to protect the runtime pipeline."""
        if not bin_path.exists() or not manifest_path.exists():
            print(f"❌ Verification Failed: Shard files missing at {bin_path.stem}")
            return False

        # Load manifest
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Check Tokenizer Match (Prevents running data processed with the wrong vocab)
        if manifest["tokenizer_fingerprint"] != self.tokenizer.vocab_hash:
            print(f"❌ Critical Error: Tokenizer mismatch on {bin_path.name}! Data corrupted or stale.")
            return False

        # Recalculate and audit cryptographic file hashes
        sha256_hash = hashlib.sha256()
        with open(bin_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        calculated_hash = sha256_hash.hexdigest()

        if calculated_hash != manifest["content_sha256"]:
            print(f"❌ Security Threat: {bin_path.name} content hash has changed! File is mutated.")
            return False

        print(f"✅ Data Audited Successfully: {bin_path.name} is intact, frozen, and secure.")
        return True


# ==========================================
# EXECUTION WORKFLOW
# ==========================================
if __name__ == "__main__":
    # Initialize immutable data assets
    frozen_tok = FrozenTokenizer()
    builder = ImmutableShardBuilder(output_dir="./data_shards", tokenizer=frozen_tok)
    auditor = DataIntegrityAuditor(tokenizer=frozen_tok)

    # Raw corpus data block input
    raw_corpus = [
        "The Project Phoenix is active",
        "The project is active"
    ]

    # Step A: Compile text inputs into secure binary shards + manifests
    bin_file, manifest_file = builder.create_shard(shard_id=1, documents=raw_corpus)
    print(f"Asset Created: {bin_file}")
    print(f"Manifest Generated: {manifest_file}\n")

    # Step B: Run pipeline startup verification checks
    is_valid = auditor.verify_shard(bin_file, manifest_file)
    
    # Step C: Simulating an unauthorized modification event (Malicious Mutation)
    try:
        # Switch permissions temporarily to force an illegal overwrite edit
        os.chmod(bin_file, 0o644)
        with open(bin_file, "ab") as f:
            f.write((99).to_bytes(2, byteorder="big")) # Append corrupt bytes
        print("\n[Simulation] Injected rogue corrupt tokens into data payload file...")
        
        # Rerun data system firewall checks
        auditor.verify_shard(bin_file, manifest_file)
    except Exception as e:
        print(f"Write block prevented or failed: {e}")
