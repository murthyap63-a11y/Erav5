# data_system_engine.py
import os
import json
import hashlib
import numpy as np
import sys
from typing import List, Dict, Any, Tuple, Iterator
from pathlib import Path
from datetime import datetime, timezone
from tokenizers import Tokenizer  # Fix: Import the generic base class

"""
====OLD CODE FOR DEMONSTRATION PURPOSES====
class FrozenTokenizer:
    def __init__(self):
        self.vocab = {"<|endoftext|>": 1, "code": 2, "text": 3, "eval": 4}
        self.vocab_hash = hashlib.sha256(json.dumps(self.vocab, sort_keys=True).encode("utf-8")).hexdigest()

    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(w, 3) for w in text.lower().replace(",", "").replace(".", "").split()]
"""
class FrozenTokenizer:
    """A strictly Read-Only, Immutable Tokenizer deployed across worker clusters."""
    
    def __init__(self, vocab_path: str = "./model/frozen_tokenizer.json"):
        self.vocab_file = Path(vocab_path)
        
        # Guard layer: Force-create parent folders if they are missing
        self.vocab_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            print(f"📦 Attempting to load Frozen Tokenizer from: {self.vocab_file}")
            # Attempt to execute the native Rust backend file-reader
            self.tokenizer = Tokenizer.from_file(str(self.vocab_file.resolve()))
            
        except Exception as e:
            print(f"⚠️ State Warning: Failed to load tokenizer file natively. Error: {e}")
            print(f"🚨 Path Checked: {self.vocab_file.absolute()}")        
            print("🛑 Halting execution to preserve ledger integrity. Please run your Tokenizer [tokenizer.py] script first.")
            sys.exit(1)
        
        # Extract operational parameters safely
        vocab_dict = self.tokenizer.get_vocab()
        self.eos_token_id = vocab_dict.get("<|endoftext|>", 0)
        
        # Verify the ledger fingerprint
        with open(self.vocab_file, "rb") as f:
            self.vocab_hash = hashlib.sha256(f.read()).hexdigest()
        print(f"📦 Pure-Read Tokenizer initialized. Signed Signature: {self.vocab_hash}")

    def encode(self, text: str) -> List[int]:
        """Direct zero-overhead routing into the native Rust tokenizer engine."""
        return self.tokenizer.encode(text).ids

class ImmutableShardBuilder:
    def __init__(self, output_dir: Path, tokenizer: FrozenTokenizer):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer

    def _compute_file_sha256(self, file_path: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def create_shard(self, shard_id: int, documents: List[str], is_evaluation: bool = False) -> Tuple[Path, Path]:
        token_ids: List[int] = []
        for doc in documents:
            token_ids.extend(self.tokenizer.encode(doc))
            token_ids.append(self.tokenizer.eos_token_id)
            #token_ids.append(self.tokenizer.vocab["<|endoftext|>"])

        bin_filename = self.output_dir / f"shard_{shard_id:03d}.bin"
        manifest_filename = self.output_dir / f"shard_{shard_id:03d}.manifest.json"

        if bin_filename.exists(): os.chmod(bin_filename, 0o666)
        if manifest_filename.exists(): os.chmod(manifest_filename, 0o666)

        # ✅ WRITE CHECK: Guarantees at least default indexes are appended if text is un-parseable
        if not token_ids:
            token_ids = [self.tokenizer.eos_token_id] * 4

        with open(bin_filename, "wb") as f:
            for token in token_ids:
                f.write(token.to_bytes(2, byteorder="big", signed=False))

        os.chmod(bin_filename, 0o444)  # Explicitly force Read-Only Immutability
        binary_hash = self._compute_file_sha256(bin_filename)
        readable_timestamp = datetime.fromtimestamp(os.path.getmtime(bin_filename), tz=timezone.utc).isoformat()
        
        manifest_data = {
            # ─── System Identifiers & Governance ───
            "shard_id": shard_id,
            "is_evaluation_data": is_evaluation,
            #"lane_name": lane_name, # Tracking the precise domain pipeline origin
            "timestamp": readable_timestamp,
            
            # ─── Cryptographic Firewalls ───
            "content_sha256": binary_hash,
            "tokenizer_hash": self.tokenizer.vocab_hash,
            
            # ─── Binary Storage Engineering Specs ───
            "byte_size": bin_filename.stat().st_size,
            "data_type": "uint16",  # Safe up to 65,535 vocabulary size limits
            "byte_order": "big",    # Guarantees safe platform cross-loading
            
            # ─── Token Metrics & Learning Ledger Accounting ───
            "total_tokens": len(token_ids),
            #"sequence_count": total_sequences, # Number of context window blocks (e.g. rows)
            #"packing_utilization": round(packing_util, 4), # Total Non-Padding / Total Window Capacity
            #"useful_loss_bearing_tokens": useful_loss_tokens, # Active gradient tracking metrics
            #"padding_tokens_count": padding_count,
            
            # ─── Sequence Structure State Controls ───
            "packing_policy": "greedy_block_diagonal",
            "eos_token_id": int(self.tokenizer.eos_token_id)
        }

        with open(manifest_filename, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, sort_keys=True)
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
        if manifest["tokenizer_hash"] != self.tokenizer.vocab_hash:
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

class PackedSequenceBatcher:
    """Implements Greedy Packing with block-diagonal causal matrices, loss weights, and position ID resets."""
    def __init__(self, context_length: int = 8, pad_token_id: int = 0):
        self.context_length = context_length
        self.pad_token_id = pad_token_id

    def read_tokens_from_shard(self, bin_path: Path) -> List[int]:
        tokens = []
        with open(bin_path, "rb") as f:
            while byte_block := f.read(2):
                tokens.append(int.from_bytes(byte_block, byteorder="big", signed=False))
        return tokens

    def pack_tokens(self, tokens: List[int]) -> Iterator[Dict[str, Any]]:
        documents: List[List[int]] = []
        current_doc = []
        for t in tokens:
            current_doc.append(t)
            if t == 1:  # <|endoftext|>
                documents.append(current_doc)
                current_doc = []
        if current_doc: documents.append(current_doc)

        current_input_ids, current_loss_mask, current_position_ids = [], [], []
        doc_boundaries = []
        # Build the complete attention mask for this packed window
        attention_mask = self._generate_block_diagonal_mask(doc_boundaries)
        for doc in documents:
            while len(current_input_ids) + len(doc) > self.context_length:
                space_left = self.context_length - len(current_input_ids)
                chunk = doc[:space_left]
                current_input_ids.extend(chunk)
                current_loss_mask.extend([1.0] * len(chunk))
                
                for idx in range(len(chunk)): current_position_ids.append(idx)
                doc_boundaries.append((len(current_input_ids) - len(chunk), len(current_input_ids)))
                
                
                yield {
                    "input_ids": current_input_ids,
                    "loss_mask": current_loss_mask,
                    "position_ids": current_position_ids,
                    "attention_mask": attention_mask.tolist()
                }
                doc = doc[space_left:]
                current_input_ids, current_loss_mask, current_position_ids, doc_boundaries = [], [], [], []
            
            if doc:
                start_in_window = len(current_input_ids)
                current_input_ids.extend(doc)
                current_loss_mask.extend([1.0] * len(doc))
                for idx in range(len(doc)): current_position_ids.append(idx)
                doc_boundaries.append((start_in_window, len(current_input_ids)))

        if current_input_ids:
            padding_needed = self.context_length - len(current_input_ids)
            current_loss_mask.extend([0.0] * padding_needed)
            current_input_ids.extend([self.pad_token_id] * padding_needed)
            current_position_ids.extend([self.pad_token_id] * padding_needed)
            yield {
                "input_ids": current_input_ids,
                "loss_mask": current_loss_mask,
                "position_ids": current_position_ids,
                "attention_mask": attention_mask.tolist()
            }

    def _generate_block_diagonal_mask(self, boundaries: List[tuple]) -> np.ndarray:
        """Generates a causal block-diagonal attention mask matrix."""
        # 1. Initialize as standard fully un-routable matrix (False/0 value blocks)
        mask = np.zeros((self.context_length, self.context_length), dtype=bool)

        # 2. Apply a base standard causal lower-triangular mask
        causal_mask = np.tril(np.ones((self.context_length, self.context_length), dtype=bool))

        # 3. Restrict attention to within document boundaries only
        for start, end in boundaries:
            doc_block = np.zeros((self.context_length, self.context_length), dtype=bool)
            doc_block[start:end, start:end] = True
            mask = mask | (causal_mask & doc_block)
        return mask.astype(int)

# ==========================================
# SAFETY & ACCORDANCE FIREWALL
# ==========================================
class ContentValidationFirewall:
    """An inline validation firewall designed to screen out hazardous tokens or terms."""
    def __init__(self, banned_token_ids: List[int]):
        # Keep a strict blocklist signature
        self.banned_token_ids = set(banned_token_ids)

    def is_safe(self, token_ids: List[int]) -> bool:
        """Inspects token arrays before they transition to active training pipelines."""
        # Returns True if zero banned items are found, False if any intersect
        return self.banned_token_ids.isdisjoint(token_ids)

# ==========================================
# OPUS MULTI-LANE CURRICULUM ROUTER
# ==========================================

class OpusCurriculumEngine:
    """Manages multi-lane streaming mixture allocations, floors, and fallbacks."""
    def __init__(self, code_tokens: List[int], text_tokens: List[int]):
        self.lanes = {"code": code_tokens, "text": text_tokens}
        self.offsets = {"code": 0, "text": 0}

    def fetch_mixture(self, target_count: int, code_weight: float, text_weight: float, code_floor: int) -> Tuple[List[int], Dict[str, Any]]:
        mixed = []
        decisions = []
        
        # Enforce structural protected floor limits first
        if code_floor > 0:
            take = min(code_floor, len(self.lanes["code"]) - self.offsets["code"])
            if take == 0:  # Protected-floor override triggered
                decisions.append("OPUS [OVERRIDE]: Code stream dry! Forcing history replay sweep loop.")
                self.offsets["code"] = 0
                take = code_floor
            mixed.extend(self.lanes["code"][self.offsets["code"]:self.offsets["code"]+take])
            self.offsets["code"] += take
            decisions.append(f"OPUS [ACCEPTANCE]: Floor allocation extracted {take} tokens from code lane.")

        # Distribute residual targets across active weights
        remaining = target_count - len(mixed)
        if remaining > 0:
            code_share = int(remaining * code_weight)
            text_share = remaining - code_share
            
            # Fetch Code slice
            mixed.extend(self.lanes["code"][self.offsets["code"]:self.offsets["code"]+code_share])
            self.offsets["code"] += code_share
            
            # Fetch Text slice
            mixed.extend(self.lanes["text"][self.offsets["text"]:self.offsets["text"]+text_share])
            self.offsets["text"] += text_share
            decisions.append(f"OPUS [MIXED]: Balanced ratio slice step: {code_share} code, {text_share} text.")

        return mixed, {"decisions": decisions, "actual_shares": {"code": code_share + code_floor, "text": text_share}}


class LearningLedger:
    def __init__(self, ledger_path: Path):
        self.ledger_path = ledger_path
        self.state = self._load_or_create()

    def _load_or_create(self) -> Dict[str, Any]:
        """Safely loads tracking logs and patches structure on-the-fly to handle stale file remnants."""
        default_state = {"global_step": 0, "offsets": {"code": 0, "text": 0}, "history": [], "is_crashed": False}
        if self.ledger_path.exists():
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        # On-the-fly patch validation loop to prevent KeyErrors across development iterations
                        for key in default_state:
                            if key not in loaded:
                                loaded[key] = default_state[key]
                        return loaded
            except Exception:
                pass
        return default_state

    def commit_step(self, step: int, offsets: Dict[str, int], batch_hash: str, loss: float):
        self.state["global_step"] = step
        self.state["offsets"] = offsets.copy()
        self.state["history"].append({"step": step, "offsets": offsets.copy(), "hash": batch_hash, "loss": loss})
        
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
