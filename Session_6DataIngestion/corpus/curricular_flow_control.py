import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

# Seamless integration with prior infrastructure steps
from data_governance import FrozenTokenizer, ImmutableShardBuilder
from sequence_packing import PackedSequenceBatcher

# ==========================================
# 1. SAFETY & ACCORDANCE FIREWALL
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
# 2. OPUS MULTI-LANE CURRICULUM ROUTER
# ==========================================

class OpusCurriculumEngine:
    """
    Manages multi-lane ingestion streams, enforcing dynamic mix weights,
    protected floor constraints, and automated floor overrides.
    """
    def __init__(self, lanes_tokens: Dict[str, List[int]], validation_firewall: ContentValidationFirewall):
        self.lanes = lanes_tokens  # Key: Lane Name (e.g., 'code'), Value: List of Token IDs
        self.firewall = validation_firewall
        
        # Track read head offsets for each distinct lane stream
        self.offsets = {lane_name: 0 for lane_name in lanes_tokens.keys()}

    def sample_batch_stream(
        self, 
        lane_weights: Dict[str, float], 
        protected_floors: Dict[str, int], 
        requested_tokens: int
    ) -> List[int]:
        """
        Samples and mixes text inputs based on curriculum settings.
        Handles Acceptances, Rejections, and Protected-Floor Overrides automatically.
        """
        combined_payload: List[int] = []
        
        # 1. Enforce Protected Floors First (Mandatory Minimum Token Counts)
        for lane_name, floor_limit in protected_floors.items():
            if floor_limit > 0:
                tokens_fetched = self._harvest_lane_tokens(lane_name, floor_limit)
                combined_payload.extend(tokens_fetched)
                requested_tokens -= len(tokens_fetched)

        # Normalize remaining lane weights to allocate the leftover token budget
        remaining_lanes = [l for l in lane_weights.keys() if lane_weights[l] > 0]
        if not remaining_lanes or requested_tokens <= 0:
            return combined_payload

        total_weight = sum(lane_weights[l] for l in remaining_lanes)
        normalized_weights = {l: lane_weights[l] / total_weight for l in remaining_lanes}

        # 2. Distribute remaining space across curriculum lanes
        for lane_name in remaining_lanes:
            lane_share = int(requested_tokens * normalized_weights[lane_name])
            if lane_share > 0:
                tokens_fetched = self._harvest_lane_tokens(lane_name, lane_share)
                combined_payload.extend(tokens_fetched)

        return combined_payload

    def _harvest_lane_tokens(self, lane_name: str, amount: int) -> List[int]:
        """Fetches tokens while executing OPUS route actions: Acceptance, Rejection, and Override."""
        gathered: List[int] = []
        attempts = 0
        max_attempts = 3  # Avoid infinite loop thresholds

        while len(gathered) < amount and attempts < max_attempts:
            attempts += 1
            current_idx = self.offsets[lane_name]
            available_tokens = self.lanes[lane_name][current_idx:]

            # --- CRITICAL FLOW CONDITION: PROTECTED FLOOR OVERRIDE ---
            if len(available_tokens) == 0:
                print(f"⚠️ [OPUS OVERRIDE] Lane '{lane_name}' ran completely dry! Triggering synthetic fallback replay...")
                self.offsets[lane_name] = 0  # Reset read counter back to start (Loop / Replay)
                continue

            # Take an chunk slice
            chunk_end = current_idx + (amount - len(gathered))
            candidate_chunk = self.lanes[lane_name][current_idx:chunk_end]
            
            # Move the pointer tracking our position
            self.offsets[lane_name] += len(candidate_chunk)

            # --- SAFETY CHECK: FIREWALL RULES ---
            if self.firewall.is_safe(candidate_chunk):
                # [ACTION: ACCEPTANCE]
                gathered.extend(candidate_chunk)
            else:
                # [ACTION: REJECTION]
                print(f"🛑 [OPUS REJECTED] Malicious content blocked in lane '{lane_name}'. Skipping block sequence.")
                # We do not append to gathered, it is discarded, and the index remains advanced.

        return gathered


# ==========================================
# SEAMLESS RUNTIME EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    print("--- Executing Step 1 & 2: Tokenization & Sharding Foundation ---")
    tokenizer = FrozenTokenizer()
    
    # Let's generate data variants simulating isolated information categories
    code_corpus = ["project project project project", "active active active"]
    text_corpus = ["the project phoenix is active", "the project is active"]
    toxic_corpus = ["this is toxic malicious text content"]  # Contains "malicious", which drops to token id 1
    
    # Write to immutable disk arrays
    builder = ImmutableShardBuilder(output_dir="./curriculum_shards", tokenizer=tokenizer)
    code_bin, _ = builder.create_shard(shard_id=101, documents=code_corpus)
    text_bin, _ = builder.create_shard(shard_id=102, documents=text_corpus)
    toxic_bin, _ = builder.create_shard(shard_id=103, documents=toxic_corpus)

    # Convert binary files back into memory token pools
    batcher = PackedSequenceBatcher(context_length=16)
    
    lane_database = {
        "code_lane": batcher.read_tokens_from_shard(code_bin),
        "text_lane": batcher.read_tokens_from_shard(text_bin),
        "security_risk_lane": batcher.read_tokens_from_shard(toxic_bin)
    }

    print("\n--- Executing Step 3: Flow Routing, Firewalls & Curriculum ---")
    # Identify explicit token targets to act as unsafe triggers (e.g. mapping word index '1' as forbidden)
    # Token ID '1' is our default out-of-vocabulary word, which fires on unknown toxic inputs here
    firewall = ContentValidationFirewall(banned_token_ids=[1])
    opus_engine = OpusCurriculumEngine(lanes_tokens=lane_database, validation_firewall=firewall)

    # Define stage curriculum constraints (e.g., Phase 1 focuses heavily on technical code injection)
    stage_one_weights = {"code_lane": 0.80, "text_lane": 0.20, "security_risk_lane": 0.00}
    stage_one_floors = {"code_lane": 4, "text_lane": 2, "security_risk_lane": 0}

    print(">> Action: Fetching Batch 1 using Stage 1 Weights...")
    mixed_tokens_1 = opus_engine.sample_batch_stream(
        lane_weights=stage_one_weights, 
        protected_floors=stage_one_floors, 
        requested_tokens=16
    )
    print(f"Resulting Mixed Token Layout: {mixed_tokens_1}")
    
    # Process the curriculum-mixed tokens back through our packing pipeline from Step 2
    packed_blocks = list(batcher.pack_tokens(mixed_tokens_1))
    print(f"Packed Mask Coordinates for Mixed Batch 1 Input IDs: {packed_blocks[0]['input_ids']}")

    print("\n>> Action: Fetching Batch 2 containing Safety Trigger Violations...")
    # Attempting to draw data directly out of a contaminated stream
    stage_two_weights = {"code_lane": 0.00, "text_lane": 0.00, "security_risk_lane": 1.00}
    mixed_tokens_2 = opus_engine.sample_batch_stream(
        lane_weights=stage_two_weights, 
        protected_floors={"code_lane": 0, "text_lane": 0, "security_risk_lane": 0}, 
        requested_tokens=8
    )
    print(f"Resulting Mixed Token Layout (Should be empty due to Firewall Rejection): {mixed_tokens_2}")

    print("\n>> Action: Forcing a Protected-Floor Override by consuming more code data than exists...")
    # Intentionally reading a tiny lane pool repeatedly to activate automated replay triggers
    for i in range(3):
        opus_engine.sample_batch_stream(
            lane_weights={"code_lane": 1.0, "text_lane": 0.0, "security_risk_lane": 0.0},
            protected_floors={"code_lane": 0, "text_lane": 0, "security_risk_lane": 0},
            requested_tokens=12
        )
