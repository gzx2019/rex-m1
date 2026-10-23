"""Shared Qwen processor and token contract for Rex training and inference."""

SCORE_ID = 151669
STATE_ID = 151670
ACTION_START_ID = STATE_ID + 1
ACTION_END_ID = ACTION_START_ID + 60
DEFAULT_PROCESSOR = "Qwen/Qwen3-VL-4B-Instruct"


def build_processor(path=DEFAULT_PROCESSOR):
    from transformers import AutoProcessor

    tokens = {"score": "<score>", "state": "<state>"}
    tokens.update({f"a_{i}": f"<a_{i}>" for i in range(60)})
    processor = AutoProcessor.from_pretrained(
        path, local_files_only=True, use_fast=True, extra_special_tokens=tokens
    )
    processor.tokenizer.padding_side = "right"
    actual = processor.tokenizer.convert_tokens_to_ids(["<score>", "<state>", "<a_0>", "<a_59>"])
    expected = [SCORE_ID, STATE_ID, ACTION_START_ID, ACTION_END_ID - 1]
    if actual != expected:
        raise ValueError(f"Unexpected Rex token IDs from {path}: {actual}; expected {expected}")
    return processor
