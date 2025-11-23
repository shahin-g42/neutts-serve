#!/usr/bin/env python3
"""
Diagnostic script to understand why vLLM generates 0 tokens with template_v2.
Run this to see what's different between the prompts.
"""

import os
os.environ['USE_TEMPLATE_V2'] = 'true'
os.environ['ENABLE_PHONEMIZATION'] = 'false'

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("shahink/avoice_v1")

# Simulate the template_v2 prompt construction
ref_codes = list(range(100, 150))  # 50 codes
ref_text = "مرحبا بك"
input_text = "هذا اختبار"

print("="*80)
print("DIAGNOSING VLLM ZERO TOKEN ISSUE")
print("="*80)
print()

# Build template v2 exactly as the code does
input_text_v2 = "<|TEXT_PROMPT_START|>" + ref_text + " " + input_text + "<|TEXT_PROMPT_END|>"

msg = [
    {
        "role": "system",
        "content": "You are an assistant/agent expert with text to speech in multi-languages and dialets. Convert the text to speech."
    },
    {
        "role": "user",
        "content": input_text_v2
    }
]

gen_prompt_ids = tokenizer.apply_chat_template(msg, add_generation_prompt=True)
speech_gen_start = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
ids = gen_prompt_ids + [speech_gen_start]

codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
codes = tokenizer.encode(codes_str, add_special_tokens=False)
final_ids = ids + codes

print(f"Prompt length: {len(final_ids)} tokens")
print()

# Decode to see the structure
decoded = tokenizer.decode(final_ids, skip_special_tokens=False)

print("PROMPT STRUCTURE:")
print("-" * 80)
print(decoded)
print("-" * 80)
print()

# Check critical points
print("CRITICAL CHECKS:")
print("-" * 80)

# 1. Check what comes after assistant header
if "<|start_header_id|>assistant<|end_header_id|>" in decoded:
    parts = decoded.split("<|start_header_id|>assistant<|end_header_id|>")
    assistant_content = parts[-1]
    print(f"1. Content after assistant header (first 200 chars):")
    print(f"   {repr(assistant_content[:200])}")
    print()
else:
    print("1. No assistant header found (using different template?)")
    print()

# 2. Check if prompt ends with speech codes
print(f"2. Prompt ends with (last 100 chars):")
print(f"   {repr(decoded[-100:])}")
print()

# 3. Check special tokens
speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
print(f"3. Special token IDs:")
print(f"   <|SPEECH_GENERATION_START|> = {speech_gen_start}")
print(f"   <|SPEECH_GENERATION_END|> = {speech_end_id}")
print(f"   First speech code <|speech_100|> = {codes[0] if codes else 'N/A'}")
print()

# 4. Check if stop token is in the prompt
if speech_end_id in final_ids:
    print(f"4. ⚠️  WARNING: Stop token <|SPEECH_GENERATION_END|> is IN the prompt!")
    print(f"   This would cause vLLM to stop immediately!")
else:
    print(f"4. ✓ Stop token NOT in prompt (good)")
print()

# 5. Check for any <eot_id> or end-of-turn tokens
eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
if eot_id and eot_id in final_ids:
    eot_positions = [i for i, x in enumerate(final_ids) if x == eot_id]
    print(f"5. ⚠️  Found <|eot_id|> tokens at positions: {eot_positions}")
    print(f"   Last <|eot_id|> is at position {eot_positions[-1]} of {len(final_ids)}")
    if eot_positions[-1] > len(final_ids) - 100:
        print(f"   This is VERY CLOSE to the end - might signal end of conversation!")
else:
    print(f"5. No <|eot_id|> tokens found")
print()

# 6. Check what vLLM would see as the "last token"
print(f"6. Last 10 token IDs in prompt:")
print(f"   {final_ids[-10:]}")
print(f"   Decoded: {tokenizer.decode(final_ids[-10:], skip_special_tokens=False)}")
print()

print("="*80)
print("HYPOTHESIS:")
print("="*80)
print()
print("If vLLM generates 0 tokens, it's likely because:")
print()
print("A. The prompt contains a stop token or end-of-turn signal")
print("B. vLLM sees the assistant already 'responded' with speech codes")
print("C. vLLM interprets the chat template differently than transformers")
print()
print("The fact that transformers works means the PROMPT is correct,")
print("but vLLM has different logic for handling chat templates.")
print()
