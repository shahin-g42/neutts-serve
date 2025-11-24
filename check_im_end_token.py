#!/usr/bin/env python3
"""
Check if <|im_end|> token is present after reference codes
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("shahink/avoice_v1")

# Get token IDs
im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
speech_gen_start_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
speech_gen_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

print("="*80)
print("CHECKING FOR <|im_end|> TOKEN ISSUE")
print("="*80)
print()
print(f"<|im_start|> token ID: {im_start_id}")
print(f"<|im_end|> token ID: {im_end_id}")
print(f"<|SPEECH_GENERATION_START|> token ID: {speech_gen_start_id}")
print(f"<|SPEECH_GENERATION_END|> token ID: {speech_gen_end_id}")
print()

# Build the template v2 prompt
ref_codes = list(range(100, 150))
ref_text = "مرحبا بك"
input_text = "هذا اختبار"

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
ids = gen_prompt_ids + [speech_gen_start_id]

codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
codes = tokenizer.encode(codes_str, add_special_tokens=False)
final_ids = ids + codes

print("="*80)
print("CHECKING TOKEN SEQUENCE")
print("="*80)
print()

# Find <|im_end|> positions
im_end_positions = [i for i, token_id in enumerate(final_ids) if token_id == im_end_id]

if im_end_positions:
    print(f"Found {len(im_end_positions)} <|im_end|> tokens at positions: {im_end_positions}")
    print()
    
    last_im_end_pos = im_end_positions[-1]
    print(f"Last <|im_end|> is at position {last_im_end_pos} of {len(final_ids)} total tokens")
    print(f"Distance from end: {len(final_ids) - last_im_end_pos} tokens")
    print()
    
    if last_im_end_pos == len(final_ids) - 1:
        print("⚠️  CRITICAL: <|im_end|> is the LAST token in the prompt!")
        print("   vLLM will see this as end of conversation and generate nothing!")
    elif len(final_ids) - last_im_end_pos < len(ref_codes):
        print("⚠️  CRITICAL: <|im_end|> comes AFTER some reference codes!")
        print(f"   Reference codes after last <|im_end|>: {len(final_ids) - last_im_end_pos}")
        print("   vLLM might stop at this <|im_end|> token!")
    else:
        print("✓ <|im_end|> is before all reference codes (normal)")
    print()
    
    # Show what comes after last <|im_end|>
    if last_im_end_pos < len(final_ids) - 1:
        tokens_after = final_ids[last_im_end_pos+1:]
        decoded_after = tokenizer.decode(tokens_after, skip_special_tokens=False)
        print(f"Tokens AFTER last <|im_end|>:")
        print(f"  Count: {len(tokens_after)}")
        print(f"  First 10 IDs: {tokens_after[:10]}")
        print(f"  Decoded (first 200 chars): {decoded_after[:200]}")
else:
    print("No <|im_end|> tokens found in prompt (unusual!)")

print()
print("="*80)
print("SOLUTION")
print("="*80)
print()
print("If <|im_end|> comes AFTER the reference codes, vLLM stops there.")
print("The fix is to ensure <|im_end|> comes BEFORE adding reference codes,")
print("or don't add <|im_end|> at all after assistant starts.")
print()
print("Current structure:")
decoded = tokenizer.decode(final_ids, skip_special_tokens=False)
if "<|im_end|>" in decoded:
    # Count how many times it appears
    count = decoded.count("<|im_end|>")
    print(f"  <|im_end|> appears {count} time(s) in the prompt")
print()
