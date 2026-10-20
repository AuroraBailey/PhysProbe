# Image input and output policy

This policy applies to the main/stress and API ablation generation scripts.

| Provider | Uploaded pixels |
| --- | --- |
| Doubao, Gemini, Qwen Image 2.0, Wan 2.7 | Original file bytes; no local crop, padding or resize |
| Azure GPT Image 1.5 and 2 | RGB PNG, directly resized to 1536 × 1024 with Lanczos; no padding |

For GPT, every supplied image follows the policy, including convention exemplars
and segmentation references. Upload dimensions do not change requested output
size, quality, aspect-ratio messages, model deployment or other API parameters.

Responses are downloaded or base64-decoded and saved byte-for-byte. There is no
output crop, resize, grayscale conversion, normalization or polarity inversion.
The filename suffix does not imply conversion of the provider's byte encoding.
Target evaluators perform their own documented decoding, resizing and masking.

Roughness diagnostic A2 deliberately derives a region-filled map from the raw
response. Its `source_prediction` remains byte-exact; the derived map is a
separate diagnostic artifact. No other setting adds output processing.

CLI model identifiers are `doubao`, `gpt-image-1.5`, `gpt-image-2.0`, `qwen2`,
`wan`, `gemini`, where supported. `qwen2` selects `qwen-image-2.0`; `wan` selects
`wan2.7-image`. The obsolete `qwen` image-generation provider identifier is rejected.

