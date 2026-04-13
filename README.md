# LLM Systems Engineering Assessment

Hey! Thanks for the assessment. I had a lot of fun diving into the lower-level mechanics of these models instead of just hitting API endpoints. Here is my approach, along with some honest reflections on building this out locally.

## Design Decisions
For the parser (Task 1), I decided to use the `rich` library to generate a visual, hierarchical tree rather than dumping a massive nested dictionary. When you're debugging architectures, seeing exactly where the LayerNorms sit inside the Attention blocks—and instantly seeing the parameter counts for each block—is just a much better developer experience. 

For the models, I used `TinyLlama` (1.1B) to show the parser can handle larger, modern architectures, but I specifically chose `GPT-2` (124M) for the fine-tuning and merging pipeline. I'll explain why in the working conditions below.

## Working Conditions & Bottlenecks
To be completely transparent, the biggest bottleneck I faced was my hardware. I ran everything locally on my laptop with the following setup:
* **GPU:** NVIDIA GTX 1650 (Strict 4GB VRAM limit)
* **CPU:** AMD Ryzen 5 4600H with Radeon Graphics
* **RAM:** 16GB DDR4

If I tried to load and fine-tune a 1.1B or 3.8B model, my 4GB GPU would instantly hit an Out-Of-Memory (OOM) error. To solve this, I:
1. Used GPT-2 for the training pipeline.
2. Kept the LoRA rank very low (`r=8`).
3. Wrote a custom PyTorch training loop instead of using HuggingFace's `Trainer`. This gave me granular control over the batch size (kept at 8) and gradient accumulation, ensuring my VRAM stayed exactly where I needed it. 

The pipeline works perfectly end-to-end (training took exactly 23.45 seconds for a small proof-of-concept dataset), and it's built to easily scale to larger models if deployed on better hardware.

## Extensibility & Scalability
**Scaling to 10-50 models:**
If we scale this to parse 50 models, the current script would crash simply because RAM/VRAM can't hold 50 `nn.Module` objects at once. The first thing that breaks is system memory. To fix this, I would implement lazy loading using `safetensors` metadata. Instead of downloading and loading entire weights, the system would only fetch the configuration headers to build the architecture trees. 

**Different Architectures (Llama vs Qwen):**
My parser is already highly resilient to different architectures because it uses PyTorch's native `named_children()` recursively. It doesn't look for hardcoded names like `q_proj`; it just traverses whatever graph is there. 

## Creativity & Future Vision
If I were to evolve this into an automated model-improvement system, here are three ideas I'd explore:

1. **"Night-Shift" Consolidation:** Instead of constantly merging weights, the system logs edge-case user prompts where the model performed poorly during the day. At night, an automated pipeline spins up, creates micro-datasets from these failures, trains a small LoRA, runs an automated eval against a golden dataset, and only merges it if the overall score improves.
2. **Dynamic Adapter Routing:** Merging everything into the base model can lead to catastrophic forgetting. A cooler approach would be parsing the architecture and attaching multiple LoRA adapters. A tiny, fast classifier network acts as a "router" at the input, sending the prompt to the right adapter based on context.
3. **Cross-Architecture Translation:** Using the parser to find mathematically identical sub-structures in completely different model families, and attempting to distill/transfer fine-tuned adapter weights from one open-source model to another.

## Honest Reflection
Setting up the LoRA config via `peft` was straightforward—HuggingFace makes that almost too easy. 

The challenging part was dealing with GPT-2's legacy architecture. GPT-2 uses `Conv1D` for its attention layers instead of standard `nn.Linear`. When it came time to merge the weights, I had to ensure `peft` was correctly mapping the `fan_in_fan_out` logic for Conv1D layers so the math didn't break during `merge_and_unload()`. 

**External Help & AI Tools:**
To be completely transparent, I used an LLM-assisted IDE (Cursor) to help speed up writing boilerplate code, specifically for generating the `rich` library tree syntax and scaffolding the standard PyTorch training loop. However, the core architectural choices—such as selecting GPT-2 to fit my 4GB VRAM, restricting the LoRA rank to `r=8`, targeting the `c_attn` modules, and the custom batch-size handling—were entirely my own design decisions based on reading the `peft` documentation. I treated the LLM as a junior assistant to execute my logic faster, making the final solution completely my own.
