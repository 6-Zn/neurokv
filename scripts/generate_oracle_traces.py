#!/usr/bin/env python3
"""
Real Oracle Trace Generation

Generate oracle traces using actual model attention weights for imitation learning.
This creates ground-truth importance labels from real inference, not synthetic data.
"""

import os
import sys
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, '/home/gloria/workspace/homework/neuronetwork/final')

import torch
import json
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from neurokv.oracle import OracleConfig, AttributionMethod, OracleAttribution
from neurokv.policy import PolicyFeatures
from baselines.unified_interface import get_cache_length, get_cache_layers, get_layer_kv


class RealOracleTraceGenerator:
    """
    Generate oracle traces using real model attention weights.

    Hooks into model attention to capture actual attention patterns,
    then computes importance scores and action labels.
    """

    def __init__(
        self,
        model_name: str = 'Qwen/Qwen2.5-0.5B-Instruct',
        oracle_config: OracleConfig = None,
    ):
        self.model_name = model_name
        self.oracle_config = oracle_config or OracleConfig(
            method=AttributionMethod.ATTENTION_ACCUMULATED,
            tier1_ratio=0.15,
            tier2_ratio=0.35,
        )

        # Load model
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True,
            attn_implementation='eager',  # Use eager attention to get output_attentions
        )

        # Get model info
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.num_heads

        print(f"Model config: {self.num_layers} layers, {self.num_heads} heads, {self.head_dim} head_dim")

        # Storage for attention weights
        self.attention_weights = {}  # layer -> tensor
        self.hooks = []

    def register_hooks(self):
        """Register forward hooks to capture attention weights."""
        # Find attention layers
        attention_layers = self._find_attention_layers()

        print(f"Found {len(attention_layers)} attention layers")

        for idx, layer in enumerate(attention_layers):
            hook = layer.register_forward_hook(self._attention_hook(idx))
            self.hooks.append(hook)

        return len(self.hooks)

    def _find_attention_layers(self):
        """Find self-attention modules in model."""
        attention_layers = []

        # Common patterns
        patterns = ['self_attn', 'attention', 'attn']

        for name, module in self.model.named_modules():
            for pattern in patterns:
                if pattern in name.lower():
                    # Check for attention indicators
                    if hasattr(module, 'q_proj') or hasattr(module, 'query'):
                        attention_layers.append(module)
                        break

        return attention_layers

    def _attention_hook(self, layer_idx: int):
        """Create hook to capture attention weights."""
        def hook(module, input, output):
            # Try different ways to get attention weights
            # For some models, attention weights are returned in output
            if isinstance(output, tuple) and len(output) > 1:
                # Check if second element is attention weights
                attn = output[1]
                if attn is not None and hasattr(attn, 'shape'):
                    self.attention_weights[layer_idx] = attn.detach().cpu()

            # Also check if module stores attention weights
            if hasattr(module, '_attention_weights'):
                self.attention_weights[layer_idx] = module._attention_weights.detach().cpu()

        return hook

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_trace(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> dict:
        """
        Generate oracle trace for a prompt.

        Returns:
            trace: Dictionary with importance scores and labels for each step
        """
        # Reset
        self.attention_weights = {}
        oracle = OracleAttribution(self.oracle_config)

        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.model.device)
        prompt_length = input_ids.shape[1]

        print(f"Prompt length: {prompt_length} tokens")

        # Register hooks
        self.register_hooks()

        # Storage
        trace_steps = []

        # Generate
        generated_ids = input_ids.clone()
        past_key_values = None

        try:
            with torch.no_grad():
                # Prefill - request attention output
                outputs = self.model(
                    generated_ids,
                    use_cache=True,
                    output_attentions=True,  # Explicitly request attention weights
                )
                past_key_values = outputs.past_key_values

                # Capture prefill attention from outputs
                if outputs.attentions:
                    for layer_idx, attn in enumerate(outputs.attentions):
                        if attn is not None:
                            self.attention_weights[layer_idx] = attn.detach().cpu()

                    prefill_data = self._process_prefill_attention(
                        self.attention_weights,
                        prompt_length,
                        oracle,
                    )
                    if prefill_data:
                        trace_steps.append(prefill_data)

                # Reset for decode
                self.attention_weights = {}

                # Decode loop
                for step in range(max_new_tokens):
                    outputs = self.model(
                        generated_ids[:, -1:].contiguous(),
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_attentions=True,  # Request attention weights
                    )
                    past_key_values = outputs.past_key_values

                    current_length = get_cache_length(past_key_values)

                    # Process decode attention from outputs
                    if outputs.attentions:
                        for layer_idx, attn in enumerate(outputs.attentions):
                            if attn is not None:
                                self.attention_weights[layer_idx] = attn.detach().cpu()

                        decode_data = self._process_decode_attention(
                            self.attention_weights,
                            current_length - 1,
                            step,
                            oracle,
                        )
                        if decode_data:
                            trace_steps.append(decode_data)

                    # Reset attention weights for next step
                    self.attention_weights = {}

                    # Next token
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        finally:
            self.remove_hooks()

        # Build trace
        trace = {
            "prompt": prompt[:200],  # Store truncated prompt
            "prompt_length": prompt_length,
            "generation_length": max_new_tokens,
            "model": self.model_name,
            "num_layers": self.num_layers,
            "oracle_config": {
                "method": self.oracle_config.method.value,
                "tier1_ratio": self.oracle_config.tier1_ratio,
                "tier2_ratio": self.oracle_config.tier2_ratio,
            },
            "steps": trace_steps,
            "num_steps": len(trace_steps),
            "generated_text": self.tokenizer.decode(
                generated_ids[0][prompt_length:],
                skip_special_tokens=True
            )[:100],
        }

        return trace

    def _process_prefill_attention(
        self,
        attention_weights: dict,
        prompt_length: int,
        oracle: OracleAttribution,
    ) -> dict:
        """Process attention weights from prefill phase."""
        # For prefill, attention weights are (batch, heads, prompt_len, prompt_len)
        # We use the last query position (attending to all previous)

        all_importances = []
        all_labels = []
        all_features = []

        for layer_idx, attn in attention_weights.items():
            if attn.shape[-1] < prompt_length:
                continue

            # Use attention from last query to all previous
            last_query_attn = attn[:, :, -1:, :]  # (batch, heads, 1, prompt_len)

            importance, labels = oracle.generate_labels_for_layer(
                last_query_attn, layer_idx, prompt_length - 1
            )

            # Extract features
            features = PolicyFeatures.extract_from_attention(
                last_query_attn, prompt_length - 1, layer_idx, importance
            )

            all_importances.append(importance)
            all_labels.append(labels)
            all_features.append(features)

        if not all_importances:
            return None

        return {
            "phase": "prefill",
            "prompt_length": prompt_length,
            "importances": all_importances,
            "labels": all_labels,
            "features": all_features,
        }

    def _process_decode_attention(
        self,
        attention_weights: dict,
        current_position: int,
        step: int,
        oracle: OracleAttribution,
    ) -> dict:
        """Process attention weights from decode phase."""
        all_importances = []
        all_labels = []
        all_features = []

        for layer_idx, attn in attention_weights.items():
            # Decode: single query attending to all cached tokens
            importance, labels = oracle.generate_labels_for_layer(
                attn, layer_idx, current_position
            )

            # Extract features
            features = PolicyFeatures.extract_from_attention(
                attn, current_position, layer_idx, importance
            )

            all_importances.append(importance)
            all_labels.append(labels)
            all_features.append(features)

        if not all_importances:
            return None

        return {
            "phase": "decode",
            "step": step,
            "current_position": current_position,
            "importances": all_importances,
            "labels": all_labels,
            "features": all_features,
        }

    def save_traces(self, traces: list, path: str):
        """Save traces to JSON file."""
        # Convert tensors to lists for JSON serialization
        serializable_traces = []
        for trace in traces:
            serial_trace = {
                "prompt": trace["prompt"],
                "prompt_length": trace["prompt_length"],
                "generation_length": trace["generation_length"],
                "model": trace["model"],
                "num_layers": trace["num_layers"],
                "oracle_config": trace["oracle_config"],
                "num_steps": trace["num_steps"],
                "generated_text": trace["generated_text"],
            }

            # Convert steps
            serial_steps = []
            for step in trace["steps"]:
                serial_step = {
                    "phase": step["phase"],
                }
                if step["phase"] == "prefill":
                    serial_step["prompt_length"] = step["prompt_length"]
                else:
                    serial_step["step"] = step["step"]
                    serial_step["current_position"] = step["current_position"]

                # Convert tensor lists
                serial_step["importances"] = [
                    imp.tolist() if hasattr(imp, 'tolist') else imp
                    for imp in step["importances"]
                ]
                serial_step["labels"] = [
                    lab.tolist() if hasattr(lab, 'tolist') else lab
                    for lab in step["labels"]
                ]
                serial_step["features"] = [
                    feat.tolist() if hasattr(feat, 'tolist') else feat
                    for feat in step["features"]
                ]

                serial_steps.append(serial_step)

            serial_trace["steps"] = serial_steps
            serializable_traces.append(serial_trace)

        with open(path, 'w') as f:
            json.dump(serializable_traces, f, indent=2)

        print(f"Saved {len(traces)} traces to {path}")

    def load_traces(self, path: str) -> list:
        """Load traces from JSON file."""
        with open(path, 'r') as f:
            traces = json.load(f)

        # Convert back to tensors
        for trace in traces:
            for step in trace["steps"]:
                step["importances"] = [
                    torch.tensor(imp) for imp in step["importances"]
                ]
                step["labels"] = [
                    torch.tensor(lab) for lab in step["labels"]
                ]
                step["features"] = [
                    torch.tensor(feat) for feat in step["features"]
                ]

        print(f"Loaded {len(traces)} traces from {path}")
        return traces


def create_sample_prompts() -> list:
    """Create sample prompts for trace generation."""
    prompts = [
        # Long context prompt
        """The development of artificial intelligence has been one of the most significant technological achievements of the 21st century. Machine learning algorithms have revolutionized how we process data, make decisions, and interact with technology. Deep learning, a subset of machine learning, has enabled breakthroughs in image recognition, natural language processing, and autonomous systems.

Based on the above discussion, summarize the key achievements of AI development: """,

        # Short context prompt
        """What is machine learning? """,

        # Mathematical reasoning
        """Given the sequence: 2, 4, 8, 16, 32, what is the next number? """,

        # Question answering
        """The Amazon rainforest covers about 5.5 million square kilometers and is home to an estimated 400 billion trees. It spans nine countries in South America.

How many countries does the Amazon rainforest span? """,
    ]

    return prompts


def main():
    parser = argparse.ArgumentParser(description="Generate real oracle traces")
    parser.add_argument('--output', type=str, default='data/oracle_traces.json',
                        help='Output path for traces')
    parser.add_argument('--num-traces', type=int, default=4,
                        help='Number of traces to generate')
    parser.add_argument('--max-tokens', type=int, default=30,
                        help='Max new tokens to generate')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='Model name')
    args = parser.parse_args()

    print("=" * 60)
    print("Real Oracle Trace Generation")
    print("=" * 60)

    # Create oracle config
    oracle_config = OracleConfig(
        method=AttributionMethod.ATTENTION_ACCUMULATED,
        tier1_ratio=0.15,
        tier2_ratio=0.35,
    )

    # Initialize generator
    generator = RealOracleTraceGenerator(
        model_name=args.model,
        oracle_config=oracle_config,
    )

    # Create prompts
    prompts = create_sample_prompts()

    # Generate traces
    traces = []
    for i, prompt in enumerate(prompts[:args.num_traces]):
        print(f"\nGenerating trace {i+1}/{args.num_traces}...")
        print(f"Prompt preview: {prompt[:50]}...")

        trace = generator.generate_trace(prompt, max_new_tokens=args.max_tokens)
        traces.append(trace)

        print(f"Generated: {trace['generated_text'][:50]}...")

    # Save traces
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generator.save_traces(traces, args.output)

    print("\n" + "=" * 60)
    print(f"Generated {len(traces)} oracle traces")
    print(f"Saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()