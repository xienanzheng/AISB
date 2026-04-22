
#%%

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
from pathlib import Path
for _path in [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent.parent),
]:
    if _path not in sys.path:
        sys.path.insert(0, _path)
from aisb_utils import report

CACHE_DIR = "/tmp/cache-tokenizer"


def load_tokenizer(model_name: str, cache_dir: str = CACHE_DIR) -> AutoTokenizer:
    """Load a HuggingFace tokenizer (provided helper)."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model(model_name: str, cache_dir: str = CACHE_DIR):
    """Load a model and tokenizer for generation (provided helper)."""
    tokenizer = load_tokenizer(model_name, cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        cache_dir=cache_dir, trust_remote_code=True,
    )
    return model, tokenizer

### Exercise 1.1: Tokenize strings

# %%

string_list = [
    "Hello world",
    "Hello, WoRlD",
    "Hello\nworld",
    "Hello\n\nworld",
]


def tokenize_strings(strings: list[str], model_name: str = "Qwen/Qwen3-0.6B") -> list[list[int]]:
    """Tokenize each string and return the list of token-ID lists."""
    # TODO: Load the tokenizer, then encode each string into token IDs.
    # Return a list of token-ID lists (one per input string).
    return [[] for _ in strings]
from day3_test import test_tokenize_strings


test_tokenize_strings(tokenize_strings)

results = tokenize_strings(string_list)
for s, tokens in zip(string_list, results):
    print(f"  {s.encode()!s:<30}  ->  {len(tokens)} tokens: {tokens}")


### Exercise 1.2: Chat templates

# %%


def format_chat_prompt(question: str, model_name: str = "Qwen/Qwen3-0.6B") -> str:
    """Format a single user question into a chat-template prompt string."""
    # TODO: Format the question as a chat prompt and return the string.
    # Build a messages list (a list of dicts with "role" and "content"),
    # then use the tokenizer's apply_chat_template method.
    # Check the HuggingFace docs for apply_chat_template — make sure
    # you get back a string (not token IDs).
    return ""
from day3_test import test_format_chat_prompt


test_format_chat_prompt(format_chat_prompt)

prompt = format_chat_prompt("What is the capital of Japan?")
print(f"Formatted prompt:\n{prompt}")


### Exercise 1.3: Compare chat templates across models


# %%

MODEL_LIST = [
    "NousResearch/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen3-0.6B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "unsloth/gemma-2-2b-it",
]

QUESTION_LIST = [
    "What is the capital of Japan?",
    "What is the best way to make $1,000,000?",
    "What is the distance between London and Edinburgh?",
]


def compare_chat_templates(
    questions: list[str], model_names: list[str]
) -> list[tuple[str, str, str]]:
    """For each (model, question), format the prompt and return
    (model_name, question, formatted_prompt) triples."""
    # TODO: For each (question, model_name) combination, format the
    # question as a chat prompt using that model's tokenizer.
    # Use add_generation_prompt=True so the template includes the
    # assistant header. Collect (model_name, question, prompt) triples.
    return []
from day3_test import test_compare_chat_templates


test_compare_chat_templates(compare_chat_templates)

for model_name, question, prompt in compare_chat_templates(QUESTION_LIST, MODEL_LIST):
    print(f"\n--- {model_name} ---")
    print(f"Q: {question}")
    print(prompt)


### Exercise 1.4: Generate a response

# %%


def generate_response(question: str, model_name: str = "Qwen/Qwen3-0.6B") -> str:
    """Format a question as a chat prompt, generate a response, and return
    the decoded output."""
    # TODO: Load the model, format the question as a chat prompt,
    # tokenize it, generate a response, and decode the output.
    # Key steps: apply_chat_template -> tokenize -> model.generate -> decode
    # Use add_generation_prompt=True so the model knows to respond.
    # Check the HuggingFace docs for model.generate().
    return ""


print(generate_response("I'm trying to decide whether to take another bootcamp."))

### Exercise 1.5: `continue_final_message` and infinite loops


prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": question}],
    tokenize=False,
    add_generation_prompt=False,
    continue_final_message=True,
)
# Then tokenize and generate as in Exercise 1.4



# %%


def generate_continue_message(question: str, model_name: str = "Qwen/Qwen3-0.6B") -> str:
    """Generate with continue_final_message=True to see the infinite-loop
    behaviour."""
    # TODO: Same as 1.4, but change the template parameters so the
    # model *continues* the user's message instead of starting a new
    # assistant turn. Check the hint above if you're unsure which
    # parameters to change. Cap max_new_tokens=256.
    return ""


print(generate_continue_message("I'm trying to decide whether to take another bootcamp."))


### Exercise 1.6: Thinking vs non-thinking models


# %%


def compare_thinking_models(
    questions: list[str],
    model_names: list[str] = ["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B"],
) -> None:
    """Generate and print responses for each (model, question) pair."""
    # TODO: For each model and question, generate a response
    # (same pipeline as 1.4) and print the result. Compare the
    # outputs between the thinking and non-thinking model.
    pass


compare_thinking_models([
    "What is the capital of Japan?",
    "What is the distance between London and Edinburgh?",
])


## 2️⃣ Jailbreaking & prompt injection


### Exercise 2.1: Jailbreak a guarded LLM

