from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save the model and tokenizer locally
tokenizer.save_pretrained("./local_model")
model.save_pretrained("./local_model")
