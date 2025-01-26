import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model
model_name = "PygmalionAI/pygmalion-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

def chat():
    print("Pygmalion Chatbot (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_length=200)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        print("Bot:", response)

if __name__ == "__main__":
    chat()
