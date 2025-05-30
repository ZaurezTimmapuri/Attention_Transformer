import warnings
import torch
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct",
                                          device_map = "auto",
                                          torch_dtype = "auto",
                                          trust_remote_code = "true")
print(model) #to get model insights
# pipeline
generator = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    return_full_text = False,
    max_new_tokens = 50,
    do_sample = False
)

prompt = "what is capital of India"
output = generator(prompt)
print(output[0]['generated_text'])

# sequential processing as in transformers

prompt = "The capital of France is"
input_tokens = tokenizer(prompt,return_tensors='pt').input_ids #tokenize the data to obtain its respective input_ids
print(input_tokens)
model_output = model.model(input_tokens) # to flow between transfomer blocks
print(model_output[0].shape)
lm_head_output = model.lm_head(model_output[0])
print(lm_head_output.shape)
final_token_id = lm_head_output[0,-1].argmax(-1)
print(tokenizer.decode(final_token_id))