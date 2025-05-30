from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

def show_tokens(text,name):
    tokenizer = AutoTokenizer.from_pretrained(name) # using pre-trained tokenizer for tokenisation
    token_ids = tokenizer(text).input_ids # obataining ids of individual tokens
    print(f"vocab length : {tokenizer}")
    print(token_ids)                        #obtaining token_ids for individual tokens
    for tokens in token_ids:
        print(tokenizer.decode(tokens)) #based upo the ids decoding the encoded tokens

text = "This is a illustration of obtaining tokens, from different models who face issues on word capitalization"

# show_tokens(text,"bert-base-cased")  #lower vocab - issues in tokenisation  - 28996
show_tokens(text,"Xenova/gpt-4")  # higher vocab count - better tokenisation  - 100263