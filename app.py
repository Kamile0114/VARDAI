import streamlit as st
import torch
import json
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

# Define the Transformer model
class GenderAwareTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, forward_expansion):
        super(GenderAwareTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gender_embed = nn.Embedding(2, embed_size)  # 2 for male/female
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, embed_size))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def forward(self, x, gender):
        char_embeddings = self.embed(x)
        gender_embeddings = self.gender_embed(gender).unsqueeze(1).expand(-1, x.size(1), -1)
        x = char_embeddings + gender_embeddings + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = self.output_layer(x)
        return x

# Load mappings
with open('name_mappings.json', 'r', encoding='utf-8') as f:
    mappings = json.load(f)

char_to_int = mappings['char_to_int']
int_to_char = {int(k): v for k, v in mappings['int_to_char'].items()}
vocab_size = mappings['vocab_size']

# Load the model
model = GenderAwareTransformer(vocab_size=vocab_size, embed_size=128, num_heads=8, forward_expansion=4)
model.load_state_dict(torch.load('name_model.pt'))
model.eval()

# Sampling function
def sample(model, gender, start_str='a', max_length=20, temperature=1.0):
    assert temperature > 0, "Temperature must be greater than 0"
    with torch.no_grad():
        start_str = start_str.lower()
        chars = [char_to_int[c] for c in start_str]
        input_seq = torch.tensor(chars).unsqueeze(0)
        gender_tensor = torch.tensor([gender])

        output_name = start_str
        last_char = start_str[-1]

        for _ in range(max_length - len(start_str)):
            output = model(input_seq, gender_tensor)
            probabilities = torch.softmax(output[0, -1] / temperature, dim=0)
            next_char_idx = torch.multinomial(probabilities, 1).item()
            next_char = int_to_char[next_char_idx]

            if next_char == last_char or next_char == ' ':
                break

            output_name += next_char
            input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]])], dim=1)

        return output_name.capitalize()

# Streamlit app styling with custom CSS
st.markdown("""
    <style>
         .stApp {
            background-color: #F0F3E7;
        }
        .css-18e3th9 {
            color: #606C38;
        }
        .css-1emrehy {
            background-color: #606C38;
            color: white;
        }
        .css-1emrehy:hover {
            background-color: #283618;
        }
        .stButton>button {
            background-color: #C6CEAA;
            color: white;
            border-radius: 8px;
            font-family: 'Times New Roman', serif;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #283618;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.title("Lietuviškų vardų generatorius")

gender = st.radio("Pasirinkti:", ["Vyro vardas", "Moters vardas"])
start_letter = st.text_input("Įrašykite pirmąją norimo vardo raidę:", value="A")
temperature = st.slider("Generatoriaus kūrybiškumas:", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

if st.button("Generuoti naują vardą"):
    gender_val = 0 if gender == "Vyro vardas" else 1
    generated_name = sample(model, gender=gender_val, start_str=start_letter, temperature=temperature)
    st.write(f"Sugeneruotas vardas: **{generated_name}**")
