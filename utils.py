from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from gtts import gTTS
import pyttsx3
import pygame

# Initialize pipelines, models, and tokenizer
pipe = pipeline("text-generation", model="distilgpt2")
generator = pipeline("text-generation", model="openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_model = AutoModel.from_pretrained("bert-base-uncased")

# Define the Attention Layer class
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, embeddings):
        scores = self.attention(embeddings)  # Shape: (batch_size, seq_length, 1)
        scores = F.softmax(scores, dim=1)    # Normalize across sequence length
        context_vector = torch.sum(scores * embeddings, dim=1)  # Weighted sum of embeddings
        return context_vector, scores

# Initialize the attention layer
input_dim = 768
attention_layer = AttentionLayer(input_dim)

# Function to preprocess input and get embeddings
def preprocess_and_get_embeddings(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)
    return embeddings, inputs['input_ids']

# Function to extract the magic word using the attention mechanism
def get_magic_word_with_attention(prompt):
    embeddings, input_ids = preprocess_and_get_embeddings(prompt, tokenizer, embedding_model)
    _, attention_weights = attention_layer(embeddings)
    magic_word_index = torch.argmax(attention_weights, dim=1).item()
    magic_word_token = input_ids[0, magic_word_index]
    magic_word = tokenizer.decode(magic_word_token, skip_special_tokens=True)
    return magic_word

# Function to initialize and play background music using pygame
def play_background_music(music_file):
    pygame.mixer.init()
    pygame.mixer.music.load(music_file)
    pygame.mixer.music.play(-1)  # Loop the music

# Function to stop background music
def stop_background_music():
    pygame.mixer.music.stop()
    pygame.mixer.quit()

# Generate a story using the attention-based magic word approach
def generate_magic_story(user_input):
    magic_word = get_magic_word_with_attention(user_input)
    story_prompt = f"Once upon a time, there was a story about {magic_word}..."
    story = generator(story_prompt, max_new_tokens=200)[0]['generated_text']
    
    # Convert story text to speech using gTTS and save it as an audio file
    audio_path = "output/magic_story_audio.mp3"
    tts = gTTS(story, lang="en")
    tts.save(audio_path)
    
    return story, audio_path

# Function to generate a story using a basic pipeline and TTS
def generate_story(user_input):
    response = pipe(
        f"You mentioned: {user_input}. Here's a reflection on your day: ",
        max_length=500,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=50256
    )
    story = response[0]['generated_text'] if response else "Sorry, I couldn't generate a response this time."
    
    # Convert story text to speech using gTTS and save it as an audio file
    audio_path = "output/basic_story_audio.mp3"
    tts = gTTS(story, lang="en")
    tts.save(audio_path)
    
    return story, audio_path
