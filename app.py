import gradio as gr
from PIL import Image
from utils import generate_magic_story, generate_story  # Ensure utils.py has these functions

# Load the logo image from the assets directory
logo = Image.open("assets/logo/green_chronicles.png")

# Set up Gradio interface
with gr.Blocks() as interface:
    # Display logo at the top
    gr.Image(logo, elem_id="logo", show_label=False)
    gr.Markdown("<h1 style='text-align: center;'>The Green Chronicles</h1>")
    gr.Markdown("<h3 style='text-align: center; color: green;'>AI Generated Eco Tales - One story at a time</h3>")

    # Radio button for story type selection
    story_type = gr.Radio(
        ["Magic Word Story", "Basic Story"],
        label="Select Story Type",
        value="Basic Story"
    )
    
    # Textbox for user input
    user_input = gr.Textbox(
        label="Describe Your Day",
        placeholder="Share a brief glimpse of your day...",
        lines=2
    )
    
    # Output elements
    story_output = gr.Textbox(label="Generated Eco-Tale")
    audio_output = gr.Audio(autoplay=True, label="Story Audio")

    # Function to choose which story generation method to run
    def main_generate(user_input, story_type):
        if story_type == "Magic Word Story":
            story, audio_path = generate_magic_story(user_input)
        else:
            story, audio_path = generate_story(user_input)
        return story, audio_path

    # Generate button and linking to the callback function
    generate_button = gr.Button("Generate Story")
    generate_button.click(fn=main_generate, inputs=[user_input, story_type], outputs=[story_output, audio_output])

# Launch the Gradio app with share enabled for remote access
interface.launch(share=True)
