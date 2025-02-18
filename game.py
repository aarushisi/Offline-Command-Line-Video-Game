from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationChain
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama.llms import OllamaLLM
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import re

def strip_thinking(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

# Instantiate memory, this will store the entire conversation history
memory = ConversationBufferMemory(return_messages=True)

# Intialize the models I downloaded from Ollama
deepseek_model = OllamaLLM(model="deepseek-r1:1.5b")
qwen_vl_model = OllamaLLM(model="bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh")

# Prompt templates
story_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are a storyteller. Generate the next part of the story based on the player's action.
Context: {history}
Player action: {input}
Story continuation:"""
)

image_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are analyzing an image to influence the story. Describe how the image affects the current situation.
Context: {history}
Image description: {input}
Story influence:"""
)

# Conversation chain with deepseek model
# Story and Image Chains
# Chain definitions
story_chain = (
    RunnablePassthrough.assign(history=lambda x: memory.load_memory_variables({})["history"])
    | story_template
    | deepseek_model
    | StrOutputParser()
)

image_chain = (
    RunnablePassthrough.assign(history=lambda x: memory.load_memory_variables({})["history"])
    | image_template
    | qwen_vl_model
    | StrOutputParser()
)

# Router function
def route_input(input_data):
    if isinstance(input_data, dict) and input_data.get("image_path"):
        return "image"
    return "story"

# Create the prompt templates given the templates I created
# prompts = [
#     PromptTemplate(template=story_template, input_variables=["context", "input"]),
#     PromptTemplate(template=image_template, input_variables=["context", "input"])
# ]

# Main chain
chain = RunnableLambda(
    lambda x: story_chain.invoke({"input": x}) if route_input(x) == "story" else image_chain.invoke({"input": x})
)

# Set up the router chain
# router_output_parser = RouterOutputParser()

# router_prompt = PromptTemplate(
#     template="Given the input: {input}, decide whether to use the story model or the image model.\n"
#              "Output format: {'destination': 'story' or 'image', 'next_inputs': {'input': 'user_input'}}",
#     input_variables=["input"],
#     output_parser=router_output_parser  # Add the output parser here
# )


# router_chain = LLMRouterChain.from_llm(deepseek_model, prompt=router_prompt, verbose=True)


# # Create destination chains
# story_chain = ConversationChain(llm=deepseek_model, memory=memory, verbose=True)
# image_chain = ConversationChain(llm=qwen_vl_model, memory=memory, verbose=True)

# Create the multi-prompt chain for routing
# chain = MultiPromptChain(
#     router_chain=router_chain,
#     destination_chains={"story": story_chain, "image": image_chain},
#     default_chain=story_chain,
#     verbose=True
# )

def process_image(image_path):
    # Use Qwen-vk for image processing???
    return qwen_vl_model.predict(f"Describe the image and its influence on the story: {image_path}")

def get_sidekick_advice(history):
    prompt = f"""As the player's sidekick, provide advice on what to do next in the game based on the current situation.
    Game history: {history}
    Advice:"""
    return deepseek_model.invoke(prompt)

# Audio stff
whisper_model = whisper.load_model("base")

def record_audio(duration=5, fs=16000):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return recording

def transcribe_audio(audio):
    # Save the numpy array as a temporary WAV file
    wav.write("temp_audio.wav", 16000, audio)
    # Transcribe the audio file using Whisper
    result = whisper_model.transcribe("temp_audio.wav")
    return result["text"]


def game_interaction():
    print("Welcome to the Mystery Adventure Game!")

    # Generate game introduction using DeepSeek model
    intro = deepseek_model.invoke("Generate a a concise, intriguing one-paragraph introduction for a mystery adventure game.")
    intro = strip_thinking(intro)
    print(intro)

    while True:
        # Get user input (text or voice)
        user_input = input("""Your action:
        (type 'quit' to exit, 'voice' for voice input, 'image' for image input, 'help' for sidekick advice)
        Or enter any other action: """)

        if user_input.lower() == "quit":
                print("Thanks for playing!")
                break
        
        if user_input.lower() == "voice":
            print("Please speak your action. (Not implemented in this demo)")
            # In a real implementation, you would record audio here
            audio_path = record_audio()
            user_input = transcribe_audio(audio_path)
            print(f"You said: {user_input}")
        
        # Check if it's an image input & Generates a response using the AI models
        if user_input.lower() == "image":
            image_path = input("Enter the path to your image: ")
            image_description = process_image(image_path)
            print("Routing to image analysis...")
            response = chain.run(input=image_description)
        elif user_input.lower() == "help":
            history = memory.load_memory_variables({})["history"]
            advice = get_sidekick_advice(history)
            advice = strip_thinking(advice)
            print("Sidekick:", advice)
            continue
        else:
            # Process the user's actions using AI-generated responses
            print("Routing to story generation...")
            response = chain.invoke(user_input)
            response = strip_thinking(response)
        
        memory.save_context({"input": user_input}, {"output": response})
        # Displays the response to the user
        print("Game:", response)
    
# Start the game
if __name__ == "__main__":
    game_interaction()