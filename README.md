# 🕹️ **Offline Command-Line Video Game** 🕹️

### **Mystery Adventure Game**  
This is an interactive, offline command-line video game that uses Ollama to run small AI models locally. The game incorporates two AI models (DeepSeek-r1 & Qwen-vl) for generating and responding to the game story, Whisper for voice input, and LangChain to manage interactions between models and handle game logic.

---

### 🎮 **Game Features**  
- **Text-based Adventure**: AI-generated storylines based on your decisions
- **Voice Input Support**: Speak your actions and watch them unfold
- **Image Inspiration**: Visual prompts to influence in-game events
- **AI Sidekick**: Get help from a friendly AI companion
- **Offline Functionality**: Fully playable without the need for an internet connection

---

### 🛠️ **Prerequisites**  
Before diving into the fun, make sure you have the following:

- Python 3.8 or higher installed  
- **Ollama** installed and running on your system  
- **Git** (for cloning the repository)

---
### 🚀 **Installation**  
Follow these steps to get your game up and running:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/aarushsi/Offline-Command-Line-Video-Game.git

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```
   On macOS/Linux:
   ```bash
   source venv/bin/activate
4. ***Install required packages***
   ```bash
   pip install -r requirements.txt
6. After downloading Ollama, download & configure the models below:
   ```bash
   ollama pull deepseek-r1:1.5b
   ollama pull bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh
---

### 🎮 **How to play**
To start the game, run:
```bash
python game.py
```

