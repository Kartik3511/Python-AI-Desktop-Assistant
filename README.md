# J.A.R.V.I.S. — Autonomous Voice Desktop Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/Platform-Windows%2010%20%2F%2011-0078D6?style=for-the-badge&logo=windows&logoColor=white" alt="Windows"/>
  <img src="https://img.shields.io/badge/STT-Faster--Whisper-FF6F00?style=for-the-badge" alt="Faster Whisper"/>
  <img src="https://img.shields.io/badge/TTS-Edge--TTS%20(RyanNeural)-008080?style=for-the-badge" alt="Edge-TTS"/>
  <img src="https://img.shields.io/badge/LLM-Groq%20%7C%20Gemini%20%7C%20OpenAI-blueviolet?style=for-the-badge" alt="LLMs"/>
  <img src="https://img.shields.io/badge/Storage-SQLite3%20ACID-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite"/>
</p>

An intelligent, local-first voice desktop assistant inspired by Iron Man's J.A.R.V.I.S., engineered specifically for Windows. Features an on-device wake word listener, dynamic voice activity detection, sub-millisecond local intent routing, an auto-recovering multi-LLM fallback cascade with streaming TTS, and persistent SQLite memory with proactive workspace automation.

---

## ⚡ Highlights & Key Features

* **🎙️ On-Device Wake-Word Detection:** Continuous microphone listening using `openwakeword` (`hey_jarvis` model) running 100% locally with a finely calibrated detection threshold.
* **🔇 Acoustic Anti-Feedback & Noise Gating:** Active audio buffer draining and speaker muting during playback prevents echo self-triggering loops. Multi-frame voice confirmation ($\ge 2$ chunks) and an RMS energy floor gate ($RMS \ge 180$) discard breaths, clicks, and fan noise.
* **⚡ Zero-Latency Fast-Path Router ($< 1\text{ms}$):** Deterministic commands (volume, brightness, app launching, lock workstation, YouTube/Google search, date/time) execute locally on-device in under 1 millisecond—**bypassing cloud LLMs completely and using 0 API tokens**.
* **🔗 Multi-Intent Compound Commands:** Handles complex chained commands in a single breath (e.g. *"Set brightness to 50% and turn volume to 100%"* or *"Open Edge and mute audio"*).
* **🧠 Auto-Recovering Multi-LLM Cascade:**
  $$\textbf{Groq (Qwen 2.5 27B)} \xrightarrow{\text{fallback}} \textbf{Google Gemini (3.6 Flash)} \xrightarrow{\text{fallback}} \textbf{OpenAI (GPT-4o-mini)}$$
  Sticky provider resilience automatically routes requests to backup models on rate limits (`429`) or timeouts without interrupting the user.
* **🛡️ Fallback Tool Idempotency Cache:** A rolling 15-second deduplication cache prevents side-effect tools (`open_website`, `open_application`, `git_commit`) from opening duplicate tabs or processes if a provider fails mid-turn.
* **🔊 Streaming Neural TTS Pipeline:** Synthesizes speech sentence-by-sentence using Microsoft Edge TTS (`en-GB-RyanNeural`) through `pygame.mixer`, dropping time-to-speech from 6 seconds down to under 2 seconds.
* **🔄 Autonomous Follow-Up Mode:** Automatically keeps the microphone active after executing actions or asking questions without requiring the wake word again, and dismisses instantly when you say *"No thanks"* or *"That'll be all"*.
* **💾 Persistent SQLite Memory:** ACID-compliant storage (`data/jarvis_memory.db`) storing user preferences, long-term semantic knowledge (`remember_fact`, `recall_memory`), and session audit history across reboots.
* **🖥️ Proactive Desktop Workspaces:** Instant one-phrase workspace orchestration (`coding`, `study`, `focus`, `media`, `meeting`) that launches required apps, adjusts audio volume, and sets screen brightness simultaneously.
* **🛠️ Developer & System Tools:** Integrated Git management (`git status`, `git commit`, `git pull`), live weather (`wttr.in`), web search, and hardware telemetry (CPU, RAM, disk, battery).

---

## 🏗️ Architecture Overview

```
                          [ User Voice ]
                                │
                                ▼
                  ┌───────────────────────────┐
                  │  Listener (openWakeWord)  │
                  └─────────────┬─────────────┘
                                │ (Wake Word Detected)
                                ▼
                  ┌───────────────────────────┐
                  │ Transcriber (Whisper+VAD) │
                  └─────────────┬─────────────┘
                                │ (Transcription)
                                ▼
                  ┌───────────────────────────┐
                  │ IntentRouter (Fast-Path)  │ ──<1ms──► [Direct System Execution]
                  └─────────────┬─────────────┘                 (Vol / Bright / Apps / YT / WS)
                                │ (Unmatched Complex Queries)
                                ▼
                  ┌───────────────────────────┐
                  │ Brain (Provider Cascade)  │
                  │  Groq ➔ Gemini ➔ OpenAI   │ ◄────► [SQLite Memory & Tools]
                  └─────────────┬─────────────┘
                                │ (Streamed Sentences)
                                ▼
                  ┌───────────────────────────┐
                  │  Speaker (EdgeTTS+Pygame) │
                  └─────────────┬─────────────┘
                                │
                        [ Audio Playback ]
                                │
                  (Follow-up expected? ──► Return to Listening)
```

---

## 📁 Repository Structure

```
jarvis_assistant/
├── core/
│   ├── listener.py            # Audio streaming, openWakeWord, WebRTC VAD, RMS gate
│   ├── transcriber.py         # Faster-Whisper, Silero VAD, hallucination suppression
│   ├── intent_router.py       # Zero-latency local regex command classifier (<1ms)
│   ├── memory_manager.py      # SQLite persistent memory, preferences, workspaces
│   ├── brain.py               # Provider orchestration, streaming, fallback, turn sync
│   ├── speaker.py             # Edge-TTS (RyanNeural), pipelined pygame streaming
│   └── providers/
│       ├── base.py            # Base provider interface, tool registry & dedup cache
│       ├── groq_provider.py   # Groq API implementation (Qwen 2.5 27B)
│       ├── gemini_provider.py # Google GenAI SDK (Gemini 3.6 Flash)
│       └── openai_provider.py # OpenAI API implementation (GPT-4o-mini)
├── tools/
│   ├── system_ops.py          # Volume (PyCAW), brightness (WMI), app lifecycle, lock
│   ├── web_ops.py             # DuckDuckGo search, weather, YouTube/Google search, browser
│   ├── dev_ops.py             # Git integration (status, commit, pull), shell execution
│   └── memory_ops.py          # Memory & workspace tools callable by LLMs
├── data/
│   └── jarvis_memory.db       # SQLite database (preferences, workspaces, facts, turns)
├── resources/                 # Audio assets (earcon chimes)
├── config.yaml                # Master configuration (API keys, models, VAD, persona)
├── main.py                    # Main application loop & state machine
└── requirements.txt           # Python dependencies
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
* **Operating System:** Windows 10 or Windows 11 (64-bit)
* **Python:** Python 3.10 or 3.11
* **Hardware:** Working microphone and speakers

### 2. Clone and Setup Environment
```powershell
# Navigate to the project directory
cd jarvis_assistant

# Create a virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys
Edit `config.yaml` with your preferred API keys. Providers without keys or with invalid keys are automatically skipped by the fallback orchestrator:

```yaml
providers:
  - name: "groq"
    model: "qwen/qwen3.8-27b"
    api_key: "YOUR_GROQ_API_KEY"

  - name: "gemini"
    model: "gemini-3.6-flash"
    api_key: "YOUR_GEMINI_API_KEY"

  - name: "openai"
    model: "gpt-4o-mini"
    api_key: "YOUR_OPENAI_API_KEY"
```

### 4. Run J.A.R.V.I.S.
```powershell
python main.py
```
Wait for the initialization chime:
> *"Jarvis online. At your service, sir."*

Say **"Hey Jarvis"** to begin.

---

## 🗣️ Voice Commands Cheatsheet

### ⚡ Fast-Path System Controls (Executed locally in $< 1\text{ms}$)
| Command | Action |
| :--- | :--- |
| *"Set volume to 50%"* / *"Volume up by 15"* / *"Mute audio"* | Adjusts master volume via PyCAW |
| *"Set brightness to 80%"* / *"Dim the display"* / *"Max brightness"* | Sets screen brightness via WMI |
| *"Open Microsoft Edge"* / *"Launch VS Code"* / *"Open Spotify"* | Launches applications directly |
| *"Close Notepad"* / *"Kill Chrome"* | Safely terminates running processes |
| *"Search Sidemen on YouTube"* / *"Play lofi on YouTube"* | Opens YouTube search in default browser |
| *"Open YouTube"* / *"Open GitHub"* / *"Open Reddit"* | Navigates directly to common web portals |
| *"Lock workstation"* / *"Lock my screen"* | Locks Windows immediately |
| *"System telemetry"* / *"Battery status"* | Reports CPU, RAM, disk, and battery levels |
| *"What time is it?"* / *"Current time in Tokyo"* | Provides localized or timezone-aware time |

### 🔗 Compound Chained Commands
* *"Dim the screen and mute audio."*
* *"Set brightness to 50% and turn volume to 100%."*
* *"Open Microsoft Edge and launch coding workspace."*

### 🖥️ Proactive Workspaces
* *"Launch coding workspace"* $\rightarrow$ Opens VS Code & Edge, volume 40%, brightness 80%.
* *"Activate study mode"* $\rightarrow$ Opens Edge & Notepad, volume 20%, brightness 70%.
* *"Focus mode"* $\rightarrow$ Dimmed display (50%), muted volume (10%), zero distracting apps.
* *"Media workspace"* $\rightarrow$ Opens Spotify & YouTube, volume 75%, brightness 90%.

### 💾 Persistent Long-Term Memory
* *"Remember that my favorite programming language is Python."*
* *"Recall what you know about my project."*
* *"Set my default city to London."*

---

## 🛡️ Key Engineering Challenges Solved

1. **Multi-LLM Fallback Side-Effect Cascades:**
   * *Problem:* When an API rate limit (`429`) occurred after executing a tool, fallback models re-executed the tool, resulting in multiple browser tabs or processes opening.
   * *Solution:* Engineered a rolling 15-second idempotency cache in `core/providers/base.py` that intercepts duplicate tool calls and returns cached results across provider failovers.
2. **End-to-End Latency Reduction (8s $\rightarrow$ <2s):**
   * *Problem:* Sequential execution (Audio $\rightarrow$ STT $\rightarrow$ Cloud LLM $\rightarrow$ TTS $\rightarrow$ Playback) took 6–8 seconds per turn.
   * *Solution:* Built a dual-speed pipeline: deterministic system queries are resolved locally in $<1\text{ms}$, while complex queries use sentence-level streaming TTS pipelined concurrently with model token generation.
3. **Acoustic Feedback & Silence Hallucinations:**
   * *Problem:* The microphone picked up J.A.R.V.I.S.'s own voice, triggering self-listening loops; in follow-up silence, Whisper hallucinated phrases like *"Thank you very much"*.
   * *Solution:* Implemented hardware buffer draining, acoustic muting, multi-frame voice confirmation ($\ge 2$ chunks), RMS energy floor gating, and Silero VAD filtering.

---

## 📜 License
Developed as a personal autonomous voice assistant. Distributed under the MIT License.
