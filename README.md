#  Automated AI YouTube Shorts Generator

An end-to-end automated pipeline that generates, renders, and schedules YouTube Shorts using **Google Gemini**, **Edge TTS**, **OpenAI Whisper**, and **MoviePy**. Available as both a **CLI tool** and an interactive **Flask Web App** with real-time render tracking.

---

##  Features

- **Automated Story Generation**: Uses Google's `gemini-2.5-flash` model to craft engaging, viral-ready stories (Scary/Horror, Reddit drama, or custom user niches) with cliffhangers and catchy titles in a single API call.
- **Natural AI Voiceover**: Free, high-fidelity neural voice generation using Microsoft's `edge-tts`.
- **Accurate Word-Synced Subtitles**: Transcribes audio and aligns dynamic, TikTok/Shorts-style subtitles using OpenAI's `whisper` and `Pillow`.
- **Dynamic Video Composition**: Loops background footage seamlessly, mixes suspenseful background audio for horror stories, and renders 1080x1920 portrait MP4s via `MoviePy`.
- **Direct YouTube Uploads & Scheduling**: Uses the official Google YouTube Data API v3 to upload and schedule videos at customizable intervals (e.g., +1 hour per video).
- **Two User Interfaces**:
  - **Flask Web UI**: Interactive dashboard with real-time rendering progress bars powered by `proglog`.
  - **Console App (CLI)**: Interactive terminal prompts to batch-render videos sequentially.
- **Fail-Safe Quota Handler**: Immediately halts background jobs if your YouTube Data API upload quota is exhausted to prevent wasted rendering resources.

---

##  Repository Structure

```text
├── background_footage/     # Directory containing your raw portrait background videos (.mp4)
├── finished_videos/        # Output directory for rendered Shorts (.mp4)
├── stories/                # Saved voiceover audio files (.mp3)
├── templates/
│   ├── index.html          # Web UI setup & configuration form
│   └── progress.html       # Web UI real-time render progress page
├── app.py                  # Flask Web Application entrypoint
├── console_app.py          # Interactive Terminal/CLI script
├── background_music.mp3    # (Optional) Ambient background music for horror/scary stories
├── client_secret.json      # Google Cloud OAuth credentials file (YouTube Uploads)
├── impact.ttf              # Font file for rendering dynamic subtitles
├── requirements.txt        # Python package dependencies
└── README.md
```

---

##  Prerequisites

1. **Python 3.10+** installed.
2. **FFmpeg** installed and added to your system's `PATH` (Required by MoviePy and Whisper):
   - **Windows (Chocolatey)**: `choco install ffmpeg`
   - **macOS (Homebrew)**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`
3. A **Google Gemini API Key** ([Get one from Google AI Studio](https://aistudio.google.com/)).
4. A **Google Cloud Console Project** with the **YouTube Data API v3** enabled and an OAuth 2.0 Client ID created.

---

##  Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/EshrakHuq/ClipMaker/
cd ClipMaker
```

### 2. Set Up a Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

##  Configuration

### 1. Add Your Gemini API Key
Open `app.py` and `console_app.py`, locate this line near the top, and paste your API key directly:

```python
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "PASTE_YOUR_GEMINI_KEY_HERE")
```
*(Alternatively, you can export `GEMINI_API_KEY` as an environment variable in your terminal).*

---

### 2. Add YouTube Data API Credentials (`client_secret.json`)
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project and enable the **YouTube Data API v3**.
3. Go to **OAuth Consent Screen** (set User Type to **External**, add your Google account email as a **Test User**).
4. Go to **Credentials** → **Create Credentials** → **OAuth client ID** → choose **Desktop app**.
5. Download the JSON credentials file, rename it to `client_secret.json`, and place it in the root folder of this project.

---

### 3. Add Media Assets
- **Background Footage**: Drop one or more vertical (`9:16` aspect ratio, 1080x1920) `.mp4` video clips into the `background_footage/` folder (e.g. gameplay, satisfying videos).
- **Background Music**: Place a `background_music.mp3` file into the root folder (this will be mixed in automatically whenever a "Scary" genre is selected).
- **Font**: Ensure you have an `impact.ttf` font file inside the root folder for rendering subtitles.

---

##  Usage

### Option 1: Web Interface (Flask)
Start the local server:
```bash
python app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```
1. Select how many videos you want to generate (1–10).
2. Choose your genre mode (Random, Preset, or Custom Prompt).
3. Click **Generate** to watch the real-time render progress bar.

---

### Option 2: Console Interface (CLI)
Run the script in your terminal:
```bash
python console_app.py
```
Follow the interactive prompts to define how many videos to make and configure specific genres/prompts for each video in the batch.

---

## Dependencies (`requirements.txt`)

```text
flask
google-genai
edge-tts
openai-whisper
Pillow
moviepy>=2.0.0
proglog
google-auth-oauthlib
google-auth-httplib2
google-api-python-client
```

---

##  Notes & YouTube Quotas

- **YouTube API Quotas**: Default Google Cloud projects receive **10,000 units per day**. A single video upload consumes **1,600 units**, meaning you can upload roughly **6 videos per day** before hitting your free daily quota. The generator will stop automatically if it hits this limit.
- **First-Time Authorization**: The first time you upload, a browser tab will open asking you to sign into your Google account and grant upload permissions. Once authorized, it saves a `token.pickle` file so you won't need to sign in every time.
- **Whisper Performance**: Audio transcription runs on the CPU by default (`fp16=False`). If you have a CUDA-compatible NVIDIA GPU, Whisper will automatically run faster using your GPU.
