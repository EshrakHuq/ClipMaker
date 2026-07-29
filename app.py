import os
import random
import tempfile
import uuid
import asyncio
import pickle
import threading
import whisper
from datetime import datetime, timedelta, timezone  
from google import genai  
import edge_tts  
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    VideoFileClip, 
    AudioFileClip, 
    CompositeVideoClip, 
    ImageClip, 
    CompositeAudioClip,
    concatenate_videoclips
)
from flask import Flask, render_template, request, redirect, url_for, jsonify

# Google API Client packages
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Progress Logger integration for MoviePy
try:
    from proglog import ProgressBarLogger
except ImportError:
    ProgressBarLogger = object

# Initialize Flask
app = Flask(__name__)

# Global progress state store
JOBS = {}

# ==============================================================================
# ENVIRONMENT AND DIRECTORY INITIALIZATION
# ==============================================================================
root_dir = os.path.dirname(os.path.abspath(__file__))
background_footage_dir = os.path.join(root_dir, "background_footage")
stories_dir = os.path.join(root_dir, "stories")
finished_videos_dir = os.path.join(root_dir, "finished_videos")

os.makedirs(stories_dir, exist_ok=True)
os.makedirs(finished_videos_dir, exist_ok=True)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "your-api-key-here")
client_gemini = genai.Client(api_key=GEMINI_API_KEY)

print("Loading Whisper Speech-to-Text Model...")
model = whisper.load_model("base")

# ==============================================================================
# MOVIEPY CUSTOM PROGRESS LOGGER
# ==============================================================================
class MoviePyJobLogger(ProgressBarLogger):
    def __init__(self, job_id, video_index, total_videos):
        super().__init__()
        self.job_id = job_id
        self.video_index = video_index
        self.total_videos = total_videos

    def callback(self, **changes):
        bars = self.state.get('bars', {})
        if 't' in bars:
            t_bar = bars['t']
            index = t_bar.get('index', 0)
            total = t_bar.get('total', 1)
            
            if total > 0:
                render_pct = int((index / total) * 100)
                base_pct = int((self.video_index / self.total_videos) * 100)
                slice_width = 100 / self.total_videos
                current_job_pct = base_pct + int(slice_width * (0.50 + (render_pct / 100.0) * 0.35))

                # Check to prevent overriding error statuses
                current_job = JOBS.get(self.job_id, {})
                if current_job.get("status") != "stopped_early":
                    JOBS[self.job_id] = {
                        "status": "processing",
                        "message": f"Rendering Video {self.video_index + 1}/{self.total_videos} - Frame {index}/{total}",
                        "progress": min(current_job_pct, 99)
                    }

# ==============================================================================
# AI CONTENT GENERATION (SINGLE PROMPT METHOD)
# ==============================================================================

def generate_story_and_title_gemini(mode="random", choice_detail=None):
    print(f"Generating story and title using Gemini API (Mode: {mode})...")
    formatting_instructions = (
        "\n\nStrict Formatting & Structural Rules:\n"
        "- Target length: Under 110 words total (essential for a 45-second TikTok/Short video voiceover).\n"
        "- Start immediately with a gripping hook.\n"
        "- Do not include an introduction or outro within the story text.\n"
        "- CLIFFHANGER RULE: Build up tension and abruptly cut off at a highly suspenseful cliffhanger, ending "
        "exactly with: '...Like for Part 2.' If the story is simple or comedic, you may finish it naturally.\n\n"
        "CRITICAL: You must format your output exactly as shown below so the system can parse the properties:\n"
        "GENRE: [Write either SCARY or OTHER here]\n"
        "TITLE: [Write a catchy high-engagement title here under 50 characters, no quotes]\n"
        "[STORY START]\n"
        "[Write the story text here]"
    )

    if mode == "random":
        prompt = (
            "Write a highly engaging, first-person story. Randomly choose between writing a "
            "SCARY story (like creepy r/nosleep horror stories) OR a dramatic/funny Reddit story.\n\n"
            "To ensure variety, randomly choose ONE of the following premises:\n"
            "1. THE SCARY / HORROR STORY (Tag as SCARY)\n"
            "2. THE DRAMATIC TWIST (Tag as OTHER)\n"
            "3. SHOCKING CONFESSIONS (Tag as OTHER)\n"
            "4. TEEN RELATIONSHIP CHAOS (Tag as OTHER)\n"
            "5. SATISFYING CLEVERNESS (Tag as OTHER)\n"
            "6. CHAOTIC COMEDY (Tag as OTHER)\n"
        ) + formatting_instructions
    elif mode == "preset":
        if choice_detail.upper() == "SCARY":
            prompt = (
                "Write a highly engaging, suspenseful, and terrifying first-person scary story (similar to r/nosleep).\n"
                "Incorporate supernatural elements, eerie isolation, or unsettling coincidences.\n"
                "Ensure you Tag the GENRE as SCARY."
            ) + formatting_instructions
        else:
            prompt = (
                f"Write a highly engaging, first-person Reddit-style narrative based on the genre: {choice_detail}.\n"
                "Focus on high-drama or chaotic comedy.\n"
                "Ensure you Tag the GENRE as OTHER."
            ) + formatting_instructions
    else:  
        prompt = (
            f"Write a highly engaging, first-person Reddit-style story based on this user prompt: '{choice_detail}'.\n"
            "Determine if this story is 'SCARY' or 'OTHER' and mark the tag accordingly."
        ) + formatting_instructions

    response = client_gemini.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    
    raw_text = response.text.strip()
    genre = "other"
    title = "Unbelievable Reddit Story"
    story_started = False
    story_lines = []
    
    lines = raw_text.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("GENRE:"):
            g_val = stripped.split(":", 1)[1].strip().upper()
            genre = "scary" if "SCARY" in g_val else "other"
        elif stripped.upper().startswith("TITLE:"):
            title = stripped.split(":", 1)[1].strip().replace('"', '').replace("'", "")
        elif stripped == "[STORY START]":
            story_started = True
        elif story_started:
            story_lines.append(line)
            
    if not story_lines:
        cleaned_lines = [l for l in lines if not l.upper().startswith("GENRE:") and not l.upper().startswith("TITLE:") and l.strip() != "[STORY START]"]
        story_text = "\n".join(cleaned_lines).strip()
    else:
        story_text = "\n".join(story_lines).strip()
        
    return {"text": story_text, "genre": genre, "title": title}

# ==============================================================================
# AUDIO GENERATION & TRANSCRIPTION
# ==============================================================================

async def save_edge_tts_audio(text, output_path):
    voice = "en-US-ChristopherNeural"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def generate_tts_audio_free(text, output_path):
    asyncio.run(save_edge_tts_audio(text, output_path))

def create_pil_subtitle_clip(text_str, font_name, font_size, color, stroke_color, stroke_width, temp_files):
    try:
        font = ImageFont.truetype(font_name, font_size)
    except Exception:
        font = ImageFont.load_default()

    left, top, right, bottom = font.getbbox(text_str)
    text_w = right - left
    text_h = bottom - top
    pad_x, pad_y = 30, 30
    img_w = int(text_w + pad_x * 2)
    img_h = int(text_h + pad_y * 2)

    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_x = pad_x - left
    draw_y = pad_y - top

    draw.text((draw_x, draw_y), text_str, font=font, 
              fill=color, stroke_width=stroke_width, stroke_fill=stroke_color)

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"sub_{uuid.uuid4().hex}.png")
    img.save(temp_path)
    temp_files.append(temp_path)
    return ImageClip(temp_path)

def generate_subtitles(audio_path, video_size, temp_files):
    result = model.transcribe(audio_path, word_timestamps=True, fp16=False)
    subtitle_clips = []
    w, h = video_size
    all_words = []
    for segment in result['segments']:
        if 'words' in segment:
            all_words.extend(segment['words'])

    chunks = []
    chunk_size = 3
    if all_words:
        for i in range(0, len(all_words), chunk_size):
            word_group = all_words[i:i+chunk_size]
            text_str = " ".join([w['word'].strip() for w in word_group]).upper()
            start_time = word_group[0]['start']
            end_time = word_group[-1]['end']
            chunks.append({'text': text_str, 'start': start_time, 'end': end_time})
    else:
        for segment in result['segments']:
            words = segment['text'].strip().split()
            if not words: continue
            seg_start = segment['start']
            seg_end = segment['end']
            seg_duration = seg_end - seg_start
            num_chunks = (len(words) + chunk_size - 1) // chunk_size
            duration_per_chunk = seg_duration / num_chunks if num_chunks > 0 else seg_duration
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i+chunk_size]
                text_str = " ".join(chunk_words).upper()
                start_time = seg_start + (i // chunk_size) * duration_per_chunk
                end_time = start_time + duration_per_chunk
                chunks.append({'text': text_str, 'start': start_time, 'end': end_time})

    font_name = "impact.ttf" 
    font_size = 80

    for chunk in chunks:
        text_str = chunk['text']
        start_time = chunk['start']
        end_time = chunk['end']
        duration = max(0.1, end_time - start_time)
        try:
            txt_clip = create_pil_subtitle_clip(text_str, font_name, font_size, 'yellow', 'black', 2, temp_files)
            safe_width = int(w * 0.8)
            if txt_clip.w > safe_width:
                txt_clip = txt_clip.resized(width=safe_width)
            vertical_position = int(h * 0.65)
            if vertical_position + txt_clip.h >= h:
                vertical_position = int(h - txt_clip.h - 20)
            txt_clip = txt_clip.with_start(start_time).with_duration(duration).with_position(('center', vertical_position))
            subtitle_clips.append(txt_clip)
        except Exception as e:
            print(f"Error creating Subtitle: {e}")
    return subtitle_clips

def adjust_background_duration(background_path, target_duration):
    temp_clip = VideoFileClip(background_path)
    base_duration = temp_clip.duration
    temp_clip.close()
    if base_duration < target_duration:
        n_loops = int(target_duration // base_duration) + 1
        clips = [VideoFileClip(background_path) for _ in range(n_loops)]
        looped_video = concatenate_videoclips(clips, method='compose')
        return looped_video.subclipped(0, target_duration)
    else:
        return VideoFileClip(background_path).subclipped(0, target_duration)

# ==============================================================================
# VIDEO COMPILATION ENGINE
# ==============================================================================

def create_video(story_id, genre, job_id=None, video_index=0, total_videos=1):
    audio_path = os.path.join(stories_dir, f"{story_id}.mp3")
    background_footage_files = [f for f in os.listdir(background_footage_dir) if f.endswith('.mp4')]
    if not background_footage_files:
        return None
    background_file = random.choice(background_footage_files)
    background_path = os.path.join(background_footage_dir, background_file)
    tts_audio = AudioFileClip(audio_path)
    adjusted_background = adjust_background_duration(background_path, tts_audio.duration)
    temp_files = []
    
    subtitle_clips = generate_subtitles(audio_path, adjusted_background.size, temp_files)
    max_duration = min(adjusted_background.duration, tts_audio.duration)

    valid_subtitle_clips = [c for c in subtitle_clips if c.start < max_duration]
    for i, c in enumerate(valid_subtitle_clips):
        if c.start + c.duration > max_duration:
            valid_subtitle_clips[i] = c.with_duration(max_duration - c.start)

    final_video = CompositeVideoClip([adjusted_background] + valid_subtitle_clips, use_bgclip=True) if valid_subtitle_clips else adjusted_background
    final_video = final_video.with_duration(max_duration)
    
    music_path = os.path.join(root_dir, "background_music.mp3") 
    bg_music = None
    if genre == "scary" and os.path.exists(music_path):
        try:
            bg_music = AudioFileClip(music_path).with_duration(max_duration).with_volume_scaled(0.12)
            combined_audio = CompositeAudioClip([tts_audio, bg_music])
        except Exception:
            combined_audio = tts_audio
    else:
        combined_audio = tts_audio

    final_video = final_video.with_audio(combined_audio)
    output_path = os.path.join(finished_videos_dir, f'{story_id}.mp4')

    custom_logger = None
    if job_id and ProgressBarLogger is not object:
        custom_logger = MoviePyJobLogger(job_id, video_index, total_videos)

    final_video.write_videofile(
        output_path, 
        codec='libx264', 
        audio_codec='aac', 
        fps=24, 
        preset='ultrafast',
        logger=custom_logger
    )

    final_video.close()
    adjusted_background.close()
    tts_audio.close()
    if bg_music: bg_music.close()

    for temp_path in temp_files:
        try: os.remove(temp_path)
        except Exception: pass
    return output_path

# ==============================================================================
# YOUTUBE UPLOADER
# ==============================================================================

def get_youtube_service():
    scopes = ["https://www.googleapis.com/auth/youtube.upload"]
    client_secrets_file = os.path.join(root_dir, "client_secret.json")
    token_file = os.path.join(root_dir, "token.pickle")
    credentials = None
    if os.path.exists(token_file):
        with open(token_file, "rb") as token:
            credentials = pickle.load(token)
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            try: credentials.refresh(Request())
            except Exception: credentials = None
        if not credentials:
            if not os.path.exists(client_secrets_file):
                return None
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
            credentials = flow.run_local_server(port=0)
        with open(token_file, "wb") as token:
            pickle.dump(credentials, token)
    return build("youtube", "v3", credentials=credentials)

def upload_to_youtube(video_path, title, description, publish_at=None):
    youtube = get_youtube_service()
    if not youtube: return False
    body = {
        "snippet": {
            "title": f"{title} #Shorts"[:100],
            "description": f"{description}\n\n#Shorts #AIStories",
            "tags": ["Shorts", "AIStory"],
            "categoryId": "22"
        },
        "status": {"selfDeclaredMadeForKids": False}
    }
    if publish_at:
        body["status"]["privacyStatus"] = "private"
        body["status"]["publishAt"] = publish_at
    else:
        body["status"]["privacyStatus"] = "public"
    try:
        media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        response = None
        while response is None:
            status, response = request.next_chunk()
        return True
    except Exception:
        return False

# ==============================================================================
# PIPELINE EXECUTION ENGINE WITH TRACKING SUPPORT
# ==============================================================================

def background_generation_pipeline(job_id, numvids, gen_mode, choice_detail):
    print(f"[Thread] Starting pipeline job: {job_id} for {numvids} videos.")
    
    for i in range(numvids):
        base_pct = int((i / numvids) * 100)
        slice_width = 100 / numvids

        # 1. Generate Story & Title (Single AI Call)
        JOBS[job_id] = {
            "status": "processing",
            "message": f"Video {i+1}/{numvids}: Generating story & title using Gemini...",
            "progress": base_pct + int(slice_width * 0.05)
        }
        
        now_local = datetime.now()
        timestamp = now_local.strftime("%Y%m%d_%H%M%S")
        story_id = f"ai_story_{timestamp}_{i}"

        try:
            story_data = generate_story_and_title_gemini(mode=gen_mode, choice_detail=choice_detail)
            story_text = story_data["text"]
            genre = story_data["genre"]
            video_title = story_data["title"]
        except Exception as e:
            print(f"Failed to generate content: {e}")
            continue

        # 2. Text-to-Speech Generation
        JOBS[job_id] = {
            "status": "processing",
            "message": f"Video {i+1}/{numvids}: Generating speech audio files...",
            "progress": base_pct + int(slice_width * 0.20)
        }
        
        audio_output_path = os.path.join(stories_dir, f"{story_id}.mp3")
        try:
            generate_tts_audio_free(story_text, audio_output_path)
        except Exception as e:
            print(f"Failed to generate audio: {e}")
            continue

        # 3. Transcribe and Sync Subtitles
        JOBS[job_id] = {
            "status": "processing",
            "message": f"Video {i+1}/{numvids}: Transcribing and timing subtitles...",
            "progress": base_pct + int(slice_width * 0.40)
        }

        # 4. Compile Video
        video_output_path = None
        try:
            video_output_path = create_video(
                story_id=story_id, 
                genre=genre, 
                job_id=job_id, 
                video_index=i, 
                total_videos=numvids
            )
        except Exception as e:
            print(f"Failed to compile video assets: {e}")
            continue

        # 5. YouTube Upload
        JOBS[job_id] = {
            "status": "processing",
            "message": f"Video {i+1}/{numvids}: Uploading video to YouTube...",
            "progress": base_pct + int(slice_width * 0.90)
        }

        scheduled_time = datetime.now(timezone.utc) + timedelta(hours=i + 1)
        publish_at_str = scheduled_time.isoformat(timespec='seconds').replace("+00:00", "Z")

        upload_success = False
        if video_output_path and os.path.exists(video_output_path):
            try:
                upload_success = upload_to_youtube(
                    video_path=video_output_path,
                    title=video_title,
                    description=story_text,
                    publish_at=publish_at_str
                )
            except Exception as e:
                print(f"YouTube upload failed: {e}")
                upload_success = False

        # If YouTube upload fails, immediately halt execution and raise warning
        if not upload_success:
            print(f"[Pipeline] YouTube API rejected video {i+1}. Halting generation immediately.")
            JOBS[job_id] = {
                "status": "stopped_early",
                "message": f"Halted at Video {i+1}/{numvids} due to YouTube upload limits.",
                "progress": base_pct + int(slice_width * 0.90),
                "alert": f"The pipeline stopped early at Video {i+1} because YouTube rejected the upload. You may have run out of your daily YouTube API upload quota."
            }
            return

    # Pipeline Complete
    JOBS[job_id] = {
        "status": "completed",
        "message": "All videos have been rendered and processed successfully.",
        "progress": 100
    }
    print(f"[Thread] Job {job_id} pipeline completed.")


# ==============================================================================
# FLASK WEB APP ROUTING
# ==============================================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        numvids = int(request.form.get('numvids', 1))
    except ValueError:
        numvids = 1

    # Strict clamping to 10 maximum videos
    if numvids > 10:
        numvids = 10
    elif numvids < 1:
        numvids = 1

    mode_selection = request.form.get('mode_selection')
    gen_mode = "random"
    choice_detail = None

    if mode_selection == "2":
        gen_mode = "preset"
        genre_choice = request.form.get('genre_choice')
        choice_detail = "SCARY" if genre_choice == "1" else "OTHER"
    elif mode_selection == "3":
        gen_mode = "custom"
        choice_detail = request.form.get('custom_prompt', '').strip()
        if not choice_detail:
            gen_mode = "random"

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "starting",
        "message": "Initializing the video generation thread...",
        "progress": 0
    }

    thread = threading.Thread(
        target=background_generation_pipeline, 
        args=(job_id, numvids, gen_mode, choice_detail)
    )
    thread.start()

    return redirect(url_for('progress_page', job_id=job_id))

@app.route('/progress/<job_id>')
def progress_page(job_id):
    return render_template('progress.html', job_id=job_id)

@app.route('/api/status/<job_id>')
def api_status(job_id):
    info = JOBS.get(job_id, {"status": "not_found", "message": "Task context could not be located.", "progress": 0})
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)