import os
import random
import tempfile
import uuid
import asyncio
import pickle
import whisper
from datetime import datetime, timedelta, timezone  # Added for accurate scheduling
from google import genai  # Modern, officially supported Google GenAI library
import edge_tts  # Free neural TTS library
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    VideoFileClip, 
    AudioFileClip, 
    CompositeVideoClip, 
    ImageClip, 
    CompositeAudioClip,
    concatenate_videoclips
)

# Google API client imports for YouTube Upload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Set up file paths
root_dir = os.path.dirname(os.path.abspath(__file__))
background_footage_dir = os.path.join(root_dir, "background_footage")
stories_dir = os.path.join(root_dir, "stories")
finished_videos_dir = os.path.join(root_dir, "finished_videos")

os.makedirs(stories_dir, exist_ok=True)
os.makedirs(finished_videos_dir, exist_ok=True)

# Configure the New Google GenAI Client
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6Iqx5utr6Umbd7TLnaU3VvTPr8eqRwC85fTSRDb4RIInQ")
client_gemini = genai.Client(api_key=GEMINI_API_KEY)

# Load Whisper model (base is fast, medium is more accurate)
model = whisper.load_model("base")


def generate_story_text_gemini():
    """
    Generates a unique first-person story. Randomly selects between writing a 
    SCARY story or another genre (comedy/drama) from Reddit-style examples.
    Returns a dictionary containing the parsed 'text' and 'genre'.
    """
    print("Generating a new Reddit-style story with Google Gemini...")
    
    prompt = (
        "Write a highly engaging, first-person story. You must randomly choose between writing a "
        "SCARY story (like creepy r/nosleep horror stories) OR a dramatic/funny Reddit story "
        "(like r/story, r/RedditStoryTime, or r/funnystories).\n\n"
        "To ensure variety and prevent repetitive themes, randomly choose ONE of the "
        "following story archetypes/premises to write:\n\n"
        "1. THE SCARY / HORROR STORY (The Original Style - Tag as SCARY):\n"
        "   - Inspired by: Suspenseful, scary, or supernatural encounters (e.g., skinwalkers, weird forest occurrences, "
        "     creepy figures, glitches in reality, or disturbing stalkers).\n\n"
        "2. THE DRAMATIC TWIST (Suspense / Mystery - Tag as OTHER):\n"
        "   - Inspired by: 'I thought my wife was cheating on me. It's worse.' or deathbed confessions.\n"
        "   - Premise: The narrator suspects something standard, but following the clue leads to a shocking, "
        "     dramatic discovery.\n\n"
        "3. SHOCKING CONFESSIONS (Family / Relationship Drama - Tag as OTHER):\n"
        "   - Inspired by: Catching a teenage sister smoking or doing something highly unexpected, "
        "     or finding a hidden relative's secret.\n\n"
        "4. TEEN RELATIONSHIP / HEARTBREAK CHAOS (Tag as OTHER):\n"
        "   - Inspired by: Messy, chaotic, or unforgettable high school breakups.\n\n"
        "5. SATISFYING CLEVERNESS (School / Work Drama - Tag as OTHER):\n"
        "   - Inspired by: Outsmarting a power-tripping teacher or boss.\n\n"
        "6. CHAOTIC COMEDY (Daily Life Mishaps - Tag as OTHER):\n"
        "   - Inspired by: Awkward medical appointments, flatulence mishaps, or bizarre misunderstandings "
        "     (like mistaking a draft stopper or vacuum hose for a dangerous snake in a cupboard).\n\n"
        "Strict Formatting & Structural Rules:\n"
        "- Target length: Under 110 words total (essential for a 45-second TikTok/Short video voiceover).\n"
        "- Start immediately with a gripping hook (e.g., 'My wife has been distant recently...', "
        "  'On her deathbed, my wife confessed to something...', 'I caught my teenage sister doing the unthinkable...').\n"
        "- Do not include a title, introduction, or outro within the story text.\n"
        "- CLIFFHANGER RULE: If the story is dramatic, scary, or complex, do not rush the resolution. "
        "  Build up the tension and abruptly cut the story off at a highly suspenseful cliffhanger, ending "
        "  exactly with: '...Like for Part 2.' If the story is simple or comedic, you may finish it naturally.\n\n"
        "CRITICAL: You must format your output exactly like this so the script can parse it:\n"
        "GENRE: [Write either SCARY or OTHER here]\n"
        "[STORY START]\n"
        "[Write the story text here]"
    )
    
    response = client_gemini.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    
    raw_text = response.text.strip()
    
    genre = "other"
    story_text = ""
    
    lines = raw_text.split("\n")
    story_started = False
    story_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("GENRE:"):
            g_val = stripped.split(":", 1)[1].strip().upper()
            if "SCARY" in g_val:
                genre = "scary"
            else:
                genre = "other"
        elif stripped == "[STORY START]":
            story_started = True
        elif story_started:
            story_lines.append(line)
            
    # Fallback if the parser fails to find [STORY START]
    if not story_lines:
        cleaned_lines = [l for l in lines if not l.upper().startswith("GENRE:") and l.strip() != "[STORY START]"]
        story_text = "\n".join(cleaned_lines).strip()
    else:
        story_text = "\n".join(story_lines).strip()
        
    print(f"\n--- Parsed Genre: {genre.upper()} ---")
    print(f"--- Generated Story Text ---\n{story_text}\n----------------------------\n")
    
    return {"text": story_text, "genre": genre}


def generate_title_gemini(story_text):
    """
    Asks Gemini to generate a short, engaging Reddit-style title suitable for a YouTube Short.
    """
    print("Generating a title with Google Gemini...")
    prompt = (
        f"Based on the following story, write a catchy, high-engagement title "
        f"under 50 characters for a YouTube Short. Match the emotional tone of the story.\n\n"
        f"- For dramatic or suspenseful stories, use intense, mystery-driven hooks (e.g., 'My wife's final confession', "
        f"  'I thought she was cheating...', 'My sister's secret life').\n"
        f"- For comedic or school stories, use relatable Reddit-style phrasing.\n"
        f"Do not include quotes or hashtags in the title:\n\n{story_text}"
    )
    try:
        response = client_gemini.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        print(f"Failed to generate title: {e}")
        return "Unbelievable Reddit Story"


async def save_edge_tts_audio(text, output_path):
    """
    Asynchronous helper to stream and save the Microsoft Edge TTS voice.
    """
    voice = "en-US-ChristopherNeural"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def generate_tts_audio_free(text, output_path):
    """
    Generates a free high-quality voiceover using Microsoft Edge TTS.
    """
    print("Generating free neural voiceover with Edge-TTS...")
    asyncio.run(save_edge_tts_audio(text, output_path))
    print(f"Audio saved to: {output_path}")


def create_pil_subtitle_clip(text_str, font_name, font_size, color, stroke_color, stroke_width, temp_files):
    try:
        font = ImageFont.truetype(font_name, font_size)
    except Exception:
        font = ImageFont.load_default()

    left, top, right, bottom = font.getbbox(text_str)
    text_w = right - left
    text_h = bottom - top

    pad_x = 30
    pad_y = 30

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
    print(f"Transcribing audio: {audio_path}")
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
            
            chunks.append({
                'text': text_str,
                'start': start_time,
                'end': end_time
            })
    else:
        print("Using segment fallback splitting...")
        for segment in result['segments']:
            words = segment['text'].strip().split()
            if not words:
                continue
            
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
                
                chunks.append({
                    'text': text_str,
                    'start': start_time,
                    'end': end_time
                })

    font_name = "impact.ttf" 
    font_size = 80

    try:
        pil_font = ImageFont.truetype(font_name, font_size)
    except Exception:
        pil_font = None

    for chunk in chunks:
        text_str = chunk['text']
        start_time = chunk['start']
        end_time = chunk['end']
        duration = max(0.1, end_time - start_time)

        try:
            txt_clip = create_pil_subtitle_clip(
                text_str=text_str,
                font_name=font_name,
                font_size=font_size,
                color='yellow',
                stroke_color='black',
                stroke_width=2,
                temp_files=temp_files
            )
            
            safe_width = int(w * 0.8)
            if txt_clip.w > safe_width:
                txt_clip = txt_clip.resized(width=safe_width)

            vertical_position = int(h * 0.65)
            if vertical_position + txt_clip.h >= h:
                vertical_position = int(h - txt_clip.h - 20)

            txt_clip = (txt_clip.with_start(start_time)
                                .with_duration(duration)
                                .with_position(('center', vertical_position)))

            subtitle_clips.append(txt_clip)
        except Exception as e:
            print(f"Error creating Subtitle for text '{text_str}': {e}")
    
    return subtitle_clips


def adjust_background_duration(background_path, target_duration):
    temp_clip = VideoFileClip(background_path)
    base_duration = temp_clip.duration
    temp_clip.close()

    if base_duration < target_duration:
        n_loops = int(target_duration // base_duration) + 1
        print(f"Looping background video {n_loops} times to cover {target_duration:.2f}s")
        
        clips = [VideoFileClip(background_path) for _ in range(n_loops)]
        looped_video = concatenate_videoclips(clips, method='compose')
        return looped_video.subclipped(0, target_duration)
    else:
        return VideoFileClip(background_path).subclipped(0, target_duration)


def create_video(story_id, genre):
    """
    Renders the final video. Only applies background music if the genre is 'scary'.
    """
    print(f"Creating video for story ID: {story_id} (Genre: {genre})")
    
    audio_path = os.path.join(stories_dir, f"{story_id}.mp3")
    background_footage_files = [f for f in os.listdir(background_footage_dir) if f.endswith('.mp4')]
    
    if not background_footage_files:
        print("No background footage files found.")
        return None
        
    background_file = random.choice(background_footage_files)
    background_path = os.path.join(background_footage_dir, background_file)
    
    # Load the TTS audio track
    tts_audio = AudioFileClip(audio_path)
    adjusted_background = adjust_background_duration(background_path, tts_audio.duration)
    
    temp_files = []
    subtitle_clips = generate_subtitles(audio_path, adjusted_background.size, temp_files)

    max_duration = min(adjusted_background.duration, tts_audio.duration)

    valid_subtitle_clips = []
    for clip in subtitle_clips:
        if clip.start >= max_duration:
            continue  
        if clip.start + clip.duration > max_duration:
            clip = clip.with_duration(max_duration - clip.start)
        valid_subtitle_clips.append(clip)

    if not valid_subtitle_clips:
        final_video = adjusted_background
    else:
        final_video = CompositeVideoClip([adjusted_background] + valid_subtitle_clips, use_bgclip=True)
        final_video = final_video.with_duration(max_duration)
        
    # --- AUDIO MIXING SECTION ---
    # Background music is applied strictly to SCARY stories
    music_path = os.path.join(root_dir, "background_music.mp3") 
    bg_music = None
    
    if genre == "scary" and os.path.exists(music_path):
        print("Scary story detected. Mixing background music...")
        try:
            bg_music = AudioFileClip(music_path).with_duration(max_duration)
            bg_music = bg_music.with_volume_scaled(0.12) 
            combined_audio = CompositeAudioClip([tts_audio, bg_music])
        except Exception as e:
            print(f"Failed to load background music: {e}. Using voiceover only.")
            combined_audio = tts_audio
    else:
        print("Genre is not scary (or background music missing). Using voiceover only.")
        combined_audio = tts_audio

    final_video = final_video.with_audio(combined_audio)
    # ----------------------------

    output_path = os.path.join(finished_videos_dir, f'{story_id}.mp4')

    print(f"Writing video to: {output_path}")
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24, preset='ultrafast')

    # Clean up resources
    final_video.close()
    adjusted_background.close()
    tts_audio.close()
    if bg_music is not None:
        bg_music.close()

    print("Cleaning up temporary subtitle assets...")
    for temp_path in temp_files:
        try:
            os.remove(temp_path)
        except Exception as e:
            print(f"Failed to remove temp file {temp_path}: {e}")

    return output_path


# ==================== YOUTUBE UPLOAD MODULE ====================

def get_youtube_service():
    """
    Authenticates the user and builds the YouTube API client service.
    """
    scopes = ["https://www.googleapis.com/auth/youtube.upload"]
    client_secrets_file = os.path.join(root_dir, "client_secret.json")
    token_file = os.path.join(root_dir, "token.pickle")
    
    credentials = None
    
    # Load existing authentication token if it exists
    if os.path.exists(token_file):
        with open(token_file, "rb") as token:
            credentials = pickle.load(token)
            
    # If credentials do not exist or are invalid, run the OAuth login flow
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
            except Exception:
                credentials = None
                
        if not credentials:
            if not os.path.exists(client_secrets_file):
                print(f"\n[Warning] '{client_secrets_file}' not found.")
                print("Please download your credentials JSON file from Google Cloud Console and save it here.")
                return None
            
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
            credentials = flow.run_local_server(port=0)
            
        # Save the credentials for future runs
        with open(token_file, "wb") as token:
            pickle.dump(credentials, token)
            
    return build("youtube", "v3", credentials=credentials)


def upload_to_youtube(video_path, title, description, publish_at=None):
    """
    Uploads the specified video path to YouTube.
    If 'publish_at' is provided, the video is scheduled as 'private' first,
    then automatically publicizes at the scheduled UTC ISO datetime string.
    """
    youtube = get_youtube_service()
    if not youtube:
        print("Skipping YouTube upload due to missing authorization.")
        return False

    upload_title = f"{title} #Shorts"
    upload_description = f"{description}\n\n#Shorts #AIStories #ShortsVideo"

    # Set up basic upload body
    body = {
        "snippet": {
            "title": upload_title[:100],
            "description": upload_description,
            "tags": ["Shorts", "AIStory", "Fascinating"],
            "categoryId": "22"
        },
        "status": {
            "selfDeclaredMadeForKids": False
        }
    }

    # Handle scheduling configurations
    if publish_at:
        body["status"]["privacyStatus"] = "private"
        body["status"]["publishAt"] = publish_at
        print(f"Scheduling video upload for (UTC): {publish_at}")
    else:
        body["status"]["privacyStatus"] = "public"
        print("Uploading video for immediate public release...")

    try:
        media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media
        )
        
        print(f"Uploading '{video_path}' to YouTube...")
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"Upload progress: {int(status.progress() * 100)}%")
                
        print(f"Success! Video uploaded. Video ID: {response.get('id')}")
        return True
    except Exception as e:
        print(f"An error occurred during upload: {e}")
        return False

# ===============================================================


def main():
    if GEMINI_API_KEY == "your-gemini-api-key-here" or not GEMINI_API_KEY:
        print("Please configure your GEMINI_API_KEY at the top of the file or set it as an environment variable.")
        return

    try:
        numvids = int(input("Enter how many videos you would like to make and schedule: "))
    except ValueError:
        print("Enter a valid integer number")
        return

    for i in range(numvids):
        print(f"\n================ Processing Video {i+1} of {numvids} ================")
        
        # Setup local generation timestamp for files
        now_local = datetime.now()
        timestamp = now_local.strftime("%Y%m%d_%H%M%S")
        story_id = f"ai_story_{timestamp}_{i}"

        # 1. Generate story text and capture genre tag
        try:
            story_data = generate_story_text_gemini()
            story_text = story_data["text"]
            genre = story_data["genre"]
        except Exception as e:
            print(f"Failed to generate story with Gemini: {e}")
            continue

        # 2. Generate a structured title for YouTube
        video_title = generate_title_gemini(story_text)

        # 3. Convert text to Edge-TTS voiceover
        audio_output_path = os.path.join(stories_dir, f"{story_id}.mp3")
        try:
            generate_tts_audio_free(story_text, audio_output_path)
        except Exception as e:
            print(f"Failed to generate Edge-TTS audio: {e}")
            continue

        # 4. Render the video with custom audio configurations depending on the genre
        video_output_path = None
        try:
            video_output_path = create_video(story_id, genre)
        except Exception as e:
            print(f"An error occurred while generating video assets: {e}")
            continue

        # 5. Calculate future publication date (Incremental hour delays)
        # Video 1: 1 hour from now, Video 2: 2 hours from now, etc.
        scheduled_time = datetime.now(timezone.utc) + timedelta(hours=i + 1)
        publish_at_str = scheduled_time.isoformat(timespec='seconds').replace("+00:00", "Z")

        # 6. Upload to YouTube with Schedule
        if video_output_path and os.path.exists(video_output_path):
            try:
                upload_to_youtube(
                    video_path=video_output_path,
                    title=video_title,
                    description=story_text,
                    publish_at=publish_at_str
                )
            except Exception as e:
                print(f"YouTube upload failed: {e}")


if __name__ == "__main__":
    main()