import os
import random
import tempfile
import uuid
import asyncio
import pickle
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

# Google API Client packages for handling OAuth authentication and YouTube uploads.
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ==============================================================================
# ENVIRONMENT AND DIRECTORY INITIALIZATION
# ==============================================================================
# We establish absolute paths using the location of this script as the root anchor.
# This prevents relative path resolution issues across different operating systems.
root_dir = os.path.dirname(os.path.abspath(__file__))
background_footage_dir = os.path.join(root_dir, "background_footage")
stories_dir = os.path.join(root_dir, "stories")
finished_videos_dir = os.path.join(root_dir, "finished_videos")

# Ensure target directories exist before running file-writing procedures
os.makedirs(stories_dir, exist_ok=True)
os.makedirs(finished_videos_dir, exist_ok=True)

# Instantiate the Google GenAI client using the modern developer SDK interface.
# The Gemini-2.5-flash model is selected for its execution speed and low latency.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "your-api-key-here")
client_gemini = genai.Client(api_key=GEMINI_API_KEY)

# Initialize OpenAI's Whisper model locally. 
# The 'base' model represents an optimized trade-off between CPU/GPU memory footprint and word-error-rate (WER).
print("Loading Whisper Speech-to-Text Model...")
model = whisper.load_model("base")


# ==============================================================================
# CONTENT GENERATION MODULE (GEMINI API INTEGRATION)
# ==============================================================================

def generate_story_text_gemini(mode="random", choice_detail=None):
    """
    Interfaces with the Gemini-2.5-Flash LLM to construct a structured story narrative.
    
    Args:
        mode (str): Determines prompt construction strategy. Options: 'random', 'preset', or 'custom'.
        choice_detail (str): Contains selected preset genre or custom prompt text.
        
    Returns:
        dict: Parsed structure containing:
              - 'text' (str): The actual generated narrative.
              - 'genre' (str): The calculated genre tag ('scary' or 'other') for routing.
    """
    print(f"Generating story text using Gemini API (Mode: {mode})...")

    # Base rules shared across all execution paths to ensure consistent structure, length, and format.
    formatting_instructions = (
        "\n\nStrict Formatting & Structural Rules:\n"
        "- Target length: Under 110 words total (essential for a 45-second TikTok/Short video voiceover).\n"
        "- Start immediately with a gripping hook.\n"
        "- Do not include a title, introduction, or outro within the story text.\n"
        "- CLIFFHANGER RULE: Build up the tension and abruptly cut the story off at a highly suspenseful cliffhanger, ending "
        "exactly with: '...Like for Part 2.' If the story is simple or comedic, you may finish it naturally.\n\n"
        "CRITICAL: You must format your output exactly like this so the parser can process it:\n"
        "GENRE: [Write either SCARY or OTHER here]\n"
        "[STORY START]\n"
        "[Write the story text here]"
    )

    # Branch 1: Random Genre Selection
    if mode == "random":
        prompt = (
            "Write a highly engaging, first-person story. You must randomly choose between writing a "
            "SCARY story (like creepy r/nosleep horror stories) OR a dramatic/funny Reddit story.\n\n"
            "To ensure variety and prevent repetitive themes, randomly choose ONE of the "
            "following story archetypes/premises to write:\n"
            "1. THE SCARY / HORROR STORY (The Original Style - Tag as SCARY)\n"
            "2. THE DRAMATIC TWIST (Suspense / Mystery - Tag as OTHER)\n"
            "3. SHOCKING CONFESSIONS (Family / Relationship Drama - Tag as OTHER)\n"
            "4. TEEN RELATIONSHIP / HEARTBREAK CHAOS (Tag as OTHER)\n"
            "5. SATISFYING CLEVERNESS (School / Work Drama - Tag as OTHER)\n"
            "6. CHAOTIC COMEDY (Daily Life Mishaps - Tag as OTHER)\n"
        ) + formatting_instructions

    # Branch 2: Preset Genre Mode
    elif mode == "preset":
        if choice_detail.upper() == "SCARY":
            prompt = (
                "Write a highly engaging, suspenseful, and terrifying first-person scary story (similar to r/nosleep).\n"
                "Incorporate supernatural elements, skinwalkers, urban legends, eerie isolation, or unsettling coincidences.\n"
                "Ensure you Tag the GENRE as SCARY."
            ) + formatting_instructions
        else:
            prompt = (
                f"Write a highly engaging, first-person Reddit-style narrative based on the genre: {choice_detail}.\n"
                "Focus on high-drama, intense conversations, or chaotic comedy.\n"
                "Ensure you Tag the GENRE as OTHER."
            ) + formatting_instructions

    # Branch 3: Custom Niche Mode
    else:  
        prompt = (
            f"Write a highly engaging, first-person Reddit-style story based strictly on this user-provided niche or prompt: '{choice_detail}'.\n"
            "Tailor the emotional depth, vocabulary, and flow to match this theme.\n"
            "Determine if this story is 'SCARY' or 'OTHER' and mark the tag accordingly."
        ) + formatting_instructions

    # Network I/O request to Google GenAI API endpoint
    response = client_gemini.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    
    raw_text = response.text.strip()
    
    # State-machine parsing logic to separate the metadata (Genre) from actual story content.
    genre = "other"
    story_started = False
    story_lines = []
    
    lines = raw_text.split("\n")
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
            
    # Parsing Fallback implementation: If the API model drifted and skipped tags, extract raw text
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
    Summarizes generated story text into a high-CTR YouTube title using Gemini.
    """
    print("Generating optimized video title...")
    prompt = (
        f"Based on the following story, write a catchy, high-engagement title "
        f"under 50 characters for a YouTube Short. Match the emotional tone of the story.\n"
        f"Do not include quotes, hashtags, or bracketed descriptions:\n\n{story_text}"
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


# ==============================================================================
# AUDIO PROCESSING & TEXT-TO-SPEECH (TTS) MODULE
# ==============================================================================

async def save_edge_tts_audio(text, output_path):
    """
    Executes an asynchronous stream to communicate with Microsoft Edge TTS servers.
    We leverage 'edge-tts' for highly-tuned, human-sounding neural voices without premium fees.
    """
    voice = "en-US-ChristopherNeural"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def generate_tts_audio_free(text, output_path):
    """
    Wraps asynchronous co-routines inside a synchronous executor thread.
    This architecture isolates asynchronous networking tasks from synchronous media operations.
    """
    print("Generating neural voiceover...")
    asyncio.run(save_edge_tts_audio(text, output_path))
    print(f"Audio saved to: {output_path}")


# ==============================================================================
# DYNAMIC GRAPHICS & TRANSCRIPT SYNC MODULE
# ==============================================================================

def create_pil_subtitle_clip(text_str, font_name, font_size, color, stroke_color, stroke_width, temp_files):
    """
    Generates dynamic transparency layers containing formatted subtitle text using Pillow (PIL).
    
    This acts as an optimization wrapper bypassing MoviePy's native text rendering,
    which often relies on direct ImageMagick OS library dependencies that can be unstable.
    """
    try:
        font = ImageFont.truetype(font_name, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Determine structural boundaries (bounding box) of the text
    left, top, right, bottom = font.getbbox(text_str)
    text_w = right - left
    text_h = bottom - top

    # Padding buffers to prevent text clipping at boundaries
    pad_x = 30
    pad_y = 30

    img_w = int(text_w + pad_x * 2)
    img_h = int(text_h + pad_y * 2)

    # Instantiating a blank, transparent (RGBA) canvas
    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw_x = pad_x - left
    draw_y = pad_y - top

    # Drawing anti-aliased font strokes to maximize visibility over changing background footages
    draw.text((draw_x, draw_y), text_str, font=font, 
              fill=color, stroke_width=stroke_width, stroke_fill=stroke_color)

    # Persist the dynamic canvas to disk inside the system temporary directory
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"sub_{uuid.uuid4().hex}.png")
    img.save(temp_path)
    temp_files.append(temp_path)

    # Return a static ImageClip wrapped in MoviePy framework
    return ImageClip(temp_path)


def generate_subtitles(audio_path, video_size, temp_files):
    """
    Extracts word-level timestamps using Whisper and groups them into rhythmic blocks.
    
    Returns:
        list: Fully structured and timed MoviePy subtitle clips.
    """
    print(f"Transcribing audio with Whisper: {audio_path}")
    result = model.transcribe(audio_path, word_timestamps=True, fp16=False)
    
    subtitle_clips = []
    w, h = video_size

    all_words = []
    for segment in result['segments']:
        if 'words' in segment:
            all_words.extend(segment['words'])

    chunks = []
    chunk_size = 3  # Chunking parameter: Limit words-per-screen to match short-form retention speeds.

    if all_words:
        # Construct text structures based on highly accurate token timestamps
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
        # Fallback segmentation parser if word-level generation fails
        print("Falling back to segment-level timeline generation...")
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

    for chunk in chunks:
        text_str = chunk['text']
        start_time = chunk['start']
        end_time = chunk['end']
        duration = max(0.1, end_time - start_time)

        try:
            # Instantiate PIL Graphic Subtitle Clip
            txt_clip = create_pil_subtitle_clip(
                text_str=text_str,
                font_name=font_name,
                font_size=font_size,
                color='yellow',
                stroke_color='black',
                stroke_width=2,
                temp_files=temp_files
            )
            
            # Subtitle Scaling Constraint: Prevent text overflowing past 80% of video width.
            safe_width = int(w * 0.8)
            if txt_clip.w > safe_width:
                txt_clip = txt_clip.resized(width=safe_width)

            # Center position dynamically calibrated lower-third (65% coordinate depth)
            vertical_position = int(h * 0.65)
            if vertical_position + txt_clip.h >= h:
                vertical_position = int(h - txt_clip.h - 20)

            # Assign timeline parameters
            txt_clip = (txt_clip.with_start(start_time)
                                .with_duration(duration)
                                .with_position(('center', vertical_position)))

            subtitle_clips.append(txt_clip)
        except Exception as e:
            print(f"Error creating Subtitle for text '{text_str}': {e}")
    
    return subtitle_clips


# ==============================================================================
# AUDIO-VISUAL COMPOSITION ENGINE (MOVIEPY PIPELINE)
# ==============================================================================

def adjust_background_duration(background_path, target_duration):
    """
    Dynamically loops background videos to match the variable target voiceover length.
    """
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
    Assembles local multimedia clips, subtitles, and music overlays into a single render tree.
    """
    print(f"Assembling video track for story ID: {story_id} (Genre: {genre})")
    
    audio_path = os.path.join(stories_dir, f"{story_id}.mp3")
    background_footage_files = [f for f in os.listdir(background_footage_dir) if f.endswith('.mp4')]
    
    if not background_footage_files:
        print("No background footage files found.")
        return None
        
    background_file = random.choice(background_footage_files)
    background_path = os.path.join(background_footage_dir, background_file)
    
    # Load primary TTS generation audio track
    tts_audio = AudioFileClip(audio_path)
    adjusted_background = adjust_background_duration(background_path, tts_audio.duration)
    
    # Track array to automate temporary asset deletion later
    temp_files = []
    subtitle_clips = generate_subtitles(audio_path, adjusted_background.size, temp_files)

    max_duration = min(adjusted_background.duration, tts_audio.duration)

    # Frame clipping validation
    valid_subtitle_clips = []
    for clip in subtitle_clips:
        if clip.start >= max_duration:
            continue  
        if clip.start + clip.duration > max_duration:
            clip = clip.with_duration(max_duration - clip.start)
        valid_subtitle_clips.append(clip)

    # Merge background video layers with the dynamic subtitle array
    if not valid_subtitle_clips:
        final_video = adjusted_background
    else:
        final_video = CompositeVideoClip([adjusted_background] + valid_subtitle_clips, use_bgclip=True)
        final_video = final_video.with_duration(max_duration)
        
    # --- DYNAMIC AUDIO MIXING ---
    # Background suspense tracks are isolated strictly for 'scary' story files to prevent sound mismatch.
    music_path = os.path.join(root_dir, "background_music.mp3") 
    bg_music = None
    
    if genre == "scary" and os.path.exists(music_path):
        print("Scary story detected. Initiating background music mix...")
        try:
            bg_music = AudioFileClip(music_path).with_duration(max_duration)
            bg_music = bg_music.with_volume_scaled(0.12)  # High voice priority over music
            combined_audio = CompositeAudioClip([tts_audio, bg_music])
        except Exception as e:
            print(f"Failed to load background music: {e}. Defaulting to voiceover.")
            combined_audio = tts_audio
    else:
        print("Standard genre detected. Rendering with primary voiceover track only.")
        combined_audio = tts_audio

    final_video = final_video.with_audio(combined_audio)

    output_path = os.path.join(finished_videos_dir, f'{story_id}.mp4')
    print(f"Compiling and writing output file: {output_path}")
    
    # Render with libx264 (H.264 video codec) and aac audio codec.
    # Uses 'ultrafast' processing preset to minimize CPU render duration constraints.
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24, preset='ultrafast')

    # De-allocate local memory file pointers to prevent operating system lock errors.
    final_video.close()
    adjusted_background.close()
    tts_audio.close()
    if bg_music is not None:
        bg_music.close()

    # Clear disk space by removing the dynamic subtitle canvas PNGs.
    print("Clearing temporary subtitle assets from system disk...")
    for temp_path in temp_files:
        try:
            os.remove(temp_path)
        except Exception as e:
            print(f"Failed to remove temp file {temp_path}: {e}")

    return output_path


# ==============================================================================
# YOUTUBE UPLOAD MODULE (OAUTH2 & GOOGLE API CLIENT)
# ==============================================================================

def get_youtube_service():
    """
    Initializes OAuth2 flow using local system file paths.
    Uses 'token.pickle' to cache verified user session state, allowing fully headless execution
    on consecutive runs once the initial local browser verification finishes.
    """
    scopes = ["https://www.googleapis.com/auth/youtube.upload"]
    client_secrets_file = os.path.join(root_dir, "client_secret.json")
    token_file = os.path.join(root_dir, "token.pickle")
    
    credentials = None
    
    # Check if a validated session has already been saved
    if os.path.exists(token_file):
        with open(token_file, "rb") as token:
            credentials = pickle.load(token)
            
    # Resolve token invalidation or expiration
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
            except Exception:
                credentials = None
                
        if not credentials:
            if not os.path.exists(client_secrets_file):
                print(f"\n[Warning] '{client_secrets_file}' not found.")
                print("Save your credentials JSON file from Google Cloud Console into this directory.")
                return None
            
            # Spawns a local host listener to accept Google's redirect authorization callback code
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
            credentials = flow.run_local_server(port=0)
            
        # Serialize existing authorization token states to local storage
        with open(token_file, "wb") as token:
            pickle.dump(credentials, token)
            
    return build("youtube", "v3", credentials=credentials)


def upload_to_youtube(video_path, title, description, publish_at=None):
    """
    Uploads the output video to the authorized YouTube Channel.
    If 'publish_at' time parameters are sent, the video is set to 'private' and scheduled.
    """
    youtube = get_youtube_service()
    if not youtube:
        print("Skipping YouTube upload step...")
        return False

    upload_title = f"{title} #Shorts"
    upload_description = f"{description}\n\n#Shorts #AIStories #ShortsVideo"

    body = {
        "snippet": {
            "title": upload_title[:100],
            "description": upload_description,
            "tags": ["Shorts", "AIStory", "Fascinating"],
            "categoryId": "22" # Category 22 correlates with 'People & Blogs'
        },
        "status": {
            "selfDeclaredMadeForKids": False
        }
    }

    # Integrate publication queues if requested
    if publish_at:
        body["status"]["privacyStatus"] = "private"
        body["status"]["publishAt"] = publish_at
        print(f"Enqueuing video schedule for (UTC Time): {publish_at}")
    else:
        body["status"]["privacyStatus"] = "public"
        print("Publishing video immediately...")

    try:
        # Initializing chunked media stream connection parameters
        media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media
        )
        
        print(f"Streaming data packet '{video_path}' to Google Server APIs...")
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"Upload execution progress: {int(status.progress() * 100)}%")
                
        print(f"Success! Remote execution finished. Video ID: {response.get('id')}")
        return True
    except Exception as e:
        print(f"An error occurred during API communication: {e}")
        return False


# ==============================================================================
# PIPELINE CONTROL FLOW & INTERFACE SYSTEM
# ==============================================================================

def main():
    """
    Initializes user configuration prompts, builds dynamic generation parameters,
    and coordinates the media processing, transcription, and distribution loops.
    """
    if GEMINI_API_KEY == "your-api-key-here" or not GEMINI_API_KEY:
        print("Please configure your GEMINI_API_KEY at the top of the file or set it as an environment variable.")
        return

    # Prompt step 1: Set output count
    try:
        numvids = int(input("Enter how many videos you would like to make and schedule: "))
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return

    # Prompt step 2: Interrogate user for content generation path parameters
    print("\nSelect Story Customization Mode:")
    print("1. Randomize (Selects randomly from preset horror, drama, or comedy genres)")
    print("2. Specific Preset Genre (Select one fixed genre for all videos)")
    print("3. Custom Niche (Manually enter a custom topic/prompt)")
    mode_selection = input("Choose option (1, 2, or 3): ").strip()

    gen_mode = "random"
    choice_detail = None

    if mode_selection == "2":
        gen_mode = "preset"
        print("\nSelect Predefined Genre:")
        print("1. Scary / Horror (With background music mix)")
        print("2. Comedy / Bizarre Reddit Drama (Without music mix)")
        genre_choice = input("Choose option (1 or 2): ").strip()
        choice_detail = "SCARY" if genre_choice == "1" else "OTHER"
        
    elif mode_selection == "3":
        gen_mode = "custom"
        choice_detail = input("\nType your custom niche/story prompt (e.g., 'A software developer who discovers hidden code'): ").strip()
        if not choice_detail:
            print("No custom prompt received. Falling back to Randomize mode.")
            gen_mode = "random"

    # Execute production queue based on selected parameters
    for i in range(numvids):
        print(f"\n================ Processing Video {i+1} of {numvids} ================")
        
        now_local = datetime.now()
        timestamp = now_local.strftime("%Y%m%d_%H%M%S")
        story_id = f"ai_story_{timestamp}_{i}"

        # Step 1: Query LLM Content Engine using established selection configurations
        try:
            story_data = generate_story_text_gemini(mode=gen_mode, choice_detail=choice_detail)
            story_text = story_data["text"]
            genre = story_data["genre"]
        except Exception as e:
            print(f"Failed to generate story with Gemini: {e}")
            continue

        # Step 2: Query title optimization engine
        video_title = generate_title_gemini(story_text)

        # Step 3: Run cloud-neural audio generation pipelines
        audio_output_path = os.path.join(stories_dir, f"{story_id}.mp3")
        try:
            generate_tts_audio_free(story_text, audio_output_path)
        except Exception as e:
            print(f"Failed to generate Edge-TTS audio: {e}")
            continue

        # Step 4: Render composite media assets into H.264 MP4 Container
        video_output_path = None
        try:
            video_output_path = create_video(story_id, genre)
        except Exception as e:
            print(f"An error occurred while generating video assets: {e}")
            continue

        # Step 5: Establish incremental queue schedules
        # Example schedule layout: First video launches in 1 hour, next launches in 2 hours, etc.
        scheduled_time = datetime.now(timezone.utc) + timedelta(hours=i + 1)
        publish_at_str = scheduled_time.isoformat(timespec='seconds').replace("+00:00", "Z")

        # Step 6: Dispatch finished video structures to YouTube API endpoints
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