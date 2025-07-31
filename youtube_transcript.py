import os
from typing import Optional
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from urllib.parse import urlparse, parse_qs
from supadata import Supadata

def is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube video URL"""
    domain = urlparse(url).netloc
    return domain in ('youtube.com', 'www.youtube.com', 'youtu.be')

def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL"""
    parsed = urlparse(url)
    if parsed.netloc == 'youtu.be':
        return parsed.path[1:]
    if parsed.netloc in ('youtube.com', 'www.youtube.com'):
        if parsed.path == '/watch':
            return parse_qs(parsed.query).get('v', [None])[0]
        if parsed.path.startswith('/embed/'):
            return parsed.path.split('/')[2]
    return None

def get_transcript_text(url: str) -> str:
    """
    Get transcript text from YouTube video URL
    Returns empty string if no transcript available
    """
    if not is_youtube_url(url):
        return ""
        
    video_id = extract_video_id(url)
    if not video_id:
        return ""
        
    try:

        supadata = Supadata(api_key=os.getenv('SUPADATA_API_KEY'))
        text_transcript = supadata.youtube.transcript(
            video_id=video_id,
            text=True # Set to False to get the transcript with timestamps
        )
        return text_transcript.content
    except (TranscriptsDisabled, NoTranscriptFound):
        return ""
    except Exception as e:
        print(f"Error getting transcript for {url}: {e}")
        return ""