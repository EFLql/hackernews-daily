from youtube_transcript import is_youtube_url, extract_video_id, get_transcript_text

def test_transcript_extraction():
    """Test YouTube transcript extraction with various URLs"""
    test_urls = [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", True, "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/3xQrXsWy2eI", True, "3xQrXsWy2eI"),
        ("https://youtu.be/shortcode", True, "shortcode"),
        ("https://www.youtube.com/embed/embedded", True, "embedded"),
        ("https://youtube.com/watch?other=param&v=vid123", True, "vid123")
    ]

    print("{:<60} {:<10} {:<20} {}".format(
        "URL", "Is YouTube", "Extracted ID", "Transcript Length"
    ))
    print("="*100)

    for url, expected_is_yt, expected_id in test_urls:
        is_yt = is_youtube_url(url)
        vid_id = extract_video_id(url)
        transcript = get_transcript_text(url)
        
        print("{:<60} {:<10} {:<20} {}, {}".format(
            url,
            "✓" if is_yt == expected_is_yt else f"✗ (expected {expected_is_yt})",
            vid_id if vid_id == expected_id else f"✗ (expected {expected_id})",
            len(transcript),
            transcript if transcript else "No transcript available"
        ))

if __name__ == '__main__':
    test_transcript_extraction()