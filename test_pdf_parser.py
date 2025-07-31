from pdf_parser import extract_text_from_pdf_url

def test_pdf_parser():
    """Test PDF extraction with sample URLs"""
    test_cases = [
        ("https://example.com/sample.pdf", False),  # Not a real PDF
        ("https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf", True),  # Small test PDF
        ("https://ml-site.cdn-apple.com/papers/the-illusion-of-thinking.pdf", True) #large pdf
    ]
    
    print("{:<60} {:<15} {} {:<60}".format(
        "URL", "Status", "Text Length (First+Last 10 pages)", "Sample Text (First 1000 chars)"
    ))
    print("="*300)

    for url, should_pass in test_cases:
        text = extract_text_from_pdf_url(url)
        status = "✓" if (bool(text) == should_pass) else "✗"
        print("{:<60} {:<15} {} {:<60}".format(
            url,
            status,
            len(text),
            text[:1000] if text else "No text extracted"
        ))

if __name__ == '__main__':
    test_pdf_parser()