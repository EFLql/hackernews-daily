import time
import openai
#import pandas as pd
import requests
from datetime import datetime, timedelta
import pytz
import os
import joblib
import torch
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sklearn.decomposition import PCA
from fusion_network import FeatureFusionNetwork
from bs4 import BeautifulSoup
from youtube_transcript import get_transcript_text
from pdf_parser import extract_text_from_pdf_url
from request_utils import fetch_url
import supabase

from dotenv import load_dotenv
# 加载 .env 文件中的环境变量
load_dotenv()

def clean_post_content(url):
    # Add YouTube check at start
    if "youtube.com" in url or "youtu.be" in url:
        if transcript := get_transcript_text(url):
            return transcript, url
        print(f"No transcript available for YouTube video: {url}")
        return "", url
    """Fetch and clean post content from URL"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36'
    }

    # Check for PDF
    if url.lower().endswith('.pdf'):
        if pdf_text := extract_text_from_pdf_url(url):
            return pdf_text, url
        print(f"No text extracted from PDF: {url}")
        return "", url
    
    def process_page(response):
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
            
        # Get clean text
        cleaned_text = ' '.join(soup.stripped_strings)
        # Consider it a good page if we have at least 100 characters of text
        return cleaned_text if len(cleaned_text) >= 100 else None
    
    try:
        response = fetch_url(url, timeout=10, headers=headers)
        response.raise_for_status()
        
        if (result := process_page(response)):
            return result, url
            
        print(f"Page has insufficient content, trying Wayback Machine for {url}")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    
    # Try Wayback Machine if first attempt failed
    try:
        wayback_url = f"https://web.archive.org/{url}"
        response = fetch_url(wayback_url, headers=headers, timeout=10)
        response.raise_for_status()
        if (result := process_page(response)):
            return result, wayback_url
    except Exception as e:
        print(f"Error fetching {url} from Wayback Machine: {e}")
    
    return "", url
    
class FeatureExtractor:
    def __init__(self):
        self.pca_models = {
            'type': joblib.load('type_pca.pkl'),
            'content': joblib.load('content_pca.pkl'),  
            'keyword': joblib.load('keyword_pca.pkl')
        }
        self.scaler = joblib.load('scaler.pkl')  # Add scaler
        self.emb_model = BGEM3FlagModel(os.environ.get('BGE_M3_PATH', 'BAAI/bge-m3'), use_fp16=True, 
                                        devices = 'cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def extract_numerical(self, score, comments):
        log_features = np.array([[np.log1p(score), np.log1p(comments), 0, 0]])
        #features_df = pd.DataFrame(log_features, columns=['score', 'descendants', 'word_count', 'author_avg_score'])  # 匹配训练时的列名
        return self.scaler.transform(log_features).flatten()[:2]
    
    def extract_embeddings(self, text, component):
        if not text:
            return np.zeros(64 if component != 'keyword' else 128)
        emb = self.emb_model.encode(text, batch_size=1)['dense_vecs']
        return self.pca_models[component].transform(emb.reshape(1, -1)).flatten()

class SummaryGenerator:
    def __init__(self):
        self.cache = {}
        openai.base_url = os.environ.get('OPENAI_BASE_URL', "https://openrouter.ai/api/v1")
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = openai.OpenAI(api_key=openai.api_key, base_url=openai.base_url)
        
    def _parse_summary(self, text):
        parts = {
            'type': None,
            'content': None, 
            'keywords': None
        }
        for line in text.split('\n'):
            if line.startswith('文章类型：'):
                parts['type'] = line.replace('文章类型：', '').strip()
            elif line.startswith('文章大体内容：'):
                parts['content'] = line.replace('文章大体内容：', '').strip()
            elif line.startswith('关键词：'):
                parts['keywords'] = line.replace('关键词：', '').strip()
        return parts
        
    def _get_summary(self, text, max_retries=3):
        # Choose model based on text length
        model_name = ("google/gemini-2.0-flash-exp:free" 
                     if len(text.split()) > 130000 
                     else "nousresearch/deephermes-3-llama-3-8b-preview:free")
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    #model="nousresearch/deephermes-3-llama-3-8b-preview:free",
                    #model="google/gemini-2.0-flash-exp:free",
                    messages=[
                        {"role": "system", "content": '''你是一个文章类型判断，关键词提取专家，帮助用户提取文章得以下内容：
    文章类型：xxx
    文章大体内容：一句话描述
    关键词：'''},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=1024,
                    temperature=0.5
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(3)
    
    def generate(self, text, max_retries=3):
        parts={
            'type': "无法提取类型",
            'content': "无法提取内容",
            'keywords': "无法提取关键词" 
        }
        for attempt in range(max_retries):
            try:
                summary = self._get_summary(text)
                parts = self._parse_summary(summary)
                
                if all(parts.values()):  # All fields present
                    return parts
                    
                print(f"Missing summary components, retrying... (attempt {attempt + 1})")
            except Exception as e:
                print(f"Summary failed: {e}")
                time.sleep(3)
                
        return parts

class HNPostProcessor:
    def __init__(self, feature_extractor, summary_generator):
        self.fe = feature_extractor
        self.sg = summary_generator
        
    def process(self, title, text, url, score, comments):
        processed_text, url = text, url if text else clean_post_content(url)
        print(f"Processing post: {title} - {url}, Score: {score}, Comments: {comments}")
        summary = self.sg.generate(f"{processed_text}")
        print(f"Generated summary for {title}: {summary}")
        features = [
            self.fe.extract_numerical(score, comments),
            self.fe.extract_embeddings(summary['type'], 'type'),
            self.fe.extract_embeddings(summary['content'], 'content'),
            self.fe.extract_embeddings(summary['keywords'], 'keyword')
        ]
        return np.concatenate(features).reshape(1, -1), summary, url

class HNPredictor:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FeatureFusionNetwork()
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, features):
        # Ensure features is 2D [1, n_features]
        #if len(features.shape) == 1:
        #    features = features.reshape(1, -1)
        print(f"features shape: {features.shape}")

        # Convert to tensor and move to model's device
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(features_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return np.argmax(probs), np.max(probs)

predictor = HNPredictor('interest_model_fold4_base2.pt')

def fetch_top_posts(days_back=1, limit=200):
    """Fetch top HN posts from Supabase where content_summary is null"""
    # Initialize Supabase client
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_KEY')
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
    
    supabase_client = supabase.create_client(supabase_url, supabase_key)
    
    # Query Supabase for posts where content_summary is null
    try:
        response = supabase_client.table('hn_posts').select(
            'id, hn_id, title, url, points, created_at, descendants, user_id, content_summary, text'
        ).is_(
            'content_summary', 'null'
        ).order(
            'created_at', desc=True  # Get most recent posts first
        ).limit(limit).execute()
        
        # Transform Supabase response to match existing format
        posts = []
        for row in response.data:
            post = {
                'objectID': row['hn_id'],  # Using hn_id as objectID for compatibility
                'title': row['title'],
                'url': row['url'],
                'points': row['points'],
                'num_comments': row['descendants'] or 0,  # descendants maps to num_comments
                'author': row['user_id'],
                'created_at_i': int(datetime.fromisoformat(row['created_at'].replace('Z', '+00:00')).timestamp()),
                'story_text': row['text'] or ''  # content_summary maps to story_text (will be empty)
            }
            posts.append(post)
        
        print(f"Fetched {len(posts)} posts from Supabase with null content_summary")
        return posts
        
    except Exception as e:
        print(f"Error fetching posts from Supabase: {e}")
        return []

def process_and_update_posts(posts):
    """Process posts, extract summaries, and batch update Supabase"""
    if not posts:
        print("No posts to process")
        return
    
    # Initialize components
    fe = FeatureExtractor()
    sg = SummaryGenerator()
    processor = HNPostProcessor(fe, sg)
    #predictor = HNPredictor('interest_model_fold4_base2.pt')
    
    # Initialize Supabase client
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_KEY')
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
    
    supabase_client = supabase.create_client(supabase_url, supabase_key)
    
    # Process posts and collect updates
    updates = []
    for i, post in enumerate(posts):
        try:
            print(f"Processing post {i+1}/{len(posts)}: {post['title']}")
            
            # Process post to get features and summary
            features, summary, url = processor.process(
                post['title'],
                post.get('story_text', ''),
                post.get('url', ''),
                post['points'],
                post['num_comments']
            )
            
            # Convert features to JSON with shape and dtype information
            features_json = {
                'data': features.flatten().tolist(),
                'shape': features.shape,
                'dtype': str(features.dtype)
            }
            
            # Create update record
            update_record = {
                'id': post['id'],  # Use the database ID for updating
                'content_summary': summary,
                'post_features': features_json,  # Store features as JSON with metadata
                'url': url  # Store cleaned URL
            }
            updates.append(update_record)
            
        except Exception as e:
            print(f"Error processing post {post['title']}: {e}")
            continue
    
    # Batch update Supabase
    if updates:
        try:
            # Update posts in batches to avoid timeouts
            batch_size = 50
            for i in range(0, len(updates), batch_size):
                batch = updates[i:i+batch_size]
                print(f"Updating batch {i//batch_size+1}/{(len(updates)-1)//batch_size+1}")
                
                # Perform batch update
                response = supabase_client.table('hn_posts').upsert(
                    batch,
                    on_conflict='id'
                ).execute()
                
                print(f"Updated {len(batch)} posts in batch {i//batch_size+1}")
                
            print(f"Successfully updated {len(updates)} posts in Supabase")
            
        except Exception as e:
            print(f"Error updating posts in Supabase: {e}")
    else:
        print("No posts to update")

if __name__ == '__main__':
    print("Fetching top HN posts...")
    posts = fetch_top_posts(limit=50)
    process_and_update_posts(posts)
    print(f"Processed {len(posts)} posts")
