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

def clean_post_content(url):
    """Fetch and clean post content from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
            
        # Get clean text
        return ' '.join(soup.stripped_strings)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""
    
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
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="nousresearch/deephermes-3-llama-3-8b-preview:free",
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
                
        # Fallback if all retries fail
        return {
            'type': parts['type'] if parts and parts['type'] else "无法提取类型",
            'content': parts['content'] if parts and parts['content'] else "无法提取内容",
            'keywords': parts['keywords'] if parts and parts['keywords'] else "无法提取关键词"
        }

class HNPostProcessor:
    def __init__(self, feature_extractor, summary_generator):
        self.fe = feature_extractor
        self.sg = summary_generator
        
    def process(self, title, text, url, score, comments):
        processed_text = text if text else clean_post_content(url)
        print(f"Processing post: {title} - {url}, Score: {score}, Comments: {comments}")
        summary = self.sg.generate(f"{processed_text}")
        print(f"Generated summary for {title}: {summary}")
        features = [
            self.fe.extract_numerical(score, comments),
            self.fe.extract_embeddings(summary['type'], 'type'),
            self.fe.extract_embeddings(summary['content'], 'content'),
            self.fe.extract_embeddings(summary['keywords'], 'keyword')
        ]
        return np.concatenate(features).reshape(1, -1), summary

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

def fetch_top_posts(days_back=1, limit=60):
    """Fetch top HN posts from the Algolia API"""
    # Calculate date range (previous day in UTC)
    end_date = datetime.now(pytz.utc)

    # Check for custom date from env var
    custom_date = os.getenv('DATE_FETCH')
    if custom_date:
        try:
            end_date = datetime.strptime(custom_date, '%Y/%m/%d').replace(
                tzinfo=pytz.utc
            )
            days_back = 1  # When using custom date, get that specific day's posts
        except ValueError:
            print(f"Invalid DATE_FETCH format, using current date")

    start_date = end_date - timedelta(days=days_back)
    
    params = {
        'tags': 'story',
        'hitsPerPage': limit,
        'numericFilters': f'created_at_i>{start_date.timestamp()},created_at_i<{end_date.timestamp()},points>=30',
        'query': '',
        'attributesToRetrieve': 'title,url,points,story_text,num_comments,created_at_i,author,objectID'
    }
    
    print(f"Fetching posts from {start_date} to {end_date}")
    response = requests.get('https://hn.algolia.com/api/v1/search', params=params)
    response.raise_for_status()
    return response.json()['hits']

def create_github_issue(posts, repo):
    fe = FeatureExtractor()
    sg = SummaryGenerator()
    processor = HNPostProcessor(fe, sg)
    predictor = HNPredictor('interest_model_fold4_base2.pt')

    """Create GitHub issue with HN posts"""
    # Determine title date
    issue_date = datetime.now(pytz.utc)
    if custom_date := os.getenv('DATE_FETCH'):
        try:
            issue_date = datetime.strptime(custom_date, '%Y/%m/%d').replace(tzinfo=pytz.utc)
        except ValueError:
            pass
            
    issue_title = f"HN Top Posts {issue_date.strftime('%Y-%m-%d')}"

    # Format markdown content
    markdown = "## Today's Top Hacker News Posts\n\n"
    markdown = "| Title | Points | Comments | Author | Category | Confidence |\n"
    markdown += "|-------|--------|----------|--------|----------|------------|\n"

    for post in posts:
        features, summary = processor.process(
            post['title'],
            post.get('story_text', ''),
            post.get('url', ''),
            post['points'],
            post['num_comments']
        )
        category, confidence = predictor.predict(features)
        emoji = ["💔", "❤️"][category]  # Replace with your category emojis
        url = post.get('url', f'https://news.ycombinator.com/item?id={post["objectID"]}')

        # Add summary as hover text
        summary_str = f"{summary.get('type','')}\n{summary.get('content','')}\nKeywords: {summary.get('keywords','')}"
        #markdown += f"| [{post['title']}]({url} \"{summary_str}\") | {post['points']} | {post['num_comments']} | {post['author']} | {emoji} {category} | {confidence:.2f} |\n"
        markdown += f"""<div style="margin-bottom: 16px">
  <h3><a href="{url}">{post['title']}</a> <small>{post['points']}pts | {post['num_comments']} comments</small></h3>
  <div><b>类型</b>: {summary['type']}<br>
  <b>摘要</b>: {summary['content']}<br>
  <b>关键词</b>: {summary['keywords']}</div>
  <div style="color: #666">分类: {emoji} {category} | 置信度: {confidence:.0%}</div>
</div>"""

    # Create issue via GitHub API
    response = requests.post(
        f"https://api.github.com/repos/{repo}/issues",
        headers={
            "Authorization": f"token {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github.v3+json"
        },
        json={
            "title": issue_title,
            "body": markdown
        }
    )
    response.raise_for_status()
    return response.json()

if __name__ == '__main__':
    print("Fetching top HN posts...")
    repo = os.environ['GITHUB_REPOSITORY']
    print(f"Using repository: {repo}")
    posts = fetch_top_posts()
    create_github_issue(posts, repo)
    print(f"Processed {len(posts)} posts")
