import os
import supabase
from datetime import datetime

def main():
    # Initialize Supabase client
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
    
    supabase_client = supabase.create_client(supabase_url, supabase_key)
    
    # Your periodic task logic here
    print(f"Periodic task executed at {datetime.now()}")
    
    # Example: Query some data from Supabase
    try:
        response = supabase_client.table('hn_posts').select('id, title').limit(5).execute()
        print(f"Found {len(response.data)} posts")
        for post in response.data:
            print(f"- {post['title']}")
    except Exception as e:
        print(f"Error querying Supabase: {e}")

if __name__ == '__main__':
    main()