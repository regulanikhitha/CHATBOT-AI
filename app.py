from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os
from dotenv import load_dotenv
import time
import logging
from functools import wraps

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyAOzO6Vm4nUGcy_6jDMsXDNNS2BhiYsW7o')
API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent'

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 60  # requests per minute
RATE_LIMIT_WINDOW = 60   # seconds
rate_limit_storage = {}

def rate_limit(max_requests=RATE_LIMIT_REQUESTS, window=RATE_LIMIT_WINDOW):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            current_time = time.time()
            
            if client_ip not in rate_limit_storage:
                rate_limit_storage[client_ip] = []
            
            # Remove old requests outside the window
            rate_limit_storage[client_ip] = [
                req_time for req_time in rate_limit_storage[client_ip]
                if current_time - req_time < window
            ]
            
            if len(rate_limit_storage[client_ip]) >= max_requests:
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'message': f'Maximum {max_requests} requests per {window} seconds'
                }), 429
            
            rate_limit_storage[client_ip].append(current_time)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def clean_markdown(text):
    """Clean markdown formatting from text"""
    import re
    
    # Remove markdown formatting
    text = re.sub(r'#{1,6}\s?', '', text)  # Headers
    text = re.sub(r'\*\*', '', text)       # Bold
    text = re.sub(r'\*', '', text)         # Italic
    text = re.sub(r'`', '', text)          # Code
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines
    
    return text.strip()

def generate_response_with_retry(prompt, max_retries=3):
    """Generate response with automatic retry for 503 errors"""
    base_delay = 1.0
    
    for attempt in range(max_retries + 1):
        try:
            headers = {
                'Content-Type': 'application/json',
            }
            
            payload = {
                'contents': [{
                    'parts': [{
                        'text': prompt
                    }]
                }],
                'generationConfig': {
                    'temperature': 0.7,
                    'topK': 40,
                    'topP': 0.95,
                    'maxOutputTokens': 1024,
                }
            }
            
            response = requests.post(
                f"{API_URL}?key={API_KEY}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse response
                if data.get('candidates') and len(data['candidates']) > 0:
                    candidate = data['candidates'][0]
                    if (candidate.get('content') and 
                        candidate['content'].get('parts') and 
                        len(candidate['content']['parts']) > 0):
                        return candidate['content']['parts'][0]['text']
                
                raise Exception('No response generated')
                
            elif response.status_code == 503 and attempt < max_retries:
                # Server overloaded, retry with exponential backoff
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Server overloaded (503). Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
                
            else:
                # Handle other errors
                try:
                    error_data = response.json()
                    error_message = error_data.get('error', {}).get('message', 'Unknown error')
                except:
                    error_message = f"HTTP {response.status_code}"
                
                if response.status_code == 400:
                    raise Exception('Invalid request. Please check your message.')
                elif response.status_code == 401:
                    raise Exception('Invalid API key configuration.')
                elif response.status_code == 429:
                    raise Exception('Rate limit exceeded. Please wait and try again.')
                else:
                    raise Exception(f'API Error: {error_message}')
                    
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Network error: {str(e)}. Retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                raise Exception('Network error. Please check your connection.')
        except Exception as e:
            if attempt < max_retries and "503" in str(e):
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            else:
                raise e
    
    raise Exception('Maximum retry attempts exceeded')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'api_configured': bool(API_KEY and API_KEY != 'YOUR_API_KEY_HERE')
    })

@app.route('/api/chat', methods=['POST'])
@rate_limit()
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Missing message field'
            }), 400
        
        user_message = data['message'].strip()
        
        if not user_message:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Message cannot be empty'
            }), 400
        
        if len(user_message) > 1000:
            return jsonify({
                'error': 'Message too long',
                'message': 'Please keep messages under 1000 characters'
            }), 400
        
        # Check API key
        if not API_KEY or API_KEY == 'YOUR_API_KEY_HERE':
            return jsonify({
                'error': 'Configuration error',
                'message': 'API key not configured'
            }), 500
        
        # Generate response
        logger.info(f"Processing message: {user_message[:50]}...")
        
        bot_response = generate_response_with_retry(user_message)
        cleaned_response = clean_markdown(bot_response)
        
        logger.info("Response generated successfully")
        
        return jsonify({
            'response': cleaned_response,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/chat/stream', methods=['POST'])
@rate_limit()
def chat_stream():
    """Streaming chat endpoint for real-time responses"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Missing message field'
            }), 400
        
        user_message = data['message'].strip()
        
        if not user_message:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Message cannot be empty'
            }), 400
        
        if len(user_message) > 1000:
            return jsonify({
                'error': 'Message too long',
                'message': 'Please keep messages under 1000 characters'
            }), 400
        
        # For now, return the same as regular chat
        # In a real implementation, you'd use Server-Sent Events or WebSockets
        bot_response = generate_response_with_retry(user_message)
        cleaned_response = clean_markdown(bot_response)
        
        return jsonify({
            'response': cleaned_response,
            'timestamp': time.time(),
            'streaming': False
        })
        
    except Exception as e:
        logger.error(f"Error in streaming chat endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get frontend configuration"""
    return jsonify({
        'max_message_length': 1000,
        'rate_limit': {
            'requests': RATE_LIMIT_REQUESTS,
            'window': RATE_LIMIT_WINDOW
        },
        'features': {
            'streaming': False,
            'file_upload': False,
            'image_generation': False
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    # Check if API key is configured
    if not API_KEY or API_KEY == 'YOUR_API_KEY_HERE':
        print("⚠️  Warning: GEMINI_API_KEY not configured!")
        print("   Please set your API key in the .env file or environment variables.")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )