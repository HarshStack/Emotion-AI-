from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YOUR_API_KEY_HERE')
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
CHATGPT_MODEL = "gpt-5-turbo"

EMOTION_KEYWORDS = {
    'joy': ['happy', 'joy', 'great', 'wonderful', 'amazing', 'fantastic', 'delighted', 'pleased', 'cheerful'],
    'sadness': ['sad', 'down', 'depressed', 'miserable', 'unhappy', 'gloomy', 'heartbroken', 'upset'],
    'anger': ['angry', 'mad', 'furious', 'rage', 'hate', 'irritated', 'frustrated'],
    'fear': ['scared', 'afraid', 'terrified', 'frightened', 'fearful'],
    'nervousness': ['anxious', 'worried', 'nervous', 'stressed', 'concerned', 'uneasy'],
    'excitement': ['excited', 'thrilled', 'pumped', 'energized', 'enthusiastic'],
    'gratitude': ['thank', 'grateful', 'appreciate', 'thanks', 'thankful'],
    'love': ['love', 'adore', 'cherish', 'affection'],
    'disappointment': ['disappointed', 'let down'],
    'surprise': ['surprised', 'shocked', 'amazed', 'astonished'],
    'pride': ['proud', 'accomplished', 'achieved'],
    'confusion': ['confused', 'puzzled', 'unclear', 'bewildered'],
    'curiosity': ['curious', 'wonder', 'interested', 'intrigued'],
    'disgust': ['disgusted', 'gross', 'revolting'],
    'embarrassment': ['embarrassed', 'ashamed', 'humiliated'],
    'grief': ['grief', 'mourning', 'loss'],
    'remorse': ['sorry', 'regret', 'guilty'],
    'relief': ['relief', 'relieved', 'calm'],
    'admiration': ['admire', 'respect', 'impressed'],
    'amusement': ['funny', 'hilarious', 'amusing', 'laugh'],
    'approval': ['approve', 'agree', 'correct'],
    'caring': ['care', 'concern', 'compassion'],
    'desire': ['want', 'wish', 'desire', 'crave'],
    'disapproval': ['disapprove', 'disagree', 'wrong'],
    'optimism': ['hope', 'optimistic', 'positive'],
    'realization': ['realize', 'understand', 'aha']
}

def query_openai(messages, max_tokens=150, temperature=0.7):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": CHATGPT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data
        elif response.status_code == 401:
            return {"error": "Invalid API key"}
        elif response.status_code == 429:
            return {"error": "Rate limit exceeded"}
        else:
            return {"error": f"API Error: {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout"}
    except Exception as e:
        return {"error": str(e)}

def detect_emotion_from_text(text):
    text_lower = text.lower()
    emotion_scores = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score
    if emotion_scores:
        top_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[top_emotion] / len(EMOTION_KEYWORDS[top_emotion])
        return top_emotion, min(confidence, 1.0)
    return 'neutral', 0.5

def detect_emotion_with_gpt(text):
    messages = [
        {
            "role": "system",
            "content": """You are an emotion detection expert. Analyze the user's message and respond with ONLY the emotion name from this list:
joy, sadness, anger, fear, nervousness, excitement, gratitude, love, disappointment, surprise, pride, confusion, curiosity, disgust, embarrassment, grief, remorse, relief, admiration, amusement, approval, caring, desire, disapproval, optimism, realization, annoyance, neutral.

Respond with just the emotion name, nothing else."""
        },
        {
            "role": "user",
            "content": f"Detect the primary emotion in this message: '{text}'"
        }
    ]
    result = query_openai(messages, max_tokens=10, temperature=0.3)
    if isinstance(result, dict) and 'error' not in result:
        try:
            emotion = result['choices'][0]['message']['content'].strip().lower()
            valid_emotions = list(EMOTION_KEYWORDS.keys()) + ['neutral', 'annoyance']
            if emotion in valid_emotions:
                return emotion, 0.9
        except:
            pass
    return detect_emotion_from_text(text)

@app.route('/api/detect-emotion', methods=['POST'])
def detect_emotion():
    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        emotion, confidence = detect_emotion_with_gpt(text)
        return jsonify({'emotion': emotion, 'confidence': confidence, 'method': 'gpt' if confidence > 0.8 else 'keyword'})
    except Exception as e:
        emotion, confidence = detect_emotion_from_text(text)
        return jsonify({'emotion': emotion, 'confidence': confidence, 'method': 'fallback', 'error': str(e)})

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        conversation_history = data.get('history', [])
        user_name = data.get('userName', 'User')
        if not user_message:
            return jsonify({'error': 'No message provided', 'success': False}), 400
        detected_emotion, confidence = detect_emotion_with_gpt(user_message)
        system_prompt = f"""You are MindfulAI, a compassionate emotional support companion chatting with {user_name}.
1. Recognize the user's current emotion: {detected_emotion}
2. Respond to their SPECIFIC situation, not just their emotion
3. If they ask for advice or help with a problem, provide actionable suggestions
4. If they share a situation, acknowledge it and offer relevant support
5. Keep responses warm, personal, and helpful (2-4 sentences)
6. Use their name occasionally: {user_name}
- Be warm and human"""

        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history[-8:]:
            messages.append({"role": msg['role'], "content": msg['content']})
        messages.append({"role": "user", "content": user_message})
        result = query_openai(messages, max_tokens=250, temperature=0.9)
        ai_response = None
        using_fallback = False
        if isinstance(result, dict) and 'error' in result:
            ai_response = get_fallback_response(detected_emotion, user_name)
            using_fallback = True
        elif isinstance(result, dict) and 'choices' in result:
            ai_response = result['choices'][0]['message']['content']
        else:
            ai_response = get_fallback_response(detected_emotion, user_name)
            using_fallback = True
        return jsonify({'response': ai_response, 'emotion': detected_emotion, 'confidence': confidence, 'fallback': using_fallback, 'success': True})
    except Exception as e:
        emotion, _ = detect_emotion_from_text(data.get('message', ''))
        fallback_resp = get_fallback_response(emotion, data.get('userName', 'User'))
        return jsonify({'response': fallback_resp, 'emotion': emotion, 'confidence': 0.5, 'fallback': True, 'success': True, 'error_details': str(e)})

def get_fallback_response(emotion, user_name):
    responses = {
        'joy': f"{user_name}, I can feel your happiness! What's bringing you this joy?",
        'sadness': f"{user_name}, I'm here for you. Tell me what's on your mind.",
        'anger': f"{user_name}, I understand you're frustrated. Let's talk about it.",
        'fear': f"{user_name}, it's okay to feel scared. I'm here with you.",
        'nervousness': f"{user_name}, anxiety can be overwhelming. Let's work through this together.",
        'excitement': f"{user_name}, your excitement is contagious! Tell me more!",
        'gratitude': f"{user_name}, your gratitude is beautiful. What are you thankful for?",
        'love': f"{user_name}, that's so heartwarming! Share more with me.",
        'disappointment': f"{user_name}, I hear that you're disappointed. Want to talk about it?",
        'surprise': f"{user_name}, wow! That sounds surprising! Tell me more.",
        'pride': f"{user_name}, you should be proud! Share your accomplishment with me.",
        'confusion': f"{user_name}, let's work through this confusion together.",
        'curiosity': f"{user_name}, I love your curiosity! What are you wondering about?",
        'grief': f"{user_name}, I'm so sorry you're going through this. I'm here to listen.",
        'admiration': f"{user_name}, it's wonderful that you appreciate that! Tell me more.",
        'relief': f"{user_name}, I'm glad you're feeling better! What changed?",
        'neutral': f"{user_name}, I'm here to listen. Tell me more about what's on your mind."
    }
    return responses.get(emotion, responses['neutral'])

@app.route('/api/sentiment-analysis', methods=['POST'])
def sentiment_analysis():
    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        emotion, confidence = detect_emotion_with_gpt(text)
        positive_emotions = ['joy', 'excitement', 'gratitude', 'love', 'pride', 'admiration', 'amusement', 'approval', 'caring', 'desire', 'optimism', 'relief']
        negative_emotions = ['sadness', 'anger', 'fear', 'nervousness', 'disappointment', 'disgust', 'embarrassment', 'grief', 'remorse', 'disapproval', 'annoyance']
        if emotion in positive_emotions:
            sentiment = 'positive'
        elif emotion in negative_emotions:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        return jsonify({'sentiment': sentiment, 'emotion': emotion, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/emotion-breakdown', methods=['POST'])
def emotion_breakdown():
    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        messages = [
            {"role": "system", "content": "You are an emotion analysis expert. Analyze the text and identify ALL emotions present with their intensity (0-1 scale). Respond in JSON format only."},
            {"role": "user", "content": f"Analyze emotions in: '{text}'\n\nRespond with JSON: {{\"emotions\": [{{\"emotion\": \"joy\", \"score\": 0.9}}, ...]}}"}
        ]
        result = query_openai(messages, max_tokens=150, temperature=0.3)
        if isinstance(result, dict) and 'choices' in result:
            try:
                content = result['choices'][0]['message']['content']
                emotion_data = json.loads(content)
                return jsonify(emotion_data)
            except:
                pass
        emotion, confidence = detect_emotion_with_gpt(text)
        return jsonify({'emotions': [{'emotion': emotion, 'score': confidence}], 'top_emotion': emotion})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    api_key_set = OPENAI_API_KEY != 'YOUR_API_KEY_HERE' and OPENAI_API_KEY != ''
    return jsonify({'status': 'healthy', 'service': 'MindfulAI Backend (ChatGPT)', 'version': '3.0.0', 'ai_provider': 'OpenAI', 'model': CHATGPT_MODEL, 'api_key_configured': api_key_set})

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({'provider': 'OpenAI', 'model': CHATGPT_MODEL, 'features': {'emotion_detection': True, 'conversation': True, 'sentiment_analysis': True, 'emotion_breakdown': True, 'fallback_mode': True, 'context_aware': True}})

if __name__ == '__main__':
    if OPENAI_API_KEY == 'YOUR_API_KEY_HERE' or OPENAI_API_KEY == '':
        print("⚠  WARNING: OPENAI_API_KEY not set!")
        print("Please create a .env file with OPENAI_API_KEY=sk-your_openai_key_here")
    else:
        print(f"✅ API Key configured: {OPENAI_API_KEY[:10]}...")
    print(" MindfulAI Backend Server Starting (ChatGPT)")
    print(f"AI Provider: OpenAI")
    print(f" Model: {CHATGPT_MODEL}")
    print(f"  Features: Emotion detection, Conversational AI")
    print(f" Server: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
