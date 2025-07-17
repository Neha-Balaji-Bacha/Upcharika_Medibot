# Main app (UI + RAG + auth + voice).
# medibot_dynamic_auth.py
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from deep_translator import GoogleTranslator
from gtts import gTTS
import re
import os
import tempfile
import pandas as pd
import hashlib
import csv
from pathlib import Path
from datetime import datetime

# ---------------------- User Database Configuration ---------------------- #
USER_DB_PATH = "data/user_database.csv"
CSV_HEADERS = ["username", "password_hash", "full_name", "role", "email", "registration_date"]

def init_user_database():
    Path("data").mkdir(exist_ok=True)
    if not Path(USER_DB_PATH).exists():
        with open(USER_DB_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()

def load_users():
    init_user_database()
    users = {}
    with open(USER_DB_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            users[row["username"]] = row
    return users

def save_user(user_data):
    init_user_database()
    with open(USER_DB_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(user_data)

# ---------------------- Authentication Functions ---------------------- #
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(form_data):
    users = load_users()
    if form_data["username"] in users:
        raise ValueError("Username already exists")
    if form_data["password"] != form_data["confirm_password"]:
        raise ValueError("Passwords do not match")
    
    user_record = {
        "username": form_data["username"],
        "password_hash": hash_password(form_data["password"]),
        "full_name": form_data.get("full_name", ""),
        "role": form_data["role"],
        "email": form_data.get("email", ""),
        "registration_date": datetime.now().isoformat()
    }
    
    save_user(user_record)
    return user_record

def authenticate_user(username, password):
    users = load_users()
    user = users.get(username)
    if not user:
        return False
    if user["password_hash"] != hash_password(password):
        return False
    return user

# ---------------------- Custom CSS Injection ---------------------- #
def inject_custom_css():
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
            min-height: 100vh;
        }
        
        [data-testid="stChatMessage"] {
            background: white !important;
            border-radius: 15px !important;
            margin: 10px 0 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
            padding: 15px !important;
        }
        
        .assistant-message {
            background: #e3f2fd !important;
            border-left: 4px solid #2196f3 !important;
        }
        
        .user-message {
            background: #ffffff !important;
            border-left: 4px solid #4caf50 !important;
        }
        
        .error-message {
            color: #b71c1c !important;
            background: #ffebee !important;
            border-left: 4px solid #ff1744 !important;
        }
        
        .auth-container {
            max-width: 500px;
            margin: 2rem auto;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            background: white;
        }
    </style>
    """, unsafe_allow_html=True)

# ---------------------- Configuration ---------------------- #
MODEL_PATH = "models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
FAISS_DB_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT_TEMPLATE = """
You are Medico, an AI medical assistant. Provide accurate, structured responses following these rules:
1. Start response directly without empty lines
2. Use simple language with bullet points
3. Add relevant emojis where appropriate 
4. Include disclaimer to consult a doctor
5. For serious conditions, emphasize professional consultation

Context: {context}
Question: {question}

Respond in this format:
Main Topic
Key Point 1
Key Point 2
Important Note

Medical Response:"""

# ---------------------- Cached Resources ---------------------- #
@st.cache_resource
def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.3,
        max_tokens=1024,
        n_ctx=2048,
        n_batch=512,
        verbose=False
    )
    
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# ---------------------- Helper Functions ---------------------- #
def translate_response(text, target_lang):
    def process_line(line):
        try:
            line = re.sub(r'\s+', ' ', line).strip()
            if not line: return ""
            
            if line.startswith("###"):
                parts = line.split(" ", 1)
                header = parts[0]
                content = parts[1] if len(parts) > 1 else ""
                translated_content = GoogleTranslator(
                    source='auto', 
                    target=target_lang
                ).translate(content)
                return f"{header} {translated_content}"
                
            return GoogleTranslator(
                source='auto', 
                target=target_lang
            ).translate(line)
            
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return line
            
    return "\n".join([process_line(l) for l in text.split('\n') if l.strip()])

def text_to_speech(text, lang_code):
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(fp.name)
            return open(fp.name, "rb").read()
    except Exception as e:
        st.error(f"Audio generation failed: {str(e)}")
        return None
    finally:
        try: os.unlink(fp.name)
        except: pass

# ---------------------- Authentication UI ---------------------- #
def auth_form():
    inject_custom_css()
    with st.container():
        st.subheader("🏥 Upcharika Medical Portal")
        
        auth_mode = st.radio("Select Mode:", ["Login", "Register"], 
                           horizontal=True, label_visibility="collapsed")
        
        with st.form(key="auth_form"):
            if auth_mode == "Register":
                full_name = st.text_input("Full Name *")
                email = st.text_input("Email")
                role = st.selectbox("Role *", ["patient", "doctor"], 
                                   format_func=lambda x: "⚕ Doctor" if x == "doctor" else "👤 Patient")
            
            username = st.text_input("Username *", key="auth_username")
            password = st.text_input("Password *", type="password", key="auth_password")
            
            if auth_mode == "Register":
                confirm_password = st.text_input("Confirm Password *", type="password")
            
            submit_button = st.form_submit_button("Login" if auth_mode == "Login" else "Register")
        
        if submit_button:
            handle_auth_submit(auth_mode, locals())
        
        st.markdown("</div>", unsafe_allow_html=True)

def handle_auth_submit(mode, form_locals):
    try:
        if mode == "Login":
            # Add validation
            if not form_locals["username"] or not form_locals["password"]:
                raise ValueError("Please fill in all required fields (marked with *)")
            
            user = authenticate_user(form_locals["username"], form_locals["password"])
            if user:
                st.session_state.update({
                    "authenticated": True,
                    "username": user["username"],
                    "user_role": user["role"],
                    "user_fullname": user["full_name"]
                })
                st.rerun()
            else:
                st.error("Invalid credentials")
        
        elif mode == "Register":
            # Add validation
            required_fields = [
                ("full_name", "Full Name is required"),
                ("username", "Username is required"),
                ("password", "Password is required"),
                ("confirm_password", "Confirm Password is required")
            ]
            
            for field, message in required_fields:
                if not form_locals.get(field):
                    raise ValueError(message)
            
            if form_locals["password"] != form_locals["confirm_password"]:
                raise ValueError("Passwords do not match")
            
            user_data = {
                "username": form_locals["username"],
                "password": form_locals["password"],
                "confirm_password": form_locals["confirm_password"],
                "role": form_locals["role"],
                "full_name": form_locals["full_name"],
                "email": form_locals["email"]
            }
            
            new_user = register_user(user_data)
            st.success("Registration successful! Please login.")
            st.session_state.auth_mode = "Login"
            st.rerun()
    
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
# ---------------------- Session Management ---------------------- #
def initialize_session():
    session_defaults = {
        'authenticated': False,
        'username': None,
        'user_role': None,
        'user_fullname': "",
        'user_history': {},
        'current_messages': [],
        'lang_code': "en",
        'voice_enabled': False
    }
    
    for key, val in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def logout_user():
    st.session_state.update({
        "authenticated": False,
        "username": None,
        "user_role": None,
        "user_fullname": "",
        "current_messages": []
    })
    st.rerun()

# ---------------------- Main Application ---------------------- #
def main_app():
    inject_custom_css()
    
    # Sidebar
    with st.sidebar:
        if st.session_state.user_role == "doctor":
            st.title(f"⚕ Dr. {st.session_state.user_fullname.split()[0]}")
        else:
            st.title(f"👤 {st.session_state.user_fullname}")
        
        st.markdown("---")
        st.subheader("🌐 Language Settings")
        new_lang = st.selectbox(
            "Select Language",
            options=["en", "hi", "mr", "gu", "ta", "te"],
            format_func=lambda x: {
                "en": "🇺🇸 English",
                "hi": "🇮🇳 Hindi", 
                "mr": "🇮🇳 Marathi",
                "gu": "🇮🇳 Gujarati",
                "ta": "🇮🇳 Tamil",
                "te": "🇮🇳 Telugu"
            }[x]
        )
        if new_lang != st.session_state.lang_code:
            st.session_state.lang_code = new_lang
            st.rerun()
        
        st.session_state.voice_enabled = st.checkbox("🔊 Enable Voice Responses")
        
        # Role-specific features
        if st.session_state.user_role == "doctor":
            st.markdown("---")
            st.subheader("Doctor Tools")
            if st.button("📁 Patient Records"):
                st.info("Doctor feature: Patient records access")
        
        st.markdown("---")
        if st.button("🚪 Logout"):
            logout_user()

    # Main Chat Interfaces
    st.title("🏥 Upcharika")
    st.caption("Your 24/7 Digital Vaidya-Prescribing Care, Not Just Cures")
    
    # Chat History
    # In the main_app() function, modify the chat message rendering:
    for idx, msg in enumerate(st.session_state.current_messages):
        # Use default avatars based on role
        with st.chat_message(msg["role"]):  # Remove explicit avatar parameter
            col1, col2 = st.columns([9,1])
            with col1:
                # Add emoji prefix based on role
                emoji_prefix = "⚕ " if msg["role"] == "assistant" else "👤 "
                message_class = "assistant-message" if msg["role"] == "assistant" else "user-message"
                
                if msg.get("error"):
                    message_class = "error-message"
                    emoji_prefix = "⚠ "
                
                st.markdown(f"""
                <div class="{message_class}">
                    {emoji_prefix}{msg["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # ... rest of the column 2 code remains the same
            with col2:
                if msg["role"] == "assistant" and not msg.get("error"):
                    if st.button("👍", key=f"like_{idx}"):
                        st.toast("Thank you for your feedback!")
                    if st.button("👎", key=f"dislike_{idx}"):
                        st.toast("We'll improve this response!")

            if st.session_state.voice_enabled and msg.get("audio"):
                st.audio(msg["audio"], format="audio/mp3")

    # Chat Input
    if prompt := st.chat_input("Type your medical question..."):
        process_query(prompt)

    # Emergency Button
    st.markdown("""
    <div style="position: fixed; bottom: 20px; right: 20px;">
        <button style="
            background: #ff4444;
            color: white;
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            font-size: 24px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            cursor: pointer;
        " onclick="alert('Emergency services contacted!')">🚑</button>
    </div>
    """, unsafe_allow_html=True)

def process_query(prompt):
    st.session_state.current_messages.append({"role": "user", "content": prompt})
    
    with st.spinner(""):
        try:
            # Remove previous errors
            st.session_state.current_messages = [
                msg for msg in st.session_state.current_messages 
                if not msg.get("error")
            ]
            
            # Process query
            qa_chain = load_qa_chain()
            result = qa_chain.invoke({"query": prompt})
            
            if not result or "result" not in result:
                raise ValueError("Invalid response from AI model")
                
            # Process response
            raw_response = result["result"]
            sanitized_response = re.sub(r'\n{3,}', '\n\n', raw_response)
            sanitized_response = re.sub(r'[^\w\s\-\.\,\?\!\#\@\:\/\n]', '', sanitized_response)
            
            # Translate
            translated = translate_response(sanitized_response, st.session_state.lang_code)
            translated = re.sub(r'###\s*', '### ', translated)
            translated = re.sub(r'^-\s*', '- ', translated)
            
            # Audio generation
            audio_bytes = None
            if st.session_state.voice_enabled:
                audio_bytes = text_to_speech(translated, st.session_state.lang_code)
            
            # Store response
            st.session_state.current_messages.append({
                "role": "assistant",
                "content": translated,
                "audio": audio_bytes
            })
            
        except Exception as e:
            error_msg = f"⚠ Error processing request: {str(e)}"
            st.session_state.current_messages.append({
                "role": "assistant",
                "content": error_msg,
                "error": True
            })
            
        finally:
            # Save to history
            if st.session_state.username not in st.session_state.user_history:
                st.session_state.user_history[st.session_state.username] = []
            st.session_state.user_history[st.session_state.username].append(
                st.session_state.current_messages.copy()
            )
            st.rerun()

# ---------------------- Main Function ---------------------- #
def main():
    st.set_page_config(
        page_title="Upcharika",
        page_icon="🏥",
        layout="centered"
    )
    initialize_session()
    
    if not st.session_state.authenticated:
        auth_form()
    else:
        main_app()

if __name__ == "__main__":
    main()