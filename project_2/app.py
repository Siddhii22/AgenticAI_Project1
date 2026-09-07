import asyncio
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

import requests
import streamlit as st
from faster_whisper import WhisperModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
import edge_tts


st.set_page_config(
    page_title="Medical Voice Assistant",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Medical Information Voice Assistant")
st.caption("General educational information only — not a diagnosis or prescription.")

# -----------------------------
# Secrets / API key
# -----------------------------
try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error(
        "OPENROUTER_API_KEY is not configured. "
        "Add it to Streamlit Secrets before using the assistant."
    )
    st.stop()


# -----------------------------
# OpenRouter model
# -----------------------------
@st.cache_resource
def get_model():
    return ChatOpenRouter(
        model="z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0,
    )


model = get_model()


# -----------------------------
# Medical information tool
# -----------------------------
@tool
def medical_information(topic: str) -> str:
    """
    Search MedlinePlus for general medical information about a topic.
    Provides educational information only and does not diagnose or prescribe.
    """
    url = "https://wsearch.nlm.nih.gov/ws/query"
    params = {
        "db": "healthTopics",
        "term": topic,
        "retmax": 3,
        "rettype": "brief",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        import xml.etree.ElementTree as ET

        root = ET.fromstring(response.text)
        results = []

        for document in root.findall(".//document"):
            title = ""
            summary = ""
            page_url = document.attrib.get("url", "")

            for content in document.findall("content"):
                name = content.attrib.get("name")
                text = "".join(content.itertext()).strip()

                if name == "title":
                    title = text
                elif name == "full-summary":
                    summary = text

            if title or summary:
                results.append(
                    {
                        "title": title,
                        "summary": summary,
                        "url": page_url,
                    }
                )

        if not results:
            return f"No medical information found for: {topic}"

        output = f"Medical information from MedlinePlus for '{topic}':\n\n"

        for i, result in enumerate(results, 1):
            output += f"{i}. {result['title']}\n"
            output += f"{result['summary']}\n"
            output += f"Source: {result['url']}\n\n"

        output += (
            "Important: This information is for educational purposes only. "
            "It does not provide a diagnosis or medical prescription."
        )
        return output

    except requests.RequestException as e:
        return f"Unable to access MedlinePlus: {e}"
    except ET.ParseError:
        return "Unable to process the medical information returned by MedlinePlus."


model_with_tools = model.bind_tools([medical_information])


SYSTEM_PROMPT = """
You are a medical information voice assistant.

Your job is to provide general educational medical information.

Important rules:
- Do not diagnose diseases.
- Do not prescribe medicines.
- Do not replace a healthcare professional.
- Use information retrieved from the medical information tool when relevant.
- Keep answers concise because the response may be spoken aloud.
- Use simple language.
- Do not use Markdown in the spoken answer.
- Avoid long lists.
"""


# -----------------------------
# Faster Whisper
# -----------------------------
@st.cache_resource
def get_whisper():
    return WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
    )


def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        audio_path = f.name

    try:
        whisper = get_whisper()
        segments, _ = whisper.transcribe(audio_path)
        return " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass


# -----------------------------
# Agent execution
# -----------------------------
def run_agent(user_text):
    messages = [
        HumanMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]

    response = model_with_tools.invoke(messages)

    tool_used = False

    if response.tool_calls:
        tool_used = True
        messages.append(response)

        for tool_call in response.tool_calls:
            if tool_call["name"] == "medical_information":
                tool_result = medical_information.invoke(tool_call["args"])
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

        final_response = model_with_tools.invoke(messages)
        answer = final_response.content
    else:
        answer = response.content

    return answer, tool_used


# -----------------------------
# Voice generation
# -----------------------------
async def generate_voice(text):
    communicate = edge_tts.Communicate(
        text=text,
        voice="en-IN-NeerjaNeural",
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        audio_path = f.name

    await communicate.save(audio_path)
    return audio_path


# -----------------------------
# UI
# -----------------------------
st.subheader("Ask the assistant")

text_input = st.text_area(
    "Type your question",
    placeholder="Example: What is hypertension?",
    height=100,
)

st.write("Or speak your question:")

audio_input = st.audio_input("🎙️ Record your question")

question = text_input.strip()

if audio_input is not None:
    with st.spinner("Transcribing your voice..."):
        try:
            question = transcribe_audio(audio_input.getvalue())
            st.info(f"Transcribed: {question}")
        except Exception as e:
            st.error(f"Voice transcription failed: {e}")

if st.button("Ask Medical Assistant", type="primary", use_container_width=True):
    if not question:
        st.warning("Please type or record a question.")
    else:
        with st.spinner("Thinking and checking medical information..."):
            try:
                answer, tool_used = run_agent(question)

                st.markdown("### 🩺 Medical Assistant")
                st.write(answer)

                if tool_used:
                    st.caption("ℹ️ The assistant used the MedlinePlus medical-information tool.")

                with st.spinner("Generating voice response..."):
                    audio_path = asyncio.run(generate_voice(answer))
                    with open(audio_path, "rb") as audio_file:
                        st.audio(audio_file.read(), format="audio/mp3")

                    try:
                        os.remove(audio_path)
                    except OSError:
                        pass

            except Exception as e:
                st.error(f"Something went wrong: {e}")

st.divider()
st.caption(
    "This app provides general educational information and is not a substitute "
    "for professional medical advice, diagnosis, or treatment."
)
