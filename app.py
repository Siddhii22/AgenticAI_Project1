import os
import asyncio
import tempfile
import requests
import xml.etree.ElementTree as ET
import streamlit as st

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

from faster_whisper import WhisperModel
import edge_tts


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found.")
    st.stop()


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Medical Information Voice Assistant",
    page_icon="🩺",
    layout="centered"
)


# ==================================================
# UI
# ==================================================

st.title("🩺 Medical Information Voice Assistant")

st.write(
    "Ask general medical questions using your voice or text."
)

st.warning(
    "⚠️ This assistant provides general educational information only. "
    "It does not diagnose diseases or prescribe medicines."
)


# ==================================================
# LOAD LLM
# ==================================================

@st.cache_resource
def load_model():

    return ChatOpenRouter(
        model="z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0
    )


model = load_model()


# ==================================================
# LOAD WHISPER
# ==================================================

@st.cache_resource
def load_whisper():

    return WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )


whisper_model = load_whisper()


# ==================================================
# MEDICAL INFORMATION TOOL
# ==================================================

@tool
def medical_information(topic: str) -> str:
    """
    Search MedlinePlus for general medical information.
    Provides educational information only and does not diagnose
    or prescribe.
    """

    url = "https://wsearch.nlm.nih.gov/ws/query"

    params = {
        "db": "healthTopics",
        "term": topic,
        "retmax": 3,
        "rettype": "brief"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        root = ET.fromstring(response.text)

        results = []

        for document in root.findall(".//document"):

            title = ""
            summary = ""

            page_url = document.attrib.get(
                "url",
                ""
            )

            for content in document.findall("content"):

                name = content.attrib.get("name")

                text = "".join(
                    content.itertext()
                ).strip()

                if name == "title":
                    title = text

                elif name == "full-summary":
                    summary = text

            if title or summary:

                results.append({
                    "title": title,
                    "summary": summary,
                    "url": page_url
                })


        if not results:

            return (
                f"No medical information found "
                f"for: {topic}"
            )


        output = (
            f"Medical information from MedlinePlus "
            f"for '{topic}':\n\n"
        )


        for i, result in enumerate(results, 1):

            output += (
                f"{i}. {result['title']}\n"
            )

            output += (
                f"{result['summary']}\n"
            )

            if result["url"]:

                output += (
                    f"Source: {result['url']}\n"
                )

            output += "\n"


        output += (
            "Important: This information is for "
            "educational purposes only. "
            "It does not provide a diagnosis "
            "or medical prescription."
        )

        return output


    except requests.RequestException as e:

        return (
            "Unable to access MedlinePlus: "
            + str(e)
        )


    except ET.ParseError:

        return (
            "Unable to process the medical "
            "information returned by MedlinePlus."
        )


# ==================================================
# BIND TOOL WITH MODEL
# ==================================================

model_with_tools = model.bind_tools(
    [medical_information]
)


# ==================================================
# TEXT TO SPEECH
# ==================================================

async def generate_voice(text):

    voice = "en-IN-NeerjaNeural"

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    )

    output_file.close()

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice
    )

    await communicate.save(
        output_file.name
    )

    return output_file.name


# ==================================================
# SPEECH TO TEXT
# ==================================================

def speech_to_text(audio_file):

    temp_audio = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    )

    temp_audio.write(
        audio_file.getvalue()
    )

    temp_audio.close()


    segments, info = whisper_model.transcribe(
        temp_audio.name
    )


    text = " ".join(
        segment.text
        for segment in segments
    ).strip()


    return text


# ==================================================
# USER INPUT
# ==================================================

st.subheader("🎤 Ask your question")

st.write(
    "Speak your medical question using the microphone."
)

audio_value = st.audio_input(
    "🎤 Record your question"
)

user_text = ""


# ==================================================
# PROCESS VOICE INPUT
# ==================================================

if audio_value is not None:

    with st.spinner(
        "🎙️ Converting your voice to text..."
    ):

        try:

            user_text = speech_to_text(
                audio_value
            )

            if user_text:

                st.success(
                    f"🗣️ You said: {user_text}"
                )

            else:

                st.warning(
                    "I could not understand your voice. "
                    "Please try again."
                )

        except Exception as e:

            st.error(
                "Speech recognition failed."
            )

            with st.expander(
                "Show Error Details"
            ):

                st.code(str(e))


# ==================================================
# TEXT FALLBACK
# ==================================================

st.write("Or type your question:")

text_input = st.text_input(
    "Medical question",
    placeholder="Example: Tell me about anemia"
)

if text_input.strip():

    user_text = text_input


# ==================================================
# ASK ASSISTANT
# ==================================================

if st.button(
    "🩺 Ask Assistant",
    use_container_width=True
):

    if not user_text.strip():

        st.warning(
            "Please speak or type a medical question."
        )

    else:

        with st.spinner(
            "🔎 Searching medical information..."
        ):

            try:

                # ==================================
                # SYSTEM PROMPT
                # ==================================

                system_prompt = """
You are a medical information voice assistant.

Your job is to provide general educational
medical information.

Important rules:

- Do not diagnose diseases.
- Do not prescribe medicines.
- Do not replace a healthcare professional.
- Use the medical information tool when relevant.
- Keep the answer concise.
- Use simple language.
- Since the answer will also be spoken aloud,
  do not use Markdown.
- Avoid unnecessarily long lists.
- Clearly mention when professional medical
  advice may be needed.
"""


                # ==================================
                # INITIAL MESSAGES
                # ==================================

                messages = [

                    SystemMessage(
                        content=system_prompt
                    ),

                    HumanMessage(
                        content=user_text
                    )

                ]


                # ==================================
                # AGENT LOOP
                # ==================================

                max_iterations = 3

                answer = None


                for _ in range(
                    max_iterations
                ):

                    response = (
                        model_with_tools.invoke(
                            messages
                        )
                    )


                    messages.append(
                        response
                    )


                    # ==================================
                    # TOOL CALL
                    # ==================================

                    if response.tool_calls:

                        for tool_call in (
                            response.tool_calls
                        ):

                            tool_name = (
                                tool_call["name"]
                            )

                            tool_args = (
                                tool_call["args"]
                            )


                            if (
                                tool_name
                                == "medical_information"
                            ):

                                tool_result = (
                                    medical_information
                                    .invoke(
                                        tool_args
                                    )
                                )


                                messages.append(

                                    ToolMessage(
                                        content=str(
                                            tool_result
                                        ),
                                        tool_call_id=(
                                            tool_call["id"]
                                        )
                                    )

                                )


                    # ==================================
                    # FINAL RESPONSE
                    # ==================================

                    else:

                        answer = (
                            response.content
                        )

                        break


                # ==================================
                # FALLBACK
                # ==================================

                if not answer:

                    answer = (
                        "I was unable to generate "
                        "a complete response. "
                        "Please try again."
                    )


                # ==================================
                # TEXT RESPONSE
                # ==================================

                st.subheader(
                    "🩺 Medical Assistant Response"
                )

                st.write(answer)


                # ==================================
                # VOICE RESPONSE
                # ==================================

                with st.spinner(
                    "🔊 Generating voice response..."
                ):

                    audio_path = asyncio.run(
                        generate_voice(answer)
                    )


                st.subheader(
                    "🔊 Voice Response"
                )


                with open(
                    audio_path,
                    "rb"
                ) as audio_file:

                    audio_bytes = (
                        audio_file.read()
                    )


                st.audio(
                    audio_bytes,
                    format="audio/mp3",
                    autoplay=True
                )


            except Exception as e:

                st.error(
                    "An error occurred while "
                    "processing your request."
                )

                with st.expander(
                    "Show Error Details"
                ):

                    st.code(
                        str(e)
                    )