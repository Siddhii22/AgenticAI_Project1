import os
import requests
import xml.etree.ElementTree as ET
import streamlit as st

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Medical Information Assistant",
    page_icon="🩺",
    layout="centered"
)


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🩺 Medical Information Assistant")

st.write(
    "Ask general medical questions and receive educational information."
)

st.warning(
    "⚠️ This assistant provides general educational information only. "
    "It does not diagnose diseases or prescribe medicines."
)


# --------------------------------------------------
# INITIALIZE MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = ChatOpenRouter(
        model="z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0
    )

    return model


model = load_model()


# --------------------------------------------------
# MEDICAL INFORMATION TOOL
# --------------------------------------------------

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
            page_url = document.attrib.get("url", "")

            for content in document.findall("content"):

                name = content.attrib.get("name")
                text = "".join(content.itertext()).strip()

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
            return f"No medical information found for: {topic}"

        output = f"Medical information from MedlinePlus for '{topic}':\n\n"

        for i, result in enumerate(results, 1):

            output += f"{i}. {result['title']}\n"
            output += f"{result['summary']}\n"

            if result["url"]:
                output += f"Source: {result['url']}\n"

            output += "\n"

        output += (
            "Important: This information is for educational purposes only. "
            "It does not provide a diagnosis or medical prescription."
        )

        return output

    except requests.RequestException as e:

        return f"Unable to access MedlinePlus: {str(e)}"

    except ET.ParseError:

        return "Unable to process the medical information returned by MedlinePlus."


# --------------------------------------------------
# BIND TOOL WITH MODEL
# --------------------------------------------------

model_with_tools = model.bind_tools(
    [medical_information]
)


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

user_text = st.text_input(
    "Ask your medical question:",
    placeholder="Example: Tell me about anemia"
)


# --------------------------------------------------
# BUTTON ACTION
# --------------------------------------------------

if st.button("Ask Assistant"):

    if not user_text.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching medical information..."):

            try:

                # --------------------------------------
                # SYSTEM PROMPT
                # --------------------------------------

                system_prompt = """
You are a medical information assistant.

Your job is to provide general educational medical information.

Important rules:
- Do not diagnose diseases.
- Do not prescribe medicines.
- Do not replace a healthcare professional.
- Use the medical information tool when relevant.
- Keep the answer concise and easy to understand.
- Clearly mention when professional medical advice may be needed.
"""


                # --------------------------------------
                # INITIAL MESSAGES
                # --------------------------------------

                messages = [

                    SystemMessage(
                        content=system_prompt
                    ),

                    HumanMessage(
                        content=user_text
                    )

                ]


                # --------------------------------------
                # AGENT LOOP
                # --------------------------------------

                max_iterations = 3
                answer = None


                for _ in range(max_iterations):

                    response = model_with_tools.invoke(
                        messages
                    )


                    # Add model response to conversation
                    messages.append(response)


                    # ----------------------------------
                    # IF MODEL WANTS TO CALL A TOOL
                    # ----------------------------------

                    if response.tool_calls:

                        for tool_call in response.tool_calls:

                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]


                            if tool_name == "medical_information":

                                tool_result = medical_information.invoke(
                                    tool_args
                                )


                                messages.append(

                                    ToolMessage(
                                        content=str(tool_result),
                                        tool_call_id=tool_call["id"]
                                    )

                                )

                    else:

                        # ----------------------------------
                        # FINAL RESPONSE RECEIVED
                        # ----------------------------------

                        answer = response.content
                        break


                # --------------------------------------
                # FALLBACK IF ANSWER IS EMPTY
                # --------------------------------------

                if not answer:

                    answer = (
                        "I was unable to generate a complete response. "
                        "Please try asking the question again."
                    )


                # --------------------------------------
                # DISPLAY ANSWER
                # --------------------------------------

                st.subheader("Medical Assistant Response")

                st.write(answer)


            except Exception as e:

                st.error("An error occurred while processing your request.")

                with st.expander("Show Error Details"):
                    st.code(str(e))