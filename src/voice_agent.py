import threading

try:
    import msvcrt
except ImportError:  # pragma: no cover - msvcrt is Windows-only
    msvcrt = None

from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from chemistry_agent import build_chemistry_chain
from english_agent import build_english_chain
from history_agent import build_history_chain
from math_agent import build_math_chain
from router_agent import route_subject

from conversation_utils import (
    barge_in_passes_threshold as _barge_in_passes_threshold,
    describe_page_range,
    extract_page_range,
    format_source,
    truncate_text,
)
from stt_module import SpeechToText
from tts_module import TextToSpeech
from vector import search_documents
from voice_config import load_subject_voice_map

MODEL_NAME = "llama3.1:8b"

VOICE_STOP_WORDS = {"quit", "stop", "exit", "bye"}
BARGE_IN_IDLE_SECONDS = 0.8
BARGE_IN_MIN_CHARS = 3
MEMORY_MAX_TURNS = 8
MEMORY_MAX_CHARS_PER_MESSAGE = 500
SUPPORTED_SUBJECTS = ("history", "chemistry", "math", "english")


def keyboard_quit_requested() -> bool:
    """Check for a non-blocking q key press in the terminal."""
    if msvcrt is None:
        return False

    try:
        return msvcrt.kbhit() and msvcrt.getwch().lower() == "q"
    except Exception:
        return False


def barge_in_passes_threshold(text: str) -> bool:
    """Apply interruption threshold rules for duplex barge-in transcripts."""
    return _barge_in_passes_threshold(text, VOICE_STOP_WORDS, BARGE_IN_MIN_CHARS)


class VoiceAgent:
    """Coordinate speech input, LLM responses, TTS output, and memory."""

    def __init__(self):
        """Initialize speech components, memory buffer, and the chat model."""
        self.ears = SpeechToText()
        self.mouth = TextToSpeech()
        self.memories = {
            subject: InMemoryChatMessageHistory() for subject in SUPPORTED_SUBJECTS
        }
        self.llm = ChatOllama(model=MODEL_NAME, streaming=True, temperature=0.7)
        self.specialist_chains = {
            "history": build_history_chain(self.llm),
            "chemistry": build_chemistry_chain(self.llm),
            "math": build_math_chain(self.llm),
            "english": build_english_chain(self.llm),
        }
        self.subject_voice_map = load_subject_voice_map()
        self.source_orchestrator = RunnableLambda(self._orchestrate_chain_inputs)

    def _set_subject_voice(self, subject: str):
        """Apply per-subject voice settings before speaking the response."""
        selected_subject = subject if subject in self.subject_voice_map else "english"
        selected_voice = self.subject_voice_map.get(selected_subject)
        if selected_voice:
            self.mouth.set_voice(selected_voice)

    def _orchestrate_chain_inputs(self, payload: dict) -> dict:
        """Resolve retrieval and prompt context for a single user question."""
        question = str(payload.get("question") or "").strip()
        subject = str(payload.get("subject") or route_subject(question))
        if subject not in self.memories:
            subject = "english"

        start_page, end_page = extract_page_range(question)
        source_documents = search_documents(
            question,
            subject=subject,
            start_page=start_page,
            end_page=end_page,
            k=5,
        )
        source = format_source(source_documents)

        return {
            "question": question,
            "subject": subject,
            "source": source,
            "page_range": describe_page_range(start_page, end_page),
            "memory_context": self._memory_context(subject),
        }

    def _remember_turn(self, user_input: str, assistant_output: str, subject: str):
        """Persist a completed turn into the selected specialist memory."""
        memory = self.memories.get(subject, self.memories["english"])
        user_text = truncate_text(user_input.strip(), MEMORY_MAX_CHARS_PER_MESSAGE)
        assistant_text = truncate_text(
            assistant_output.strip(), MEMORY_MAX_CHARS_PER_MESSAGE
        )
        memory.add_message(HumanMessage(content=user_text))
        memory.add_message(AIMessage(content=assistant_text))

    def _memory_context(self, subject: str) -> str:
        """Return compact memory text from the selected specialist history."""
        memory = self.memories.get(subject, self.memories["english"])
        recent_messages = memory.messages[-(MEMORY_MAX_TURNS * 2) :]
        if not recent_messages:
            return "No prior conversation yet."

        lines = []
        for message in recent_messages:
            if isinstance(message, HumanMessage):
                role = "Student"
            elif isinstance(message, AIMessage):
                role = "Tutor"
            else:
                role = "Message"

            text = truncate_text(
                str(message.content).strip(), MEMORY_MAX_CHARS_PER_MESSAGE
            )
            lines.append(f"{role}: {text}")

        return "\n".join(lines)

    def _listen_for_barge_in(self, stop_event: threading.Event, shared_state: dict):
        """Listen in short windows for interruption speech while tutor is speaking."""
        while not stop_event.is_set():
            heard = self.ears.listen(
                max_idle_seconds=BARGE_IN_IDLE_SECONDS,
                announce=False,
                silence_chunks_to_stop=8,
                beam_size=1,
            )
            if stop_event.is_set():
                return
            if heard and barge_in_passes_threshold(heard):
                # Stop speech output immediately so the user can take the floor.
                self.mouth.stop(wait=False)
                shared_state["text"] = heard.strip()
                stop_event.set()
                return

    def _stream_response_with_barge_in(
        self, user_input: str
    ) -> tuple[str, str | None, str]:
        """Stream model output, speak in chunks, and allow real-time interruption."""
        subject = route_subject(user_input)
        chain_inputs = self.source_orchestrator.invoke(
            {"question": user_input, "subject": subject}
        )
        active_subject = str(chain_inputs.get("subject") or "english")
        specialist_chain = self.specialist_chains.get(
            active_subject, self.specialist_chains["english"]
        )
        self._set_subject_voice(active_subject)

        print(f"[Agent] {active_subject.capitalize()} specialist")
        print(f"[Sources] {chain_inputs['page_range']}")
        print("AI: ", end="", flush=True)

        full_response = ""
        sentence_buffer = ""
        interrupted = {"text": None}
        barge_stop = threading.Event()
        barge_thread = threading.Thread(
            target=self._listen_for_barge_in,
            args=(barge_stop, interrupted),
            daemon=True,
        )
        barge_thread.start()

        try:
            for chunk in specialist_chain.stream(chain_inputs):
                if keyboard_quit_requested():
                    interrupted["text"] = "quit"
                    barge_stop.set()
                    break

                if barge_stop.is_set():
                    break

                content = chunk or ""
                if not content:
                    continue

                print(content, end="", flush=True)
                full_response += content
                sentence_buffer += content

                if any(p in sentence_buffer for p in [".", "!", "?", "\n"]):
                    clean_sentence = sentence_buffer.strip()
                    if clean_sentence:
                        self.mouth.speak_async(clean_sentence)
                    sentence_buffer = ""
        finally:
            barge_stop.set()
            barge_thread.join(timeout=0.2)

        if interrupted["text"]:
            self.mouth.stop(wait=False)
            interruption_text = interrupted["text"].strip()
            print(f"\n[Barge-in] {interruption_text}\n")
            return full_response, interruption_text, active_subject

        if sentence_buffer.strip():
            self.mouth.speak_async(sentence_buffer.strip())

        print("\n")
        self.mouth.wait_until_done()
        return full_response, None, active_subject

    def run(self):
        """Run the main conversational loop until the user exits."""
        print(f"--- Voice Tutor Agent Ready (Using {MODEL_NAME}) ---")
        print("Voice conversation is active. Say 'quit'/'stop' or press 'q' to end.\n")

        pending_user_input = None

        while True:
            try:
                if keyboard_quit_requested():
                    self.mouth.speak("Goodbye! Keep studying!")
                    print("Shutting down...")
                    break

                if pending_user_input:
                    user_input = pending_user_input
                    pending_user_input = None
                else:
                    user_input = self.ears.listen()
                    if not user_input or len(user_input) < 2:
                        print("No clear speech detected.")
                        continue

                if user_input.strip().lower() in VOICE_STOP_WORDS:
                    self.mouth.speak("Goodbye! Keep studying!")
                    print("Shutting down...")
                    break

                print(f"\nUser: {user_input}")
                full_response, interruption_text, active_subject = (
                    self._stream_response_with_barge_in(user_input)
                )

                if interruption_text:
                    if interruption_text.lower() in VOICE_STOP_WORDS:
                        self.mouth.speak("Goodbye! Keep studying!")
                        print("Shutting down...")
                        break

                    pending_user_input = interruption_text
                    continue

                if full_response.strip():
                    self._remember_turn(
                        user_input,
                        full_response.strip(),
                        active_subject,
                    )

            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as error:
                print(f"Error: {error}")
                try:
                    self.mouth.speak("I encountered an error. Please try again.")
                except Exception as tts_error:
                    print(f"[Mouth] TTS error while reporting failure: {tts_error}")
                continue
