# Configure offline-only mode BEFORE any other imports
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import threading
import time

try:
    import msvcrt
except ImportError:  # pragma: no cover - msvcrt is Windows-only
    msvcrt = None

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from chemistry_agent import build_chemistry_chain
from english_agent import build_english_chain
from history_agent import build_history_chain
from math_agent import build_math_chain
from router_agent import route_subject, route_subject_sticky

from conversation_utils import (
    barge_in_passes_threshold as _barge_in_passes_threshold,
    describe_page_range,
    extract_page_range,
    format_source,
    truncate_text,
)
from stt_module import SpeechToText
from tts_module import TextToSpeech

try:
    from tts_module import stop_all_tts
except ImportError:  # pragma: no cover - compatibility for lightweight test stubs
    def stop_all_tts(*, except_instance=None, wait: bool = False):
        return None
from vector import search_documents
from voice_config import load_subject_voice_map
from persistence import TutorPersistence
from note_taker_agent import QuestionNoteTakerAgent
from local_providers import create_chat_model
from settings_store import UserSettings, load_user_settings

MODEL_NAME = "llama3.1:8b"
DEFAULT_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "220"))
UI_TURN_TIMEOUT_SECONDS = float(os.getenv("VOICE_TUTOR_TURN_TIMEOUT_SECONDS", "60"))

VOICE_STOP_WORDS = {"quit", "stop", "exit", "bye"}
BARGE_IN_ENABLED = os.getenv("VOICE_BARGE_IN_ENABLED", "0").lower() in {
    "1",
    "true",
    "yes",
}
BARGE_IN_IDLE_SECONDS = float(os.getenv("VOICE_BARGE_IN_IDLE_SECONDS", "0.8"))
BARGE_IN_MIN_CHARS = int(os.getenv("VOICE_BARGE_IN_MIN_CHARS", "8"))
BARGE_IN_MIN_WORDS = int(os.getenv("VOICE_BARGE_IN_MIN_WORDS", "2"))
BARGE_JOIN_TIMEOUT_SECONDS = 1.2
PLAYBACK_POLL_SECONDS = 0.05
MEMORY_MAX_TURNS = 8
MEMORY_MAX_CHARS_PER_MESSAGE = 500
PERSISTED_MAX_MESSAGES_PER_SUBJECT = int(
    os.getenv("PERSISTED_MAX_MESSAGES_PER_SUBJECT", "400")
)
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
    cleaned = text.strip().lower()
    if cleaned in VOICE_STOP_WORDS:
        return True

    if len(cleaned.split()) < BARGE_IN_MIN_WORDS:
        return False

    return _barge_in_passes_threshold(text, VOICE_STOP_WORDS, BARGE_IN_MIN_CHARS)


class VoiceAgent:
    """Coordinate speech input, LLM responses, TTS output, and memory."""

    def __init__(
        self, settings: UserSettings | None = None, *, load_stt: bool = True
    ):
        """Initialize speech components, memory buffer, and the chat model."""
        self.settings = settings or load_user_settings()
        self.persistence = TutorPersistence(
            subjects=SUPPORTED_SUBJECTS,
            max_messages_per_subject=PERSISTED_MAX_MESSAGES_PER_SUBJECT,
        )
        self.note_taker = QuestionNoteTakerAgent(self.persistence)
        self.mouth = TextToSpeech(settings=self.settings)
        self.ears = (
            SpeechToText(tts_instance=self.mouth) if load_stt else None
        )  # Pass TTS reference to prevent self-pickup
        self.memories = {
            subject: InMemoryChatMessageHistory() for subject in SUPPORTED_SUBJECTS
        }
        self.llm = create_chat_model(self.settings, num_predict=DEFAULT_NUM_PREDICT)
        self.specialist_chains = {
            "history": build_history_chain(self.llm),
            "chemistry": build_chemistry_chain(self.llm),
            "math": build_math_chain(self.llm),
            "english": build_english_chain(self.llm),
        }
        self.subject_voice_map = load_subject_voice_map(
            backend=self.settings.tts_backend
        )
        self.source_orchestrator = RunnableLambda(self._orchestrate_chain_inputs)
        self.current_subject = self.settings.current_subject
        self.barge_in_enabled = BARGE_IN_ENABLED
        self._turn_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._restore_persisted_state()

    def cancel_current_turn(self):
        """Request cancellation of the active turn and stop any current speech."""
        self._cancel_event.set()
        try:
            self.mouth.stop(wait=False)
        except Exception:
            pass

    def _restore_persisted_state(self):
        """Restore conversation history and active subject from disk."""
        try:
            conversation_by_subject, persisted_subject = (
                self.persistence.load_conversation()
            )
        except Exception as error:
            print(f"[Persistence] Failed to load conversation history: {error}")
            return

        restored_messages = 0
        for subject in SUPPORTED_SUBJECTS:
            memory = self.memories.get(subject)
            if memory is None:
                continue

            records = conversation_by_subject.get(subject, [])
            for record in records:
                if not isinstance(record, dict):
                    continue

                role = str(record.get("role", "")).lower().strip()
                content = str(record.get("content", "")).strip()
                if not content:
                    continue

                if role == "human":
                    memory.add_message(HumanMessage(content=content))
                    restored_messages += 1
                elif role == "ai":
                    memory.add_message(AIMessage(content=content))
                    restored_messages += 1

        if persisted_subject in SUPPORTED_SUBJECTS:
            self.current_subject = persisted_subject

        if restored_messages > 0:
            restored_turns = restored_messages // 2
            print(
                f"[Persistence] Restored {restored_turns} prior turn(s). "
                f"Current subject: {self.current_subject}"
            )

    def _set_subject_voice(self, subject: str):
        """Apply per-subject voice settings before speaking the response."""
        selected_subject = subject if subject in self.subject_voice_map else "english"
        selected_voice = self.settings.selected_voice(selected_subject)
        if not selected_voice:
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
        retrieval_error = ""
        try:
            source_documents = search_documents(
                question,
                subject=subject,
                start_page=start_page,
                end_page=end_page,
                k=5,
                settings=self.settings,
            )
        except Exception as error:
            source_documents = []
            retrieval_error = str(error)
        source = format_source(source_documents)
        if retrieval_error:
            source = (
                "No source material is currently available because local retrieval "
                f"is still starting or failed: {retrieval_error}"
            )

        # Log which filters were applied
        filter_info = f"subject={subject}"
        if start_page is not None or end_page is not None:
            page_desc = describe_page_range(start_page, end_page)
            filter_info += f", {page_desc.lower()}"
        if source_documents:
            filter_info += f" (found {len(source_documents)} results)"

        return {
            "question": question,
            "subject": subject,
            "source": source,
            "source_cards": self._source_cards(source_documents),
            "page_range": describe_page_range(start_page, end_page),
            "memory_context": self._memory_context(subject),
            "retrieval_error": retrieval_error,
        }

    def _source_cards(self, source_documents) -> list[dict[str, str]]:
        """Return compact source metadata for the web UI."""
        cards = []
        for doc in source_documents or []:
            metadata = doc.metadata or {}
            snippet = " ".join(str(doc.page_content or "").split())[:220]
            cards.append(
                {
                    "source_file": str(metadata.get("source_file") or "unknown"),
                    "subject": str(metadata.get("subject") or "unknown"),
                    "page_label": str(metadata.get("page_label") or ""),
                    "title": str(metadata.get("title") or ""),
                    "snippet": snippet,
                }
            )
        return cards

    def _remember_turn(self, user_input: str, assistant_output: str, subject: str):
        """Persist a completed turn into the selected specialist memory."""
        memory = self.memories.get(subject, self.memories["english"])
        user_text = truncate_text(user_input.strip(), MEMORY_MAX_CHARS_PER_MESSAGE)
        assistant_text = truncate_text(
            assistant_output.strip(), MEMORY_MAX_CHARS_PER_MESSAGE
        )
        memory.add_message(HumanMessage(content=user_text))
        memory.add_message(AIMessage(content=assistant_text))

        try:
            self.persistence.append_turn(subject, user_text, assistant_text)
            changed = self.note_taker.persist_from_response(
                subject,
                assistant_output,
                source_prompt=user_input,
            )
            if changed:
                print(
                    f"[Persistence] Stored {len(changed)} question(s) under {subject}."
                )
        except Exception as error:
            print(f"[Persistence] Failed to persist turn/question data: {error}")

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
        if self.ears is None:
            return
        while not stop_event.is_set():
            try:
                heard = self.ears.listen(
                    max_idle_seconds=BARGE_IN_IDLE_SECONDS,
                    announce=False,
                    silence_chunks_to_stop=6,  # More responsive to short interruptions
                    beam_size=1,
                    stop_event=stop_event,
                )
                if stop_event.is_set():
                    return

                # Even single words or short utterances should be treated as interrupts
                if heard and len(heard.strip()) > 1:
                    if barge_in_passes_threshold(heard):
                        # Stop speech output immediately so the user can take the floor.
                        self.mouth.stop(wait=False)
                        shared_state["text"] = heard.strip()
                        stop_event.set()
                        return
            except Exception:
                # Continue listening even if there's an error
                if stop_event.is_set():
                    return
                continue

    def _stream_response_with_barge_in(
        self, user_input: str
    ) -> tuple[str, str | None, str]:
        """Stream model output, speak in chunks, and allow real-time interruption."""
        # Use sticky subject routing to maintain subject context
        subject, is_explicit_switch = route_subject_sticky(
            user_input, self.current_subject
        )

        # Only log subject switch if it's explicit (user clearly asked for a topic change)
        if is_explicit_switch:
            print(f"[Subject Switch] Changed from {self.current_subject} to {subject}")

        # Update current subject for sticky persistence
        self.current_subject = subject
        try:
            self.persistence.set_current_subject(subject)
        except Exception as error:
            print(f"[Persistence] Failed to update current subject: {error}")

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
        barge_thread = None
        stream_speech = self.mouth.backend != "pyttsx3"
        if self.barge_in_enabled:
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

                if stream_speech and any(p in sentence_buffer for p in [".", "!", "?", "\n"]):
                    clean_sentence = sentence_buffer.strip()
                    if clean_sentence:
                        self.mouth.speak_async(clean_sentence)
                    sentence_buffer = ""

            if stream_speech and sentence_buffer.strip() and not interrupted["text"]:
                self.mouth.speak_async(sentence_buffer.strip())
            elif (
                not stream_speech
                and full_response.strip()
                and not interrupted["text"]
            ):
                self.mouth.speak_async(full_response.strip())

            # Keep interruption listening active while queued TTS is still playing.
            while not interrupted["text"] and self.mouth.has_pending_audio():
                if keyboard_quit_requested():
                    interrupted["text"] = "quit"
                    barge_stop.set()
                    break
                if self.barge_in_enabled and barge_stop.is_set():
                    break
                time.sleep(PLAYBACK_POLL_SECONDS)
        finally:
            barge_stop.set()
            if barge_thread is not None:
                barge_thread.join(timeout=BARGE_JOIN_TIMEOUT_SECONDS)

        if self.barge_in_enabled and interrupted["text"]:
            self.mouth.stop(wait=False)
            interruption_text = interrupted["text"].strip()
            print(f"\n[Barge-in] {interruption_text}\n")
            return full_response, interruption_text, active_subject

        print("\n")
        return full_response, None, active_subject

    def stream_ui_turn(
        self,
        user_input: str,
        *,
        speak: bool = True,
        stop_event: threading.Event | None = None,
        timeout_seconds: float | None = None,
    ):
        """Yield structured events for the browser UI while streaming a turn."""
        with self._turn_lock:
            self._cancel_event.clear()
            started_at = time.monotonic()
            timeout = (
                UI_TURN_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
            )

            def stop_requested() -> bool:
                return self._cancel_event.is_set() or (
                    stop_event is not None and stop_event.is_set()
                )

            def timed_out() -> bool:
                return timeout > 0 and (time.monotonic() - started_at) >= timeout

            def stop_message(reason: str) -> dict[str, str]:
                if reason == "timeout":
                    message = (
                        f"I stopped because this response took longer than "
                        f"{int(timeout)} seconds."
                    )
                else:
                    message = "Stopped."
                return {
                    "type": "stopped",
                    "reason": reason,
                    "message": message,
                }

            if speak:
                stop_all_tts(except_instance=self.mouth, wait=False)
                self.mouth.stop(wait=False, release_owner=False)

            subject, is_explicit_switch = route_subject_sticky(
                user_input, self.current_subject
            )
            self.current_subject = subject
            try:
                self.persistence.set_current_subject(subject)
            except Exception as error:
                print(f"[Persistence] Failed to update current subject: {error}")

            chain_inputs = self.source_orchestrator.invoke(
                {"question": user_input, "subject": subject}
            )
            active_subject = str(chain_inputs.get("subject") or "english")
            specialist_chain = self.specialist_chains.get(
                active_subject, self.specialist_chains["english"]
            )
            self._set_subject_voice(active_subject)

            yield {
                "type": "subject",
                "subject": active_subject,
                "explicit_switch": is_explicit_switch,
            }
            yield {
                "type": "sources",
                "page_range": chain_inputs.get("page_range", ""),
                "sources": chain_inputs.get("source_cards", []),
            }

            if stop_requested():
                self.mouth.stop(wait=False)
                yield stop_message("cancelled")
                return

            full_response = ""
            sentence_buffer = ""
            stream_speech = self.mouth.backend != "pyttsx3"
            try:
                for chunk in specialist_chain.stream(chain_inputs):
                    if stop_requested():
                        self.mouth.stop(wait=False)
                        yield stop_message("cancelled")
                        return
                    if timed_out():
                        self._cancel_event.set()
                        self.mouth.stop(wait=False)
                        yield stop_message("timeout")
                        return

                    content = chunk or ""
                    if not content:
                        continue
                    full_response += content
                    sentence_buffer += content
                    yield {"type": "token", "content": content}

                    if speak and stream_speech and any(
                        p in sentence_buffer for p in [".", "!", "?", "\n"]
                    ):
                        clean_sentence = sentence_buffer.strip()
                        if clean_sentence:
                            self.mouth.speak_async(clean_sentence)
                        sentence_buffer = ""

                if speak and stream_speech and sentence_buffer.strip():
                    self.mouth.speak_async(sentence_buffer.strip())
                elif speak and not stream_speech and full_response.strip():
                    self.mouth.speak_async(full_response.strip())

                while speak and self.mouth.has_pending_audio():
                    if stop_requested():
                        self.mouth.stop(wait=False)
                        yield stop_message("cancelled")
                        return
                    if timed_out():
                        self._cancel_event.set()
                        self.mouth.stop(wait=False)
                        yield stop_message("timeout")
                        return
                    time.sleep(PLAYBACK_POLL_SECONDS)
            except Exception as error:
                if stop_requested():
                    yield stop_message("cancelled")
                    return
                yield {"type": "error", "message": str(error)}
                return

            if stop_requested():
                self.mouth.stop(wait=False)
                yield stop_message("cancelled")
                return
            if timed_out():
                self._cancel_event.set()
                self.mouth.stop(wait=False)
                yield stop_message("timeout")
                return

            if full_response.strip():
                self._remember_turn(user_input, full_response, active_subject)

            yield {
                "type": "done",
                "response": full_response,
                "subject": active_subject,
            }

    def run(self):
        """Run the main conversational loop until the user exits."""
        print(f"--- Voice Tutor Agent Ready (Using {MODEL_NAME}) ---")
        print("Voice conversation is active. Say 'quit'/'stop' or press 'q' to end.")
        print(
            f"[Barge-in] {'Enabled' if self.barge_in_enabled else 'Disabled'} "
            f"(set VOICE_BARGE_IN_ENABLED=1 to enable).\n"
        )

        pending_user_input = None
        empty_listen_count = 0
        use_keyboard_fallback = False
        voice_prompt_shown = False
        keyboard_prompt_shown = False

        try:
            while True:
                try:
                    if keyboard_quit_requested():
                        self.mouth.speak("Goodbye! Keep studying!")
                        print("Shutting down...")
                        break

                    if pending_user_input:
                        user_input = pending_user_input
                        pending_user_input = None
                        voice_prompt_shown = False
                        keyboard_prompt_shown = False
                    else:
                        if use_keyboard_fallback:
                            # Fallback to keyboard input if audio keeps failing
                            if not keyboard_prompt_shown:
                                print("\n[Ready] Listening via keyboard input.")
                                keyboard_prompt_shown = True
                            user_input = input(
                                "\n[Keyboard Input] Enter your question: "
                            ).strip()
                        else:
                            if not voice_prompt_shown:
                                print(
                                    "\n[Ready] App is ready. [Listening] Speak now..."
                                )
                                voice_prompt_shown = True
                            user_input = self.ears.listen(announce=False)
                            # Track empty returns to detect missing audio device
                            if not user_input:
                                empty_listen_count += 1
                                if empty_listen_count > 5:
                                    print(
                                        "\n[Warning] No audio input detected after 5 attempts."
                                    )
                                    print("[Switching to keyboard input mode]")
                                    use_keyboard_fallback = True
                                    voice_prompt_shown = False
                                    keyboard_prompt_shown = False
                                    continue
                            else:
                                empty_listen_count = 0
                                voice_prompt_shown = False
                                keyboard_prompt_shown = False

                        if not user_input or len(user_input) < 2:
                            continue

                    if user_input.strip().lower() in VOICE_STOP_WORDS:
                        self.mouth.speak("Goodbye! Keep studying!")
                        print("Shutting down...")
                        break

                    print(f"\nUser: {user_input}")
                    print("[Processing] Thinking...")
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
        finally:
            # Ensure background playback is fully stopped before interpreter teardown.
            try:
                self.mouth.stop(wait=True)
            except Exception as error:
                print(f"[Mouth] Shutdown cleanup error: {error}")
