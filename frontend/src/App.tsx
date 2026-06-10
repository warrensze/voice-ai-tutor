import {
  CheckCircle2,
  FileText,
  Library,
  Loader2,
  Mic,
  RefreshCw,
  Send,
  Settings,
  SlidersHorizontal,
  Square,
  Trash2,
  Upload,
  Volume2,
  Wifi,
  WifiOff,
  X
} from "lucide-react";
import { FormEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";

type TutorSettings = {
  llm_provider: "llamacpp" | "ollama";
  embedding_provider: "llamacpp" | "ollama";
  tts_backend: "piper" | "kokoro" | "pyttsx3";
  stt_provider: "faster-whisper" | "whispercpp";
  current_subject: Subject;
  speak_responses: boolean;
  ollama_base_url: string;
  ollama_chat_model: string;
  ollama_embedding_model: string;
  llamacpp_chat_base_url: string;
  llamacpp_chat_model: string;
  llamacpp_embedding_base_url: string;
  llamacpp_embedding_model: string;
  piper_voice: string;
  piper_data_dir: string;
  kokoro_voice: string;
  kokoro_device: "auto" | "cpu" | "cuda";
  kokoro_allow_cpu: boolean;
  pyttsx3_voice: string;
  stt_language: string;
  faster_whisper_model: string;
  faster_whisper_device: "auto" | "cpu" | "cuda";
  faster_whisper_compute_type: string;
  whispercpp_binary_path: string;
  whispercpp_model_path: string;
  whispercpp_language: string;
  subject_voices: Record<string, Record<Subject, string>>;
};

type Subject = "english" | "history" | "chemistry" | "math";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  subject?: Subject;
};

type SourceCard = {
  source_file: string;
  subject: string;
  page_label: string;
  title: string;
  snippet: string;
};

type LibraryAsset = {
  id: string;
  title: string;
  subject: Subject;
  source_file: string;
  file_type: string;
  status: string;
  chunk_count: number;
  error: string;
  duplicate?: boolean;
};

type BuiltInSource = {
  title: string;
  subject: Subject;
  source_file: string;
  file_type: string;
  status: string;
  chunk_count: number;
  has_ocr_text: boolean;
};

type StatusPayload = {
  providers: {
    llm_provider: string;
    chat_endpoint: string;
    chat_model: string;
    embedding_provider: string;
    embedding_endpoint: string;
    embedding_model: string;
    tts_backend: string;
    chat_health: { ok: boolean; error?: string };
    embedding_health: { ok: boolean; error?: string };
    tts_health?: {
      ok: boolean;
      backend: string;
      voice: string;
      device?: string;
      strict: boolean;
      error?: string;
    };
    stt_health?: {
      ok: boolean;
      provider: string;
      model: string;
      device: string;
      strict: boolean;
      binary?: string;
      ffmpeg?: string;
      error?: string;
    };
    llamacpp_bootstrap?: {
      status: string;
      message: string;
      chat?: { status: string; model: string; endpoint: string; message: string };
      embedding?: { status: string; model: string; endpoint: string; message: string };
    };
  };
  vector: {
    total_chunks: number;
    builtin_sources: BuiltInSource[];
    embedding_provider: string;
    embedding_model: string;
  };
};

type VoiceOption = {
  id: string;
  label: string;
  available?: boolean;
  path?: string;
  error?: string;
};

const subjects: Subject[] = ["english", "history", "chemistry", "math"];

const defaultSettings: TutorSettings = {
  llm_provider: "llamacpp",
  embedding_provider: "llamacpp",
  tts_backend: "piper",
  stt_provider: "faster-whisper",
  current_subject: "english",
  speak_responses: true,
  ollama_base_url: "http://127.0.0.1:11434",
  ollama_chat_model: "llama3.1:8b",
  ollama_embedding_model: "mxbai-embed-large",
  llamacpp_chat_base_url: "http://127.0.0.1:8080/v1",
  llamacpp_chat_model: "Qwen/Qwen3-8B-GGUF:Q4_K_M",
  llamacpp_embedding_base_url: "http://127.0.0.1:8081/v1",
  llamacpp_embedding_model: "nomic-ai/nomic-embed-text-v1.5-GGUF:Q4_K_M",
  piper_voice: "en_US-lessac-medium",
  piper_data_dir: "models/piper",
  kokoro_voice: "af_heart",
  kokoro_device: "auto",
  kokoro_allow_cpu: true,
  pyttsx3_voice: "",
  stt_language: "en",
  faster_whisper_model: "base.en",
  faster_whisper_device: "auto",
  faster_whisper_compute_type: "auto",
  whispercpp_binary_path: "whisper-cli",
  whispercpp_model_path: "models/stt/whisper.cpp/ggml-base.en.bin",
  whispercpp_language: "en",
  subject_voices: {
    kokoro: {
      history: "am_adam",
      chemistry: "bf_alice",
      math: "af_sky",
      english: "af_heart"
    },
    piper: {
      history: "en_US-lessac-medium",
      chemistry: "en_US-lessac-medium",
      math: "en_US-lessac-medium",
      english: "en_US-lessac-medium"
    },
    pyttsx3: {
      history: "",
      chemistry: "",
      math: "",
      english: ""
    }
  }
};

function newId() {
  return Math.random().toString(36).slice(2);
}

function sttProviderLabel(provider: string) {
  return provider === "whispercpp" ? "whisper.cpp" : provider;
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: init?.body instanceof FormData
      ? init.headers
      : { "Content-Type": "application/json", ...(init?.headers || {}) }
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json() as Promise<T>;
}

export default function App() {
  const [settings, setSettings] = useState<TutorSettings>(defaultSettings);
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [voiceOptions, setVoiceOptions] = useState<Record<string, VoiceOption[]>>({});
  const [messages, setMessages] = useState<Message[]>([]);
  const [sources, setSources] = useState<SourceCard[]>([]);
  const [assets, setAssets] = useState<LibraryAsset[]>([]);
  const [draft, setDraft] = useState("");
  const [appState, setAppState] = useState<"idle" | "listening" | "transcribing" | "thinking" | "speaking">("idle");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [uploadSubject, setUploadSubject] = useState<Subject>("english");
  const wsRef = useRef<WebSocket | null>(null);
  const turnActiveRef = useRef(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    refreshAll();
    const interval = window.setInterval(() => {
      refreshLibrary();
      refreshStatus();
    }, 2500);
    return () => window.clearInterval(interval);
  }, []);

  const ttsHealth = status?.providers.tts_health;
  const sttHealth = status?.providers.stt_health;
  const ttsOk = Boolean(ttsHealth?.ok);
  const providerOk = Boolean(
    status?.providers.chat_health.ok &&
    status.providers.embedding_health.ok &&
    (!settings.speak_responses || ttsOk) &&
    sttHealth?.ok !== false
  );
  const builtInSources = status?.vector?.builtin_sources || [];
  const llamaStatus = status?.providers.llamacpp_bootstrap;
  const micLabel = useMemo(() => {
    if (appState === "listening") return "Listening";
    if (appState === "transcribing") return "Transcribing";
    if (appState === "thinking") return "Thinking";
    if (appState === "speaking") return "Speaking";
    return "Ready";
  }, [appState]);

  async function refreshAll() {
    const payload = await api<{ settings: TutorSettings }>("/api/settings");
    setSettings({ ...defaultSettings, ...payload.settings });
    await Promise.all([refreshStatus(), refreshLibrary(), refreshVoices()]);
  }

  async function refreshStatus() {
    try {
      setStatus(await api<StatusPayload>("/api/status"));
    } catch {
      setStatus(null);
    }
  }

  async function refreshLibrary() {
    try {
      const payload = await api<{ assets: LibraryAsset[] }>("/api/library");
      setAssets(payload.assets);
    } catch {
      setAssets([]);
    }
  }

  async function refreshVoices() {
    try {
      const payload = await api<{ voices: Record<string, VoiceOption[]> }>("/api/voices");
      setVoiceOptions(payload.voices || {});
    } catch {
      setVoiceOptions({});
    }
  }

  async function saveSettings(patch: Partial<TutorSettings>) {
    const localRuntimeKeys: Array<keyof TutorSettings> = [
      "tts_backend",
      "speak_responses",
      "piper_voice",
      "piper_data_dir",
      "kokoro_voice",
      "kokoro_device",
      "kokoro_allow_cpu",
      "pyttsx3_voice",
      "subject_voices",
      "stt_provider",
      "stt_language",
      "faster_whisper_model",
      "faster_whisper_device",
      "faster_whisper_compute_type",
      "whispercpp_binary_path",
      "whispercpp_model_path",
      "whispercpp_language"
    ];
    const shouldStopSpeech =
      localRuntimeKeys.some((key) => patch[key] !== undefined && patch[key] !== settings[key]) ||
      (patch.speak_responses === false && settings.speak_responses);
    if (shouldStopSpeech) {
      await api("/api/voice/stop", { method: "POST", body: JSON.stringify({}) });
    }

    const next = { ...settings, ...patch };
    setSettings(next);
    const payload = await api<{ settings: TutorSettings }>("/api/settings", {
      method: "PUT",
      body: JSON.stringify(patch)
    });
    setSettings({ ...defaultSettings, ...payload.settings });
    refreshStatus();
    refreshVoices();
  }

  function selectedSubjectVoice() {
    const backend = settings.tts_backend;
    return (
      settings.subject_voices?.[backend]?.[settings.current_subject] ||
      (backend === "piper" ? settings.piper_voice : backend === "kokoro" ? settings.kokoro_voice : settings.pyttsx3_voice)
    );
  }

  function voicePatch(backend: TutorSettings["tts_backend"], voiceId: string, allSubjects: boolean) {
    const patch: Partial<TutorSettings> = {};
    if (backend === "piper") patch.piper_voice = voiceId;
    if (backend === "kokoro") patch.kokoro_voice = voiceId;
    if (backend === "pyttsx3") patch.pyttsx3_voice = voiceId;

    if (allSubjects) {
      patch.subject_voices = {
        ...settings.subject_voices,
        [backend]: subjects.reduce(
          (acc, subject) => ({ ...acc, [subject]: voiceId }),
          {} as Record<Subject, string>
        )
      };
    } else {
      patch.subject_voices = {
        ...settings.subject_voices,
        [backend]: {
          ...(settings.subject_voices?.[backend] || {}),
          [settings.current_subject]: voiceId
        }
      };
    }
    return patch;
  }

  async function saveSubjectVoice(voiceId: string) {
    const backend = settings.tts_backend;
    await saveSettings(voicePatch(backend, voiceId, false));
  }

  async function saveVoiceForAllSubjects(voiceId: string) {
    const backend = settings.tts_backend;
    await saveSettings(voicePatch(backend, voiceId, true));
  }

  async function applyVoiceToAllSubjects() {
    await saveVoiceForAllSubjects(selectedSubjectVoice());
  }

  async function sendQuestion(question: string) {
    const trimmed = question.trim();
    if (!trimmed || turnActiveRef.current) return;

    const userId = newId();
    const assistantId = newId();
    turnActiveRef.current = true;
    setMessages((current) => [
      ...current,
      { id: userId, role: "user", content: trimmed, subject: settings.current_subject },
      { id: assistantId, role: "assistant", content: "", subject: settings.current_subject }
    ]);
    setDraft("");
    setSources([]);
    setAppState("thinking");

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${window.location.host}/ws/chat`);
    wsRef.current = socket;

    socket.onopen = () => {
      socket.send(JSON.stringify({
        question: trimmed,
        subject: settings.current_subject,
        speak: settings.speak_responses
      }));
    };
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "subject") {
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId ? { ...message, subject: payload.subject } : message
          )
        );
      }
      if (payload.type === "sources") {
        setSources(payload.sources || []);
      }
      if (payload.type === "tts_status" && payload.ok === false) {
        setAppState("thinking");
        refreshStatus();
      }
      if (payload.type === "token") {
        setAppState(settings.speak_responses && ttsOk ? "speaking" : "thinking");
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId
              ? { ...message, content: message.content + payload.content }
              : message
          )
        );
      }
      if (payload.type === "done") {
        turnActiveRef.current = false;
        setAppState("idle");
        socket.close();
        refreshLibrary();
      }
      if (payload.type === "error") {
        turnActiveRef.current = false;
        setAppState("idle");
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId
              ? { ...message, content: payload.message || "Something went wrong." }
              : message
          )
        );
        socket.close();
      }
      if (payload.type === "stopped") {
        turnActiveRef.current = false;
        setAppState("idle");
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId && !message.content
              ? { ...message, content: payload.message || "Stopped." }
              : message
          )
        );
        socket.close();
      }
    };
    socket.onerror = () => {
      turnActiveRef.current = false;
      setAppState("idle");
    };
    socket.onclose = () => {
      turnActiveRef.current = false;
    };
  }

  async function submitDraft(event: FormEvent) {
    event.preventDefault();
    await sendQuestion(draft);
  }

  async function toggleRecording() {
    if (appState === "listening") {
      recorderRef.current?.stop();
      return;
    }
    if (sttHealth && !sttHealth.ok) {
      setMessages((current) => [
        ...current,
        {
          id: newId(),
          role: "assistant",
          content: `STT unavailable: ${sttHealth.error || "Selected local speech recognizer is not ready."}`,
          subject: settings.current_subject
        }
      ]);
      return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    audioChunksRef.current = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) audioChunksRef.current.push(event.data);
    };
    recorder.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop());
      setAppState("transcribing");
      const blob = new Blob(audioChunksRef.current, { type: recorder.mimeType || "audio/webm" });
      const form = new FormData();
      form.append("file", blob, "recording.webm");
      try {
        const result = await api<{ ok?: boolean; text: string; provider?: string; error?: string }>("/api/transcribe", {
          method: "POST",
          body: form
        });
        if (result.ok === false) {
          setMessages((current) => [
            ...current,
            {
              id: newId(),
              role: "assistant",
              content: `STT unavailable: ${result.error || "Transcription failed."}`,
              subject: settings.current_subject
            }
          ]);
          setAppState("idle");
          refreshStatus();
          return;
        }
        if (result.text) {
          await sendQuestion(result.text);
        } else {
          setAppState("idle");
        }
      } catch (error) {
        setMessages((current) => [
          ...current,
          {
            id: newId(),
            role: "assistant",
            content: `Transcription failed: ${error instanceof Error ? error.message : "Unknown local STT error."}`,
            subject: settings.current_subject
          }
        ]);
        setAppState("idle");
      }
    };
    recorderRef.current = recorder;
    recorder.start();
    setAppState("listening");
  }

  async function stopEverything() {
    turnActiveRef.current = false;
    setAppState("idle");
    try {
      await api("/api/voice/stop", { method: "POST", body: JSON.stringify({}) });
    } catch {
      // The local backend may already be busy stopping; the UI should still reset.
    }
    try {
      wsRef.current?.send(JSON.stringify({ type: "stop" }));
    } catch {
      // Ignore a socket that has already closed.
    }
    wsRef.current?.close();
    recorderRef.current?.stop();
    setAppState("idle");
  }

  async function uploadFiles(files: FileList | File[]) {
    for (const file of Array.from(files)) {
      const form = new FormData();
      form.append("file", file);
      form.append("subject", uploadSubject);
      form.append("title", file.name.replace(/\.[^.]+$/, ""));
      await api<LibraryAsset>("/api/library/assets", { method: "POST", body: form });
    }
    refreshLibrary();
  }

  async function reindexAsset(id: string) {
    await api(`/api/library/assets/${id}/reindex`, { method: "POST", body: JSON.stringify({}) });
    refreshLibrary();
  }

  async function removeAsset(id: string) {
    await api(`/api/library/assets/${id}`, { method: "DELETE" });
    refreshLibrary();
  }

  async function testVoice() {
    await api("/api/voice/stop", { method: "POST", body: JSON.stringify({}) });
    await api("/api/voice/test", { method: "POST", body: JSON.stringify({}) });
    refreshStatus();
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="product-name">Voice AI Tutor</div>
          <div className="local-line">
            {providerOk ? <Wifi size={14} /> : <WifiOff size={14} />}
            <span>Local only</span>
            <span>{settings.llm_provider}</span>
            <span>{settings.tts_backend}</span>
            <span>{sttProviderLabel(settings.stt_provider)}</span>
          </div>
        </div>
        <div className="topbar-actions">
          <button className="icon-button" onClick={() => setLibraryOpen(true)} title="Study library">
            <Library size={20} />
          </button>
          <button className="icon-button" onClick={() => setSettingsOpen(true)} title="Settings">
            <Settings size={20} />
          </button>
        </div>
      </header>

      <main className="workspace">
        <section className="chat-panel">
          <div className="subject-row">
            {subjects.map((subject) => (
              <button
                key={subject}
                className={subject === settings.current_subject ? "subject active" : "subject"}
                onClick={() => saveSettings({ current_subject: subject })}
              >
                {subject}
              </button>
            ))}
          </div>

          <div className="messages">
            {messages.length === 0 ? (
              <div className="empty-state">
                <Volume2 size={28} />
                <p>Ask a question, practice a topic, or add a study asset from the library.</p>
              </div>
            ) : (
              messages.map((message) => (
                <article key={message.id} className={`message ${message.role}`}>
                  <div className="message-meta">{message.role === "user" ? "You" : `${message.subject || "Tutor"} tutor`}</div>
                  <div>{message.content || (message.role === "assistant" ? "..." : "")}</div>
                </article>
              ))
            )}
          </div>

          <form className="composer" onSubmit={submitDraft}>
            <input
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Type a question or use the mic"
            />
            <button className="send-button" type="submit" disabled={!draft.trim() || turnActiveRef.current}>
              <Send size={18} />
            </button>
          </form>
        </section>

        <aside className="voice-panel">
          <div className={`mic-orb ${appState}`}>
            <button onClick={toggleRecording} disabled={appState === "thinking" || appState === "speaking"}>
              {appState === "listening" ? <Square size={34} /> : <Mic size={38} />}
            </button>
          </div>
          <div className="mic-label">{micLabel}</div>
          <div className="voice-picker-card">
            <div className="voice-picker-head">
              <span>Voice</span>
              <strong>{settings.tts_backend}</strong>
            </div>
            <VoiceSelect
              label="Selected voice"
              value={selectedSubjectVoice()}
              options={voiceOptions[settings.tts_backend] || []}
              onChange={saveVoiceForAllSubjects}
            />
            <div className="voice-picker-actions">
              <span>All subjects</span>
              <button className="secondary-action compact" onClick={testVoice}>
                <Volume2 size={16} />
                Test
              </button>
            </div>
          </div>
          {sttHealth && (
            <div className={`model-status ${sttHealth.ok ? "ready" : "error"}`}>
              <div className="model-status-head">
                <span>Speech Input</span>
                <strong>{sttHealth.ok ? "ready" : "unavailable"}</strong>
              </div>
              <div className="model-status-line">
                {sttProviderLabel(sttHealth.provider)} · {sttHealth.model || "default"}
                {sttHealth.device ? ` · ${sttHealth.device}` : ""}
              </div>
              {!sttHealth.ok && sttHealth.error && (
                <p>{sttHealth.error}</p>
              )}
            </div>
          )}
          {settings.llm_provider === "llamacpp" && (
            <div className={`model-status ${llamaStatus?.status || "idle"}`}>
              <div className="model-status-head">
                <span>Local Model</span>
                <strong>{llamaStatus?.status || "idle"}</strong>
              </div>
              <div className="model-status-line">
                Qwen3 · {llamaStatus?.chat?.status || "waiting"}
              </div>
              <div className="model-status-line">
                Nomic Embed · {llamaStatus?.embedding?.status || "waiting"}
              </div>
              {llamaStatus?.message && (
                <p>{llamaStatus.message}</p>
              )}
            </div>
          )}
          {settings.speak_responses && ttsHealth && (
            <div className={`model-status ${ttsHealth.ok ? "ready" : "error"}`}>
              <div className="model-status-head">
                <span>Voice</span>
                <strong>{ttsHealth.ok ? "ready" : "unavailable"}</strong>
              </div>
              <div className="model-status-line">
                {ttsHealth.backend} · {ttsHealth.voice || "default"}
                {ttsHealth.device ? ` · ${ttsHealth.device}` : ""}
              </div>
              {!ttsHealth.ok && ttsHealth.error && (
                <p>{ttsHealth.error}</p>
              )}
            </div>
          )}
          <button className="stop-button" onClick={stopEverything}>
            <Square size={16} />
            Stop
          </button>

          <div className="source-panel">
            <div className="panel-title">Sources</div>
            {sources.length === 0 ? (
              <p className="muted">Sources appear here after a response.</p>
            ) : (
              sources.map((source, index) => (
                <div className="source-card" key={`${source.source_file}-${index}`}>
                  <div className="source-title">
                    <FileText size={15} />
                    <span>{source.title || source.source_file}</span>
                  </div>
                  <div className="source-meta">
                    {source.subject}
                    {source.page_label ? ` · page ${source.page_label}` : ""}
                  </div>
                  <p>{source.snippet}</p>
                </div>
              ))
            )}
          </div>
        </aside>
      </main>

      {settingsOpen && (
        <Drawer title="Settings" onClose={() => setSettingsOpen(false)}>
          <Segmented
            label="LLM"
            value={settings.llm_provider}
            options={["llamacpp", "ollama"]}
            onChange={(value) => saveSettings({ llm_provider: value as TutorSettings["llm_provider"], embedding_provider: value as TutorSettings["embedding_provider"] })}
          />
          <Segmented
            label="TTS"
            value={settings.tts_backend}
            options={["piper", "kokoro", "pyttsx3"]}
            onChange={(value) => saveSettings({ tts_backend: value as TutorSettings["tts_backend"] })}
          />
          <Segmented
            label="STT"
            value={settings.stt_provider}
            options={["faster-whisper", "whispercpp"]}
            optionLabels={{ whispercpp: "whisper.cpp" }}
            onChange={(value) => saveSettings({ stt_provider: value as TutorSettings["stt_provider"] })}
          />
          {settings.stt_provider === "faster-whisper" ? (
            <>
              <SettingsField label="faster-whisper model" value={settings.faster_whisper_model} onChange={(value) => saveSettings({ faster_whisper_model: value })} />
              <Segmented
                label="faster-whisper device"
                value={settings.faster_whisper_device}
                options={["auto", "cpu", "cuda"]}
                onChange={(value) => saveSettings({ faster_whisper_device: value as TutorSettings["faster_whisper_device"] })}
              />
              <SettingsField label="faster-whisper compute" value={settings.faster_whisper_compute_type} onChange={(value) => saveSettings({ faster_whisper_compute_type: value })} />
              <SettingsField label="STT language" value={settings.stt_language} onChange={(value) => saveSettings({ stt_language: value })} />
            </>
          ) : (
            <>
              <SettingsField label="whisper.cpp binary" value={settings.whispercpp_binary_path} onChange={(value) => saveSettings({ whispercpp_binary_path: value })} />
              <SettingsField label="whisper.cpp model" value={settings.whispercpp_model_path} onChange={(value) => saveSettings({ whispercpp_model_path: value })} />
              <SettingsField label="whisper.cpp language" value={settings.whispercpp_language} onChange={(value) => saveSettings({ whispercpp_language: value })} />
            </>
          )}
          <VoiceSelect
            label={`${settings.current_subject} voice`}
            value={selectedSubjectVoice()}
            options={voiceOptions[settings.tts_backend] || []}
            onChange={saveSubjectVoice}
          />
          {settings.tts_backend === "kokoro" && (
            <>
              <Segmented
                label="Kokoro device"
                value={settings.kokoro_device}
                options={["auto", "cpu", "cuda"]}
                onChange={(value) => saveSettings({ kokoro_device: value as TutorSettings["kokoro_device"] })}
              />
              <label className="toggle-line">
                <input
                  type="checkbox"
                  checked={settings.kokoro_allow_cpu}
                  onChange={(event) => saveSettings({ kokoro_allow_cpu: event.target.checked })}
                />
                Allow CPU Kokoro
              </label>
            </>
          )}
          <button className="secondary-action" onClick={applyVoiceToAllSubjects}>
            <CheckCircle2 size={16} />
            Use voice for all subjects
          </button>
          <label className="toggle-line">
            <input
              type="checkbox"
              checked={settings.speak_responses}
              onChange={(event) => saveSettings({ speak_responses: event.target.checked })}
            />
            Speak tutor responses
          </label>
          <SettingsField label="llama.cpp chat URL" value={settings.llamacpp_chat_base_url} onChange={(value) => saveSettings({ llamacpp_chat_base_url: value })} />
          <SettingsField label="llama.cpp chat model" value={settings.llamacpp_chat_model} onChange={(value) => saveSettings({ llamacpp_chat_model: value })} />
          <SettingsField label="llama.cpp embed URL" value={settings.llamacpp_embedding_base_url} onChange={(value) => saveSettings({ llamacpp_embedding_base_url: value })} />
          <SettingsField label="llama.cpp embed model" value={settings.llamacpp_embedding_model} onChange={(value) => saveSettings({ llamacpp_embedding_model: value })} />
          <SettingsField label="Ollama URL" value={settings.ollama_base_url} onChange={(value) => saveSettings({ ollama_base_url: value })} />
          <SettingsField label="Piper voice" value={settings.piper_voice} onChange={(value) => saveSettings({ piper_voice: value })} />
          <SettingsField label="Piper data dir" value={settings.piper_data_dir} onChange={(value) => saveSettings({ piper_data_dir: value })} />
          <button className="secondary-action" onClick={testVoice}>
            <Volume2 size={16} />
            Test voice
          </button>
        </Drawer>
      )}

      {libraryOpen && (
        <Drawer title="Study Library" onClose={() => setLibraryOpen(false)}>
          <div className="upload-box" onDrop={(event) => {
            event.preventDefault();
            uploadFiles(event.dataTransfer.files);
          }} onDragOver={(event) => event.preventDefault()}>
            <Upload size={24} />
            <span>Drop PDFs, EPUBs, or OCR text files</span>
            <input type="file" multiple accept=".pdf,.epub,.txt" onChange={(event) => event.target.files && uploadFiles(event.target.files)} />
          </div>
          <div className="field">
            <span>Upload subject</span>
            <select value={uploadSubject} onChange={(event) => setUploadSubject(event.target.value as Subject)}>
              {subjects.map((subject) => <option key={subject}>{subject}</option>)}
            </select>
          </div>
          <div className="library-section-title">Built-in Sources</div>
          <div className="asset-list">
            {builtInSources.length === 0 ? (
              <div className="asset empty-asset">
                <div>
                  <div className="asset-title">No built-in sources found</div>
                  <div className="asset-meta">Add PDFs to the project assets folder.</div>
                </div>
              </div>
            ) : (
              builtInSources.map((source) => (
                <div className="asset" key={`${source.subject}-${source.source_file}`}>
                  <div>
                    <div className="asset-title">{source.title || source.source_file}</div>
                    <div className="asset-meta">
                      {source.subject} · {source.file_type} · {source.status} · {source.chunk_count} chunks
                      {source.has_ocr_text ? " · OCR sidecar" : ""}
                    </div>
                  </div>
                  <div className={`status-pill ${source.status}`}>
                    {source.status}
                  </div>
                </div>
              ))
            )}
          </div>
          <div className="library-section-title">Added By You</div>
          <div className="asset-list">
            {assets.length === 0 ? (
              <div className="asset empty-asset">
                <div>
                  <div className="asset-title">No uploaded study assets yet</div>
                  <div className="asset-meta">Drop PDFs, EPUBs, or OCR text files above.</div>
                </div>
              </div>
            ) : assets.map((asset) => (
              <div className="asset" key={asset.id}>
                <div>
                  <div className="asset-title">{asset.title}</div>
                  <div className="asset-meta">{asset.subject} · {asset.file_type} · {asset.status} · {asset.chunk_count} chunks</div>
                  {asset.error && <div className="asset-error">{asset.error}</div>}
                </div>
                <div className="asset-actions">
                  <button onClick={() => reindexAsset(asset.id)} title="Reindex">
                    {asset.status === "embedding" || asset.status === "extracting" ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
                  </button>
                  <button onClick={() => removeAsset(asset.id)} title="Remove">
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </Drawer>
      )}
    </div>
  );
}

function Drawer({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  return (
    <div className="drawer-backdrop">
      <aside className="drawer">
        <div className="drawer-head">
          <div className="drawer-title">
            <SlidersHorizontal size={18} />
            {title}
          </div>
          <button className="icon-button" onClick={onClose}>
            <X size={18} />
          </button>
        </div>
        {children}
      </aside>
    </div>
  );
}

function Segmented({
  label,
  value,
  options,
  optionLabels = {},
  onChange
}: {
  label: string;
  value: string;
  options: string[];
  optionLabels?: Record<string, string>;
  onChange: (value: string) => void;
}) {
  return (
    <div className="field">
      <span>{label}</span>
      <div className="segmented">
        {options.map((option) => (
          <button key={option} className={option === value ? "active" : ""} onClick={() => onChange(option)}>
            {option === value && <CheckCircle2 size={14} />}
            {optionLabels[option] || option}
          </button>
        ))}
      </div>
    </div>
  );
}

function VoiceSelect({ label, value, options, onChange }: { label: string; value: string; options: VoiceOption[]; onChange: (value: string) => void }) {
  const hasCurrent = options.some((option) => option.id === value);
  const renderedOptions = hasCurrent || !value
    ? options
    : [{ id: value, label: value, available: false }, ...options];

  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {renderedOptions.length === 0 ? (
          <option value="">No local voices found</option>
        ) : (
          renderedOptions.map((option) => (
            <option key={option.id} value={option.id} disabled={option.available === false}>
              {option.label || option.id}
              {option.available === false ? " (unavailable)" : ""}
            </option>
          ))
        )}
      </select>
    </label>
  );
}

function SettingsField({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  const [local, setLocal] = useState(value);
  useEffect(() => setLocal(value), [value]);
  return (
    <label className="field">
      <span>{label}</span>
      <input
        value={local}
        onChange={(event) => setLocal(event.target.value)}
        onBlur={() => onChange(local)}
      />
    </label>
  );
}
