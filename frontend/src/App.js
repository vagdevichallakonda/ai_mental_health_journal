// frontend/src/App.js
import React, { useEffect, useRef, useState } from "react";
import { predictMood, predictVoiceEmotion } from "./services/api";

export default function App() {
  // Text state
  const [text, setText] = useState("");
  const [textLoading, setTextLoading] = useState(false);
  const [textResult, setTextResult] = useState(null);
  const [textError, setTextError] = useState("");

  // Audio state
  const [audioFile, setAudioFile] = useState(null);
  const [audioURL, setAudioURL] = useState("");
  const [audioLoading, setAudioLoading] = useState(false);
  const [audioResult, setAudioResult] = useState(null);
  const [audioError, setAudioError] = useState("");

  // Recording
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSupported, setRecordingSupported] = useState(false);

  useEffect(() => {
    setRecordingSupported(!!(navigator.mediaDevices && window.MediaRecorder));
  }, []);

  // --- Text handlers ---
  const onSubmitText = async (e) => {
    e.preventDefault();
    setTextError("");
    setTextResult(null);
    if (!text.trim()) {
      setTextError("Please enter some text.");
      return;
    }
    try {
      setTextLoading(true);
      const data = await predictMood(text.trim());
      setTextResult(data);
    } catch (err) {
      setTextError(err.message);
    } finally {
      setTextLoading(false);
    }
  };

  // --- File upload handlers ---
  const onPickAudio = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    if (!["audio/wav", "audio/x-wav", "audio/mpeg", "audio/ogg"].includes(f.type)) {
      setAudioError("Please upload a WAV, MP3, or OGG file.");
      return;
    }
    setAudioError("");
    setAudioFile(f);
    setAudioURL(URL.createObjectURL(f));
  };

  const onSubmitAudio = async (e) => {
    e.preventDefault();
    setAudioError("");
    setAudioResult(null);
    if (!audioFile) {
      setAudioError("Please upload or record an audio file first.");
      return;
    }
    try {
      setAudioLoading(true);
      const data = await predictVoiceEmotion(audioFile);
      setAudioResult(data);
    } catch (err) {
      setAudioError(err.message);
    } finally {
      setAudioLoading(false);
    }
  };

  // --- Recording handlers ---
  const startRecording = async () => {
    setAudioError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      mediaRecorderRef.current = mr;
      chunksRef.current = [];

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        // Convert to a file with a valid extension the backend accepts (ogg)
        const file = new File([blob], `recording_${Date.now()}.ogg`, {
          type: "audio/ogg",
        });
        setAudioFile(file);
        setAudioURL(URL.createObjectURL(file));
      };

      mr.start();
      setIsRecording(true);
    } catch (err) {
      setAudioError("Mic access denied or unsupported.");
    }
  };

  const stopRecording = () => {
    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== "inactive") {
      mr.stop();
      mr.stream.getTracks().forEach((t) => t.stop());
      setIsRecording(false);
    }
  };

  // --- UI helpers ---
  const JsonBlock = ({ data }) => (
    <pre
      style={{
        background: "#0b1220",
        color: "#e6edf3",
        padding: "12px",
        borderRadius: 8,
        overflowX: "auto",
      }}
    >
      {JSON.stringify(data, null, 2)}
    </pre>
  );

  return (
    <div style={styles.container}>
      <h1 style={{ marginBottom: 4 }}>AI Mental Health Journal</h1>
      <p style={{ color: "#667085", marginTop: 0 }}>
        Enter text or provide a short audio clip to analyze mood/emotion and receive tips.
      </p>

      {/* TEXT PANEL */}
      <section style={styles.card}>
        <h2 style={styles.h2}>Text Mood Prediction</h2>
        <form onSubmit={onSubmitText}>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Write a short journal entry..."
            rows={5}
            style={styles.textarea}
          />
          <div style={{ display: "flex", gap: 8 }}>
            <button type="submit" style={styles.button} disabled={textLoading}>
              {textLoading ? "Analyzing..." : "Analyze Text"}
            </button>
            <button
              type="button"
              style={styles.secondary}
              onClick={() => {
                setText("");
                setTextResult(null);
                setTextError("");
              }}
            >
              Clear
            </button>
          </div>
        </form>
        {textError && <div style={styles.error}>{textError}</div>}
        {textResult && <JsonBlock data={textResult} />}
      </section>

      {/* AUDIO PANEL */}
      <section style={styles.card}>
        <h2 style={styles.h2}>Voice Emotion Prediction</h2>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <input
            type="file"
            accept=".wav,.mp3,.ogg,audio/*"
            onChange={onPickAudio}
          />
          {recordingSupported && !isRecording && (
            <button style={styles.button} onClick={startRecording}>
              Record
            </button>
          )}
          {isRecording && (
            <button style={styles.danger} onClick={stopRecording}>
              Stop
            </button>
          )}
          <button
            style={styles.button}
            onClick={onSubmitAudio}
            disabled={audioLoading}
          >
            {audioLoading ? "Analyzing..." : "Analyze Audio"}
          </button>
          <button
            style={styles.secondary}
            onClick={() => {
              setAudioFile(null);
              setAudioURL("");
              setAudioResult(null);
              setAudioError("");
            }}
          >
            Clear
          </button>
        </div>

        {audioURL && (
          <div style={{ marginTop: 12 }}>
            <audio src={audioURL} controls />
          </div>
        )}
        {audioError && <div style={styles.error}>{audioError}</div>}
        {audioResult && <JsonBlock data={audioResult} />}
      </section>

      <footer style={{ color: "#98a2b3", fontSize: 12, marginTop: 24 }}>
        Tip: The backend accepts wav, mp3, or ogg. Keep recordings ~3 seconds as per model defaults.
      </footer>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: 840,
    margin: "32px auto",
    padding: "0 16px",
    fontFamily: "Inter, system-ui, Arial, sans-serif",
  },
  card: {
    border: "1px solid #e5e7eb",
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    background: "#ffffff",
    boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
  },
  h2: { marginTop: 0, marginBottom: 8 },
  textarea: {
    width: "100%",
    borderRadius: 8,
    border: "1px solid #d0d5dd",
    padding: 10,
    fontSize: 14,
    resize: "vertical",
    marginBottom: 8,
  },
  button: {
    background: "#2563eb",
    color: "white",
    border: "none",
    padding: "10px 14px",
    borderRadius: 8,
    cursor: "pointer",
  },
  secondary: {
    background: "#f2f4f7",
    color: "#111827",
    border: "1px solid #e5e7eb",
    padding: "10px 14px",
    borderRadius: 8,
    cursor: "pointer",
  },
  danger: {
    background: "#dc2626",
    color: "white",
    border: "none",
    padding: "10px 14px",
    borderRadius: 8,
    cursor: "pointer",
  },
  error: {
    marginTop: 8,
    background: "#fef2f2",
    color: "#b91c1c",
    border: "1px solid #fecaca",
    padding: "8px 10px",
    borderRadius: 8,
  },
};
