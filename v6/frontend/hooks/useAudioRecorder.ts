"use client";

import { useRef, useState } from "react";

/**
 * Microphone capture via MediaRecorder. Picks the first supported MIME type,
 * records until stopRecording(), then hands the final Blob to onStop.
 * (The V6 server transcodes whatever container it receives.)
 */
export function useAudioRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startRecording = async (onStop?: (audioBlob: Blob) => void) => {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        sampleRate: 16000,
      },
    });

    streamRef.current = stream;
    chunksRef.current = [];

    const supportedTypes = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/mp4",
      "audio/ogg;codecs=opus",
      "audio/wav",
    ];
    let mimeType = "";
    for (const type of supportedTypes) {
      if (MediaRecorder.isTypeSupported(type)) {
        mimeType = type;
        break;
      }
    }

    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
    recorderRef.current = recorder;

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };

    recorder.onstop = () => {
      const audioBlob = new Blob(chunksRef.current, {
        type: mimeType || "audio/webm",
      });
      chunksRef.current = [];
      onStop?.(audioBlob);
    };

    recorder.start();
    setIsRecording(true);
    setRecordingTime(0);
    timerRef.current = setInterval(
      () => setRecordingTime((prev) => prev + 1),
      1000
    );

    return mimeType || "audio/webm";
  };

  const stopRecording = () => {
    if (!isRecording) return;
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsRecording(false);
  };

  return { isRecording, recordingTime, startRecording, stopRecording };
}
