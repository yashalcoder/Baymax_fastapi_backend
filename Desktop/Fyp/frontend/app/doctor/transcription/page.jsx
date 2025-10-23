"use client";
import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import Swal from "sweetalert2";
import { Loader2 } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  ArrowLeft,
  Mic,
  Square,
  Play,
  Pause,
  Save,
  FileText,
  Clock,
  Download,
  Copy,
  Trash2,
  Settings,
  Upload,
} from "lucide-react";

const Transcription = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [transcription, setTranscription] = useState("");
  const [recordingTime, setRecordingTime] = useState(0);
  const [selectedLanguage, setSelectedLanguage] = useState("en");
  const [audioBlob, setAudioBlob] = useState(null);
  const [urduAudioBlob, setUrduAudioBlob] = useState(null);
  const [audioURL, setAudioURL] = useState(null);
  const endpoint = process.env.NEXT_PUBLIC_BACKEND_URL;
  const [urduTranscription, setUrduTranscription] = useState("");
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const streamRef = useRef(null);
  const [isTranscribing, setIsTranscribing] = useState(false);
  // Timer effect
  useEffect(() => {
    if (isRecording && !isPaused) {
      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    }
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [isRecording, isPaused]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs
      .toString()
      .padStart(2, "0")}`;
  };

  const handleStartRecording = async () => {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
        },
      });

      streamRef.current = stream;

      // Try different MIME types based on browser support
      let mimeType = "audio/webm";
      if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
        mimeType = "audio/webm;codecs=opus";
      } else if (MediaRecorder.isTypeSupported("audio/webm")) {
        mimeType = "audio/webm";
      } else if (MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) {
        mimeType = "audio/ogg;codecs=opus";
      } else if (MediaRecorder.isTypeSupported("audio/mp4")) {
        mimeType = "audio/mp4";
      }

      // Create MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, { mimeType });

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      // Collect audio data
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      // Handle recording stop
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        const audioUrl = URL.createObjectURL(audioBlob);

        setAudioBlob(audioBlob);
        setUrduAudioBlob(audioBlob);
        setAudioURL(audioUrl);

        console.log(
          "Recording stopped. Blob size:",
          audioBlob.size,
          "Type:",
          mimeType
        );

        // Stop all tracks
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
        }
      };

      // Start recording
      mediaRecorder.start(1000); // Collect data every second
      setIsRecording(true);
      setRecordingTime(0);

      console.log("Recording started with MIME type:", mimeType);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert("Could not access microphone. Please check permissions.");
      Swal.fire({
        title: "The Internet?",
        text: "That thing is still around?",
        icon: "question",
      });
    }
  };

  const handlePauseResume = () => {
    if (mediaRecorderRef.current) {
      if (isPaused) {
        mediaRecorderRef.current.resume();
        setIsPaused(false);
      } else {
        mediaRecorderRef.current.pause();
        setIsPaused(true);
      }
    }
  };

  const handleStopRecording = () => {
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      mediaRecorderRef.current.stop();
    }

    setIsRecording(false);
    setIsPaused(false);

    if (timerRef.current) {
      clearInterval(timerRef.current);
    }
  };

  const handleDownloadAudio = () => {
    if (audioBlob) {
      const url = URL.createObjectURL(audioBlob);
      const a = document.createElement("a");
      a.style.display = "none";
      a.href = url;
      a.download = `recording-urdu${new Date().toISOString()}.webm`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };
  const handleDownloadUrduAudio = () => {
    if (urduAudioBlob) {
      const url = URL.createObjectURL(urduAudioBlob);
      const a = document.createElement("a");
      a.style.display = "none";
      a.href = url;
      a.download = `recording-${new Date().toISOString()}.webm`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  const handleSendToWhisper = async () => {
    if (!audioBlob) {
      alert("Please record audio first!");
      return;
    }
    setIsTranscribing(true); // Start loading
    // Convert webm to format Whisper accepts (you'll implement this in your backend)
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");
    formData.append("language", selectedLanguage);

    try {
      // This is where you'll call your Whisper API
      // Example API call structure:

      const response = await fetch(`${endpoint}/api/transcribe`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setTranscription(data.english_translation);
      setUrduTranscription(data.urdu_or_punjabi_text);
      console.log(data);

      console.log("Audio blob ready for Whisper:", audioBlob);
      console.log("File size:", (audioBlob.size / 1024).toFixed(2), "KB");
    } catch (error) {
      console.error("Error sending to Whisper:", error);
      alert("Error processing audio");
    } finally {
      setIsTranscribing(false); // Stop loading
    }
  };

  const handleClearTranscription = () => {
    if (confirm("Are you sure you want to clear the transcription?")) {
      setTranscription("");
      setTranscription("");
      setRecordingTime(0);
      setAudioBlob(null);
      setAudioURL(null);
    }
  };
  const handleClearUrduTranscription = () => {
    if (confirm("Are you sure you want to clear the transcription?")) {
      setUrduTranscription("");
      setUrduTranscription("");
      setRecordingTime(0);
      setUrduAudioBlob(null);
      setAudioURL(null);
    }
  };
  const handleCopyTranscription = () => {
    navigator.clipboard.writeText(transcription);
    alert("Transcription copied to clipboard!");
    Swal.fire({
      title: "Transcription successfull!",
      text: "My name is Youshal, and I work here.",
      icon: "success",
    });
  };
  const handleCopyUrduTranscription = () => {
    navigator.clipboard.writeText(urduTranscription);
    alert("Transcription copied to clipboard!");
  };
  const handleDownloadTranscription = () => {
    const element = document.createElement("a");
    const file = new Blob([transcription], { type: "text/plain" });
    element.href = URL.createObjectURL(file);
    element.download = `transcription-${new Date().toISOString()}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };
  const handleDownloadUrduTranscription = () => {
    const element = document.createElement("a");
    const file = new Blob([urduTranscription], { type: "text/plain" });
    element.href = URL.createObjectURL(file);
    element.download = `transcription-urdu-${new Date().toISOString()}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const languages = [
    { code: "en", name: "English", label: "English" },
    { code: "ur", name: "Urdu", label: "اردو" },
    { code: "pa", name: "Punjabi", label: "ਪੰਜਾਬੀ" },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-indigo-50/20">
      {/* Professional Navbar */}
      <nav className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center justify-between h-16">
            {/* Left section */}
            <div className="flex items-center gap-8">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-md">
                  <span className="text-white font-bold text-lg">B+</span>
                </div>
                <div>
                  <h1 className="text-xl font-bold text-gray-900">Baymax+</h1>
                  <p className="text-xs text-gray-500">Doctor Portal</p>
                </div>
              </div>
            </div>

            {/* Center section */}
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                className="text-gray-600 hover:text-gray-900"
              >
                Dashboard
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="text-gray-600 hover:text-gray-900"
              >
                Patients
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="bg-blue-50 text-blue-600 hover:bg-blue-100"
              >
                Transcription
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="text-gray-600 hover:text-gray-900"
              >
                Prescriptions
              </Button>
            </div>

            {/* Right section */}
            <div className="flex items-center gap-3">
              <Button variant="ghost" size="sm">
                <Settings className="w-4 h-4" />
              </Button>
              <div className="flex items-center gap-2">
                <div className="w-9 h-9 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white font-semibold shadow-md">
                  DA
                </div>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto p-8 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold text-gray-900">
              Speech Transcription & Recording
            </h2>
            <p className="text-gray-600 mt-1">
              Record patient consultations with real-time audio capture
            </p>
          </div>
          <Button
            variant="outline"
            onClick={() => (window.location.href = "/doctor")}
            className="border-gray-300 hover:bg-gray-50"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Dashboard
          </Button>
        </div>

        {/* Recording Controls Card */}
        <Card className="shadow-lg border-gray-200">
          <CardHeader className="border-b border-gray-100 bg-gradient-to-r from-white to-gray-50">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2 text-xl">
                  <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center">
                    <Mic className="w-5 h-5 text-blue-600" />
                  </div>
                  Recording Controls
                </CardTitle>
                <CardDescription className="mt-2">
                  Record from microphone and prepare audio for transcription
                </CardDescription>
              </div>

              {/* Recording Timer */}
              {isRecording && (
                <div className="flex items-center gap-3 px-6 py-3 bg-white rounded-xl border-2 border-blue-200 shadow-sm">
                  <div
                    className={`w-3 h-3 rounded-full ${
                      isPaused
                        ? "bg-amber-500"
                        : "bg-red-500 animate-pulse shadow-lg shadow-red-500/50"
                    }`}
                  />
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-gray-600" />
                    <span className="text-2xl font-mono font-bold text-gray-900">
                      {formatTime(recordingTime)}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </CardHeader>

          <CardContent className="p-8">
            {/* Language Selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-3">
                Select Language for Transcription
              </label>
              <div className="flex gap-3">
                {languages.map((lang) => (
                  <button
                    key={lang.code}
                    onClick={() => setSelectedLanguage(lang.code)}
                    disabled={isRecording}
                    className={`px-6 py-3 rounded-xl border-2 transition-all font-medium ${
                      selectedLanguage === lang.code
                        ? "bg-blue-600 text-white border-blue-600 shadow-md"
                        : "bg-white text-gray-700 border-gray-200 hover:border-blue-300"
                    } ${isRecording ? "opacity-50 cursor-not-allowed" : ""}`}
                  >
                    {lang.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Recording Buttons */}
            <div className="flex flex-wrap gap-4">
              {!isRecording ? (
                <Button
                  size="lg"
                  className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg hover:shadow-xl transition-all px-8"
                  onClick={handleStartRecording}
                >
                  <Mic className="w-5 h-5 mr-2" />
                  Start Recording
                </Button>
              ) : (
                <>
                  <Button
                    size="lg"
                    variant="outline"
                    className="border-2 border-amber-500 text-amber-600 hover:bg-amber-50 px-8"
                    onClick={handlePauseResume}
                  >
                    {isPaused ? (
                      <>
                        <Play className="w-5 h-5 mr-2" />
                        Resume
                      </>
                    ) : (
                      <>
                        <Pause className="w-5 h-5 mr-2" />
                        Pause
                      </>
                    )}
                  </Button>
                  <Button
                    size="lg"
                    variant="outline"
                    className="border-2 border-red-500 text-red-600 hover:bg-red-50 px-8"
                    onClick={handleStopRecording}
                  >
                    <Square className="w-5 h-5 mr-2" />
                    Stop Recording
                  </Button>
                </>
              )}

              {/* Download and Send to Whisper buttons */}
              {audioBlob && !isRecording && (
                <>
                  <Button
                    size="lg"
                    variant="outline"
                    className="border-2 border-green-500 text-green-600 hover:bg-green-50 px-8"
                    onClick={handleDownloadAudio}
                  >
                    <Download className="w-5 h-5 mr-2" />
                    Download Audio
                  </Button>
                  <Button
                    size="lg"
                    className="bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-lg hover:shadow-xl px-8"
                    onClick={handleSendToWhisper}
                    disabled={isTranscribing || !audioBlob} // Disable when loading or no audio
                  >
                    {isTranscribing ? (
                      <>
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Upload className="w-5 h-5 mr-2" />
                        Send to Whisper AI
                      </>
                    )}
                  </Button>
                </>
              )}
            </div>

            {/* Recording Status */}
            {isRecording && (
              <div className="mt-6 p-5 rounded-xl bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 rounded-full bg-blue-600 animate-pulse" />
                    <span className="text-sm font-semibold text-gray-900">
                      {isPaused
                        ? "Recording Paused - Click Resume to continue"
                        : "Recording in Progress - Audio being captured from microphone"}
                    </span>
                  </div>
                  <Badge
                    variant="outline"
                    className="border-blue-300 text-blue-700 bg-white"
                  >
                    {languages.find((l) => l.code === selectedLanguage)?.name}
                  </Badge>
                </div>
              </div>
            )}

            {/* Audio Preview */}
            {audioURL && !isRecording && (
              <div className="mt-6 p-5 rounded-xl bg-green-50 border border-green-200">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-semibold text-gray-900 mb-2">
                      Recording Complete! ✅
                    </p>
                    <audio controls src={audioURL} className="w-full max-w-md">
                      Your browser does not support audio playback.
                    </audio>
                    <p className="text-xs text-gray-600 mt-2">
                      File size:{" "}
                      {audioBlob ? (audioBlob.size / 1024).toFixed(2) : 0} KB
                    </p>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
        <div className="grid grid-cols-2 gap-4">
          {/* Transcription Card */}
          <Card className="shadow-lg border-gray-200">
            <CardHeader className="border-b border-gray-100 bg-gradient-to-r from-white to-gray-50">
              <div className="items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2 text-xl">
                    <div className="w-10 h-10 rounded-full bg-indigo-100 flex items-center justify-center">
                      <FileText className="w-5 h-5 text-indigo-600" />
                    </div>
                    Transcription Output
                  </CardTitle>
                  <CardDescription className="mt-2">
                    Whisper AI transcription will appear here • Click to edit
                  </CardDescription>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleCopyTranscription}
                    disabled={!transcription}
                    className="border-gray-300 hover:bg-gray-50"
                  >
                    <Copy className="w-4 h-4 mr-2" />
                    Copy
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleDownloadTranscription}
                    disabled={!transcription}
                    className="border-gray-300 hover:bg-gray-50"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleClearTranscription}
                    disabled={!transcription}
                    className="border-red-300 text-red-600 hover:bg-red-50"
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    Clear
                  </Button>
                  <Button
                    size="sm"
                    disabled={!transcription}
                    className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md hover:shadow-lg"
                  >
                    <Save className="w-4 h-4 mr-2" />
                    Save Transcript
                  </Button>
                </div>
              </div>
            </CardHeader>

            <CardContent className="p-8">
              <Textarea
                placeholder="Transcription from Whisper AI will appear here. You can also edit the text manually after transcription..."
                className="min-h-[500px] text-base resize-none border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent p-6 leading-relaxed"
                value={transcription}
                onChange={(e) => setTranscription(e.target.value)}
              />

              {/* Transcription Stats */}
              <div className="mt-6 flex items-center justify-between p-4 bg-gray-50 rounded-xl border border-gray-200">
                <div className="flex gap-6 text-sm text-gray-600">
                  <span className="font-medium">
                    {transcription.length} characters
                  </span>
                  <span>•</span>
                  <span className="font-medium">
                    {transcription.split(/\s+/).filter(Boolean).length} words
                  </span>
                  <span>•</span>
                  <span className="font-medium">
                    {transcription.split(/[.!?]+/).filter(Boolean).length}{" "}
                    sentences
                  </span>
                </div>

                {transcription && (
                  <Badge
                    variant="outline"
                    className="border-green-300 text-green-700 bg-white"
                  >
                    Ready to save
                  </Badge>
                )}
              </div>
            </CardContent>
          </Card>
          <Card className="shadow-lg border-gray-200">
            <CardHeader className="border-b border-gray-100 bg-gradient-to-r from-white to-gray-50">
              <div className="items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2 text-xl">
                    <div className="w-10 h-10 rounded-full bg-indigo-100 flex items-center justify-center">
                      <FileText className="w-5 h-5 text-indigo-600" />
                    </div>
                    Transcription Output
                  </CardTitle>
                  <CardDescription className="mt-2">
                    Whisper AI transcription will appear here • Click to edit
                  </CardDescription>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleCopyUrduTranscription}
                    disabled={!urduTranscription}
                    className="border-gray-300 hover:bg-gray-50"
                  >
                    <Copy className="w-4 h-4 mr-2" />
                    Copy
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleDownloadUrduTranscription}
                    disabled={!urduTranscription}
                    className="border-gray-300 hover:bg-gray-50"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleClearUrduTranscription}
                    disabled={!urduTranscription}
                    className="border-red-300 text-red-600 hover:bg-red-50"
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    Clear
                  </Button>
                  <Button
                    size="sm"
                    disabled={!urduTranscription}
                    className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md hover:shadow-lg"
                  >
                    <Save className="w-4 h-4 mr-2" />
                    Save Transcript
                  </Button>
                </div>
              </div>
            </CardHeader>

            <CardContent className="p-8">
              <Textarea
                placeholder="Transcription from Whisper AI will appear here. You can also edit the text manually after transcription..."
                className="min-h-[500px] text-base resize-none border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent p-6 leading-relaxed"
                value={urduTranscription}
                onChange={(e) => setUrduTranscription(e.target.value)}
              />

              {/* Transcription Stats */}
              <div className="mt-6 flex items-center justify-between p-4 bg-gray-50 rounded-xl border border-gray-200">
                <div className="flex gap-6 text-sm text-gray-600">
                  <span className="font-medium">
                    {urduTranscription.length} characters
                  </span>
                  <span>•</span>
                  <span className="font-medium">
                    {urduTranscription.split(/\s+/).filter(Boolean).length}{" "}
                    words
                  </span>
                  <span>•</span>
                  <span className="font-medium">
                    {urduTranscription.split(/[.!?]+/).filter(Boolean).length}{" "}
                    sentences
                  </span>
                </div>

                {urduTranscription && (
                  <Badge
                    variant="outline"
                    className="border-green-300 text-green-700 bg-white"
                  >
                    Ready to save
                  </Badge>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Transcription;
