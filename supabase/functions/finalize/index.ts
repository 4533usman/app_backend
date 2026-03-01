//@ts-ignore
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
//@ts-ignore
const OPENAI_API_KEY = Deno.env.get("OPENAI_API_KEY")!;
//@ts-ignore
const GOOGLE_API_KEY = Deno.env.get("GOOGLE_API_KEY")!;
//@ts-ignore
const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
//@ts-ignore
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

// ─── Whisper ────────────────────────────────────────────────────────────────
async function whisperTranslate(audioBlob: Blob, segLabel: string): Promise<string> {
  const formData = new FormData();
  formData.append("file", audioBlob, `${segLabel}.mp3`);
  formData.append("model", "whisper-1");
  formData.append("response_format", "text");
  formData.append(
    "prompt",
    "Translate to natural English, handling Urdu-English mix code-switching naturally."
  );

  const res = await fetch("https://api.openai.com/v1/audio/translations", {
    method: "POST",
    headers: { Authorization: `Bearer ${OPENAI_API_KEY}` },
    body: formData,
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Whisper failed for ${segLabel}: ${err}`);
  }
  return (await res.text()).trim();
}

// ─── GPT ────────────────────────────────────────────────────────────────────
async function gptAnalysis(fullVisual: string, fullTranscript: string): Promise<string> {
  const combinedPrompt =
    `Visual Description from Florence-2 (frame captions per segment):\n${fullVisual}\n\n` +
    `Audio Transcript from Whisper (per segment):\n${fullTranscript}\n\n` +
    "You are an expert video analyst. Provide:\n" +
    "1. A detailed, coherent English summary of the video content.\n" +
    "2. Key topics, main ideas, and any important points discussed.\n" +
    "3. A catchy, SEO-friendly YouTube title suggestion (max 70 characters).\n" +
    "4. Optional: 3-5 relevant YouTube tags.\n" +
    "Be concise yet comprehensive.";

  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content:
            "You are a professional video content analyst specializing in educational, vlog, lecture, or mixed-language (Urdu-English) videos. Always produce structured, accurate, natural-English outputs.",
        },
        { role: "user", content: combinedPrompt },
      ],
      temperature: 0.4,
      max_tokens: 1200,
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`GPT failed: ${err}`);
  }

  const data = await res.json();
  return data.choices[0].message.content.trim();
}

// ─── Gemini ──────────────────────────────────────────────────────────────────
async function geminiHtml(gptAnalysisText: string): Promise<string> {
  const customHeadStyle = `
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <script src="https://cdn.tailwindcss.com"></script>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
      <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;700&display=swap');
        body { font-family: 'Inter', sans-serif; background-color: #000000; color: #e2e8f0; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        .diagnostic-panel {
          background-color: #0f172a; border: 1px solid #1e293b;
          box-shadow: 0 0 15px rgba(252, 165, 165, 0.05);
          transition: all 0.3s ease-in-out; margin-bottom: 1.5rem;
          border-radius: 0.5rem; padding: 1.5rem;
        }
        .diagnostic-panel:hover {
          border-color: #ef4444; transform: translateY(-2px);
          box-shadow: 0 5px 20px rgba(239, 68, 68, 0.15);
        }
        @keyframes pulse-red {
          0%, 100% { color: #f87171; text-shadow: 0 0 5px #ef4444; }
          50% { color: #fee2e2; text-shadow: 0 0 10px #f87171; }
        }
        .pulse-text { animation: pulse-red 2s infinite; }
      </style>
    </head>`;

  const prompt =
    "Design a comprehensive, vertically scrolling *Diagnostic Report* from the analysis text below, " +
    "where each numbered section is its own distinct, visually segmented panel.\n\n" +
    "*MANDATORY STYLING & HEAD SECTION:*\n" +
    "You MUST start the HTML file with the exact <head> block provided below. " +
    "Do not create your own styles; use the classes defined in this block.\n" +
    "```html\n" + customHeadStyle + "\n```\n\n" +
    "*File Generation & Technology (Mandatory):*\n" +
    "1. Generate the output as a single, fully self-contained, mobile-responsive HTML file.\n" +
    "2. Mandatory use of Tailwind CSS and appropriate Font Awesome icons.\n" +
    "3. Optimized for vertical scrolling on mobile devices.\n\n" +
    "*Structural Constraints (Mandatory):*\n" +
    "1. Each panel corresponds exactly to one numbered section in the analysis text.\n" +
    "2. Include every single word of the provided analysis text verbatim.\n\n" +
    "*Aesthetic Directive:*\n" +
    "1. Visual design must reflect the theme and emotional content of the analysis.\n\n" +
    gptAnalysisText + "\n" +
    "Return ONLY the final HTML file output. Do not include explanations or markdown.";

  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GOOGLE_API_KEY}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.3, maxOutputTokens: 8192 },
      }),
    }
  );

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Gemini failed: ${err}`);
  }

  const data = await res.json();
  return data.candidates[0].content.parts[0].text.trim();
}

// ─── Main handler ────────────────────────────────────────────────────────────
//@ts-ignore
Deno.serve(async (req: Request) => {
  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: { "Content-Type": "application/json" },
    });
  }

  const { job_id } = await req.json();

  if (!job_id) {
    return new Response(JSON.stringify({ error: "job_id is required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  // 1. Check job status
  const { data: job } = await supabase
    .from("api_jobs")
    .select("*")
    .eq("id", job_id)
    .single();

  if (!job) {
    return new Response(JSON.stringify({ error: "Job not found" }), {
      status: 404,
      headers: { "Content-Type": "application/json" },
    });
  }

  if (job.status === "processing") {
    return new Response(
      JSON.stringify({ job_id, status: "processing", message: "Still processing. Try again shortly." }),
      { status: 202, headers: { "Content-Type": "application/json" } }
    );
  }

  if (job.status === "failed") {
    return new Response(
      JSON.stringify({ job_id, status: "failed", error: job.error }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }

  // 2. Get all batches
  const { data: batches } = await supabase
    .from("video_batches")
    .select("*")
    .eq("job_id", job_id)
    .order("segment_index");

  if (!batches || batches.length === 0) {
    return new Response(JSON.stringify({ error: "No batch data found for this job" }), {
      status: 404,
      headers: { "Content-Type": "application/json" },
    });
  }

  // 3. Whisper — download audio from Storage and transcribe each segment
  const allFlorence: string[] = [];
  const allWhisper: string[] = [];

  for (const batch of batches) {
    const segNum = batch.segment_index + 1;
    allFlorence.push(`[Segment ${segNum}]\n${batch.florence_captions}`);

    if (batch.audio_path) {
      try {
        const { data: audioData, error: dlErr } = await supabase.storage
          .from("audio-segments")
          .download(batch.audio_path);

        if (dlErr || !audioData) throw new Error(dlErr?.message ?? "Download failed");

        const transcript = await whisperTranslate(audioData, `seg_${segNum}`);
        allWhisper.push(`[${batch.segment_index}:00] ${transcript}`);

        // Clean up audio from storage after transcription
        await supabase.storage.from("audio-segments").remove([batch.audio_path]);

      } catch (e) {
        console.error(`Whisper failed for segment ${segNum}:`, e);
        allWhisper.push(`[${batch.segment_index}:00] [transcription failed]`);
      }
    } else {
      allWhisper.push(`[${batch.segment_index}:00] [no audio]`);
    }
  }

  const fullVisual = allFlorence.join("\n\n");
  const fullTranscript = allWhisper.join("\n");

  // 4. GPT analysis
  let gptResult: string;
  try {
    gptResult = await gptAnalysis(fullVisual, fullTranscript);
  } catch (e) {
    const msg = `GPT failed: ${e}`;
    await supabase.from("api_jobs").update({ status: "failed", error: msg }).eq("id", job_id);
    return new Response(JSON.stringify({ job_id, error: msg }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  // 5. Gemini HTML
  let htmlResult: string;
  try {
    htmlResult = await geminiHtml(gptResult);
  } catch (e) {
    const msg = `Gemini failed: ${e}`;
    await supabase.from("api_jobs").update({ status: "failed", error: msg }).eq("id", job_id);
    return new Response(JSON.stringify({ job_id, error: msg }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  // 6. Save final result to Supabase
  await supabase
    .from("api_jobs")
    .update({ response_payload: { html: htmlResult } })
    .eq("id", job_id);

  return new Response(
    JSON.stringify({
      job_id,
      segments_processed: batches.length,
      whisper_transcript: fullTranscript,
      final_analysis: htmlResult,
    }),
    { status: 200, headers: { "Content-Type": "application/json" } }
  );
});
