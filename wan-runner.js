#!/usr/bin/env node
/**
 * wan-runner.js — WanVideo I2V Clip Generator
 * Children's Story Animation Pipeline — Stage 3
 *
 * Reads story.json → submits each approved scene image to WanVideo I2V
 * on HuggingFace Spaces → polls queue → downloads .mp4 clip per scene.
 *
 * Usage:
 *   node wan-runner.js                  # all image-approved, clip-pending scenes
 *   node wan-runner.js --scene=s02      # single scene
 *   node wan-runner.js --regen          # force regen even if clip exists
 *   node wan-runner.js --dry-run        # print what would run, no API calls
 *
 * Setup:
 *   npm install
 *   export HF_TOKEN=hf_xxxxxxxxxxxxxxx  # free at hf.co/settings/tokens
 *   # Ensure story.json has image_approved: true + image_selected per scene
 *
 * HF Spaces free tier notes:
 *   - Queue wait can be 5–30 min per clip depending on traffic
 *   - Run during off-peak hours (early morning IST) for shorter queues
 *   - 480P space is faster; 720P produces higher quality
 *   - Each clip is ~5–8 seconds; generation takes 3–8 min on GPU
 */

import { Client }    from "@gradio/client";
import fs            from "fs";
import path          from "path";
import https         from "https";
import http          from "http";
import { fileURLToPath } from "url";
import dotenv        from "dotenv";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config();

// ─────────────────────────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────────────────────────
const CONFIG = {
  storyFile:  "./story.json",
  outputDir:  "./pipeline",

  // HuggingFace Space to use.
  // 480P: faster queue, lower res, good for testing
  // 720P: slower, better quality for final output
  // Change to a different public WanVideo I2V space if these are busy:
  //   e.g. "multimodalart/wan2.1-i2v"
  hfSpace: "multimodalart/wan2-1-fast",

  // Gradio endpoint — check your space's API page (/api) if this changes
  gradioEndpoint: "/generate_video",

  // WanVideo generation parameters
  wan: {
    negativePrompt:  "blurry, low quality, distorted, flickering, artifacts, ugly, deformed, extra limbs, morphing faces, color shift, overexposed, underexposed",
    guidanceScale:   1.0,
    inferenceSteps:  4,
    fps:             24,

    // Frame count — controls clip duration.
    // Frames must satisfy: (frames - 1) % 4 == 0
    // Common values: 49 (~3s), 65 (~4s), 81 (~5s), 97 (~6s), 113 (~7s)
    // We compute this dynamically from audio_duration_sec (see calcFrames)
    defaultDurationSec: 3.4,
  },

  // How long to wait between polling status messages (ms)
  pollLogIntervalMs: 15000,

  // Max total wait time per scene before giving up (ms)
  // HF free queue can be very slow — 40 min is realistic on busy days
  timeoutMs: 45 * 60 * 1000,
};

// ─────────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────────
if (!process.env.HF_TOKEN) {
  console.error("\n✗  HF_TOKEN not set.");
  console.error("   Get a free token: https://huggingface.co/settings/tokens");
  console.error("   Then: export HF_TOKEN=hf_your_token_here\n");
  process.exit(1);
}

if (!fs.existsSync(CONFIG.storyFile)) {
  console.error(`\n✗  ${CONFIG.storyFile} not found.\n`);
  process.exit(1);
}

const story = JSON.parse(fs.readFileSync(CONFIG.storyFile, "utf-8"));
const args  = parseArgs();

// ─────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────
function parseArgs() {
  const a = {};
  process.argv.slice(2).forEach(arg => {
    const [k, v] = arg.replace(/^--/, "").split("=");
    a[k] = v !== undefined ? v : true;
  });
  return a;
}

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function saveStory() {
  fs.writeFileSync(CONFIG.storyFile, JSON.stringify(story, null, 2));
}

function elapsed(startMs) {
  const s = Math.round((Date.now() - startMs) / 1000);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

/**
 * Calculate frame count from target duration.
 * WanVideo constraint: (frames - 1) must be divisible by 4.
 */
function calcFrames(durationSec) {
  const raw = Math.round(durationSec * CONFIG.wan.fps);
  // Round up to nearest valid frame count: 1 + 4n
  const n = Math.ceil((raw - 1) / 4);
  return 1 + (n * 4);
}

/**
 * Download a file from url to destPath.
 * Works for http and https, follows redirects.
 */
function downloadFile(url, destPath) {
  return new Promise((resolve, reject) => {
    const proto = url.startsWith("https") ? https : http;
    const file  = fs.createWriteStream(destPath);

    function doGet(u) {
      proto.get(u, res => {
        if (res.statusCode === 301 || res.statusCode === 302) {
          file.close();
          return doGet(res.headers.location);
        }
        if (res.statusCode !== 200) {
          reject(new Error(`Download failed: HTTP ${res.statusCode} for ${u}`));
          return;
        }
        res.pipe(file);
        file.on("finish", () => file.close(resolve));
      }).on("error", reject);
    }

    doGet(url);
    file.on("error", err => { fs.unlink(destPath, () => {}); reject(err); });
  });
}

/**
 * Get the clip duration target for a scene.
 * Priority: audio_duration_sec (filled after recording) > target_clip_duration_sec > default
 */
function getTargetDuration(scene) {
  const duration = scene.audio_duration_sec
    ? scene.audio_duration_sec + 1.5
    : (scene.target_clip_duration_sec || CONFIG.wan.defaultDurationSec);

  return Math.min(3.4, Math.max(0.4, duration));
}

/**
 * Build the WanVideo motion prompt.
 * Combines scene mood + wan_motion_hint into a concise I2V motion prompt.
 */
function buildMotionPrompt(scene) {
  const moodMap = {
    wonder:         "gentle sense of awe, slow discovery motion",
    joy:            "light cheerful energy, subtle bouncy movement",
    calm:           "very still, soft ambient motion, gentle breathing",
    curious:        "slow cautious movement, attentive posture",
    excited:        "lively playful motion, bright animated energy",
    gentle_sadness: "slow heavy motion, soft drooping, quiet atmosphere",
    relief:         "relaxing motion, exhale, settling stillness",
    cozy:           "warm still scene, soft flicker, minimal motion",
    playful:        "bouncy energetic movement, light and fun",
  };

  const moodHint = moodMap[scene.mood] || "gentle natural motion";

  return [
    scene.wan_motion_hint,
    moodHint,
    "smooth cinematic motion, children's animation style, no abrupt cuts, no camera shake"
  ].join(", ");
}

// ─────────────────────────────────────────────────────────────────
// CORE — Submit to HF Spaces and stream queue status
// ─────────────────────────────────────────────────────────────────
async function generateClip(scene, imageBlob, client) {
  const prompt    = buildMotionPrompt(scene);
  const duration  = getTargetDuration(scene);
  const numFrames = calcFrames(duration);

  console.log(`     motion prompt : ${prompt.substring(0, 80)}...`);
  console.log(`     target        : ${duration.toFixed(1)}s → ${numFrames} frames @ ${CONFIG.wan.fps}fps`);

  if (args["dry-run"]) {
    console.log(`     [dry-run] would submit to ${CONFIG.hfSpace}${CONFIG.gradioEndpoint}`);
    return null;
  }

  const startMs = Date.now();
  let lastLogMs = 0;
  let queuePos  = null;

  // Submit job and stream status updates
  const job = client.submit(CONFIG.gradioEndpoint, {
    input_image:      imageBlob,
    prompt:           prompt,
    height:           480,
    width:            832,
    negative_prompt:  CONFIG.wan.negativePrompt,
    duration_seconds: duration,
    guidance_scale:   CONFIG.wan.guidanceScale,
    steps:            CONFIG.wan.inferenceSteps,
    seed:             scene.flux_seed ?? 42,
    randomize_seed:   false,
  });

  // Create a timeout race
  const timeoutPromise = new Promise((_, reject) =>
    setTimeout(() => reject(new Error(`Timed out after ${CONFIG.timeoutMs / 60000}min`)), CONFIG.timeoutMs)
  );

  const resultPromise = (async () => {
    for await (const msg of job) {
      const now = Date.now();
      const s = msg?.data ?? msg;

      if (process.env.DEBUG) {
        console.log("     [DEBUG] WanVideo message:", JSON.stringify(msg, null, 2).slice(0, 400));
      }

      if ((msg?.type === "status") || s?.stage || s?.status) {
        if (s?.queue_position !== undefined && s.queue_position !== queuePos) {
          queuePos = s.queue_position;
          console.log(`     ⟳  Queue position: ${queuePos}  [${elapsed(startMs)}]`);
        }

        if (now - lastLogMs > CONFIG.pollLogIntervalMs) {
          const statusStr = s?.stage || s?.status || "waiting";
          console.log(`     ⟳  Status: ${statusStr}  [${elapsed(startMs)}]`);
          lastLogMs = now;
        }
      }

      if (msg?.type === "data") {
        console.log(`     ✓  Generation complete  [${elapsed(startMs)}]`);
        return msg.data;
      }

      if (s?.error || msg?.type === "error") {
        throw new Error(`Gradio error: ${JSON.stringify(s?.error ?? msg?.data ?? msg)}`);
      }

      if (s?.url || s?.video || Array.isArray(s)) {
        return s;
      }
    }
    throw new Error("Job stream ended without data");
  })();

  return Promise.race([resultPromise, timeoutPromise]);
}

// ─────────────────────────────────────────────────────────────────
// PROCESS ONE SCENE
// ─────────────────────────────────────────────────────────────────
async function processSelectedScene(scene, client) {
  const clipDir  = path.join(CONFIG.outputDir, "clips", scene.scene_id);
  ensureDir(clipDir);

  // Resolve selected image path
  const selectedFile = scene.outputs?.image_selected
    ?? scene.review?.image_selected;

  if (!selectedFile) {
    console.log(`  ⚠  ${scene.scene_id} — no image_selected set in story.json, skipping`);
    return false;
  }

  // Support both filename-only and full path in image_selected
  let imagePath = selectedFile;
  if (!path.isAbsolute(selectedFile) && !fs.existsSync(selectedFile)) {
    imagePath = path.join(CONFIG.outputDir, "scenes", scene.scene_id, selectedFile);
  }

  if (!fs.existsSync(imagePath)) {
    console.log(`  ✗  ${scene.scene_id} — image not found: ${imagePath}`);
    return false;
  }

  // Check if clip already exists
  const existingClip = scene.outputs?.video_clip;
  if (existingClip && fs.existsSync(existingClip) && !args.regen) {
    console.log(`  ✓  ${scene.scene_id} — clip exists, skipping (use --regen to overwrite)`);
    return false;
  }

  console.log(`\n  ─── [${scene.scene_id}] ${"─".repeat(50)}`);
  console.log(`  narration : "${scene.narration_text.substring(0, 70)}${scene.narration_text.length > 70 ? "…" : ""}"`);
  console.log(`  image     : ${path.basename(imagePath)}`);

  // Load image as Blob for Gradio
  const imgBuffer = fs.readFileSync(imagePath);
  const imageBlob = new Blob([imgBuffer], { type: "image/png" });

  let result;
  try {
    result = await generateClip(scene, imageBlob, client);
  } catch (err) {
    console.error(`  ✗  ${scene.scene_id} — ${err.message}`);
    // Write failure note into story.json for visibility
    scene.review.notes = `wan-runner error: ${err.message}`;
    saveStory();
    return false;
  }

  if (args["dry-run"] || !result) return false;

  // Extract video URL from Gradio result
  // Gradio can return: { video: { url: "..." } } or { url: "..." } or a direct array
  let videoUrl = null;

  if (Array.isArray(result) && result[0]?.url)             videoUrl = result[0].url;
  else if (Array.isArray(result) && result[0]?.video?.url) videoUrl = result[0].video.url;
  else if (Array.isArray(result) && result[0]?.video?.path) videoUrl = result[0].video.path;
  else if (result?.video?.url)                             videoUrl = result.video.url;
  else if (result?.video?.path)                            videoUrl = result.video.path;
  else if (result?.url)                                    videoUrl = result.url;
  else if (result?.path)                                   videoUrl = result.path;
  else if (typeof result === "string")                     videoUrl = result;

  if (!videoUrl) {
    console.error(`  ✗  ${scene.scene_id} — unexpected result shape:`, JSON.stringify(result).substring(0, 200));
    return false;
  }

  // Download clip
  const clipName = `${scene.scene_id}_clip.mp4`;
  const clipPath = path.join(clipDir, clipName);

  process.stdout.write(`     ↓  Downloading clip → ${clipPath} ... `);
  try {
    if (/^https?:\/\//.test(videoUrl)) {
      await downloadFile(videoUrl, clipPath);
    } else if (fs.existsSync(videoUrl)) {
      fs.copyFileSync(videoUrl, clipPath);
    } else {
      throw new Error(`unsupported video output: ${videoUrl}`);
    }
    process.stdout.write(`done\n`);
  } catch (err) {
    process.stdout.write(`failed: ${err.message}\n`);
    return false;
  }

  // Update story.json
  scene.outputs        = scene.outputs || {};
  scene.outputs.video_clip = clipPath;
  scene.target_clip_duration_sec = getTargetDuration(scene);
  saveStory();

  console.log(`\n     Clip saved: ${path.resolve(clipPath)}`);
  console.log(`     → Watch the clip, then set in story.json:`);
  console.log(`       "clip_approved": true\n`);

  return true;
}

// ─────────────────────────────────────────────────────────────────
function listSceneImages(scene) {
  const sceneDir = path.join(CONFIG.outputDir, "scenes", scene.scene_id);
  if (!fs.existsSync(sceneDir)) return [];

  return fs.readdirSync(sceneDir)
    .filter(file => file.toLowerCase().endsWith(".png"))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
    .map(file => path.join(sceneDir, file));
}

function candidateIdFromImage(imagePath) {
  const base = path.basename(imagePath, path.extname(imagePath));
  return base.split("_")[0] || base;
}

function getVideoUrl(result) {
  if (Array.isArray(result) && result[0]?.url) return result[0].url;
  if (Array.isArray(result) && result[0]?.video?.url) return result[0].video.url;
  if (Array.isArray(result) && result[0]?.video?.path) return result[0].video.path;
  if (result?.video?.url) return result.video.url;
  if (result?.video?.path) return result.video.path;
  if (result?.url) return result.url;
  if (result?.path) return result.path;
  if (typeof result === "string") return result;
  return null;
}

async function saveVideoOutput(videoUrl, clipPath) {
  if (/^https?:\/\//.test(videoUrl)) {
    await downloadFile(videoUrl, clipPath);
    return;
  }

  if (fs.existsSync(videoUrl)) {
    fs.copyFileSync(videoUrl, clipPath);
    return;
  }

  throw new Error(`unsupported video output: ${videoUrl}`);
}

async function processScene(scene, client) {
  const clipDir = path.join(CONFIG.outputDir, "clips", scene.scene_id);
  ensureDir(clipDir);

  const imagePaths = listSceneImages(scene);
  if (imagePaths.length === 0) {
    console.log(`  ${scene.scene_id} - no PNG images found in pipeline/scenes/${scene.scene_id}, skipping`);
    return 0;
  }

  console.log(`\n  --- [${scene.scene_id}] ${"-".repeat(50)}`);
  console.log(`  narration : "${scene.narration_text.substring(0, 70)}${scene.narration_text.length > 70 ? "..." : ""}"`);
  console.log(`  images    : ${imagePaths.length} candidate(s)`);

  scene.outputs = scene.outputs || {};
  scene.outputs.video_clips = Array.isArray(scene.outputs.video_clips)
    ? scene.outputs.video_clips
    : [];

  let generated = 0;

  for (const imagePath of imagePaths) {
    const candidateId = candidateIdFromImage(imagePath);
    const clipName = `${scene.scene_id}_${candidateId}_clip.mp4`;
    const clipPath = path.join(clipDir, clipName);

    if (fs.existsSync(clipPath) && !args.regen) {
      console.log(`     ${path.basename(imagePath)} -> ${clipName} exists, skipping`);
      if (!scene.outputs.video_clips.some(c => c.path === clipPath)) {
        scene.outputs.video_clips.push({ image: imagePath, path: clipPath });
        if (!scene.outputs.video_clip) scene.outputs.video_clip = clipPath;
        saveStory();
      }
      continue;
    }

    console.log(`\n     image: ${path.basename(imagePath)}`);

    const imgBuffer = fs.readFileSync(imagePath);
    const imageBlob = new Blob([imgBuffer], { type: "image/png" });

    let result;
    try {
      result = await generateClip(scene, imageBlob, client);
    } catch (err) {
      console.error(`     failed: ${err.message}`);
      scene.review.notes = `wan-runner error on ${path.basename(imagePath)}: ${err.message}`;
      saveStory();
      continue;
    }

    if (args["dry-run"] || !result) continue;

    const videoUrl = getVideoUrl(result);
    if (!videoUrl) {
      console.error(`     failed: unexpected result shape: ${JSON.stringify(result).substring(0, 200)}`);
      continue;
    }

    process.stdout.write(`     downloading -> ${clipPath} ... `);
    try {
      await saveVideoOutput(videoUrl, clipPath);
      process.stdout.write("done\n");
    } catch (err) {
      process.stdout.write(`failed: ${err.message}\n`);
      continue;
    }

    scene.outputs.video_clips = scene.outputs.video_clips.filter(c => c.path !== clipPath);
    scene.outputs.video_clips.push({ image: imagePath, path: clipPath });
    if (!scene.outputs.video_clip) scene.outputs.video_clip = clipPath;
    scene.target_clip_duration_sec = getTargetDuration(scene);
    saveStory();

    generated++;
    console.log(`     Clip saved: ${path.resolve(clipPath)}`);
  }

  if (generated > 0) {
    console.log(`\n     Generated ${generated}/${imagePaths.length} clip(s) for ${scene.scene_id}`);
  }

  return generated;
}

// STATUS SUMMARY
// ─────────────────────────────────────────────────────────────────
function printSingleClipSummary(processed) {
  console.log("━━ Pipeline Status " + "━".repeat(52) + "\n");

  const cols   = ["Scene", "Img ✓", "Clip path", "Clip ✓", "Audio ✓"];
  const widths = [8, 7, 32, 8, 8];
  const header = cols.map((c, i) => c.padEnd(widths[i])).join(" ");

  console.log("  " + header);
  console.log("  " + "─".repeat(header.length));

  for (const s of story.scenes) {
    const clipName = s.outputs?.video_clip
      ? path.basename(s.outputs.video_clip)
      : "—";
    const row = [
      s.scene_id,
      s.review.image_approved  ? "✓" : "—",
      clipName.substring(0, 31),
      s.review.clip_approved   ? "✓" : "—",
      s.review.audio_approved  ? "✓" : "—",
    ].map((v, i) => v.padEnd(widths[i])).join(" ");
    console.log("  " + row);
  }

  const total     = story.scenes.length;
  const clipDone  = story.scenes.filter(s => s.review.clip_approved).length;
  const audioDone = story.scenes.filter(s => s.review.audio_approved).length;

  console.log(`\n  Clips approved : ${clipDone}/${total} scenes`);
  console.log(`  Audio approved : ${audioDone}/${total} scenes`);
  console.log(`\n  This run: ${processed} clip(s) generated\n`);

  if (clipDone < total) {
    const pending = story.scenes
      .filter(s => s.outputs?.video_clip && !s.review.clip_approved)
      .map(s => s.scene_id);
    if (pending.length > 0) {
      console.log(`  Awaiting clip review: ${pending.join(", ")}`);
      console.log(`  → Watch each clip, then set "clip_approved": true in story.json`);
    }
  }

  const allClipsApproved = story.scenes.every(s => s.review.clip_approved);
  if (allClipsApproved) {
    console.log("  ✓  All clips approved — ready for audio recording stage!\n");
    console.log("  Next: record narration per scene, split audio, fill audio_duration_sec in story.json");
    console.log("  Then run: node ffmpeg-assembler.js\n");
  }
}

// ─────────────────────────────────────────────────────────────────
function printSummary(processed) {
  console.log("\nPipeline Status\n");

  const cols = ["Scene", "Image", "Clips", "Clip", "Audio"];
  const widths = [8, 7, 7, 7, 7];
  const header = cols.map((c, i) => c.padEnd(widths[i])).join(" ");

  console.log("  " + header);
  console.log("  " + "-".repeat(header.length));

  for (const s of story.scenes) {
    const clipCount = Array.isArray(s.outputs?.video_clips)
      ? s.outputs.video_clips.length
      : (s.outputs?.video_clip ? 1 : 0);

    const row = [
      s.scene_id,
      s.review.image_approved ? "yes" : "-",
      String(clipCount),
      s.review.clip_approved ? "yes" : "-",
      s.review.audio_approved ? "yes" : "-",
    ].map((v, i) => v.padEnd(widths[i])).join(" ");
    console.log("  " + row);
  }

  const total = story.scenes.length;
  const clipDone = story.scenes.filter(s => s.review.clip_approved).length;
  const audioDone = story.scenes.filter(s => s.review.audio_approved).length;

  console.log(`\n  Clips approved : ${clipDone}/${total} scenes`);
  console.log(`  Audio approved : ${audioDone}/${total} scenes`);
  console.log(`\n  This run: ${processed} clip candidate(s) generated\n`);
}

// MAIN
// ─────────────────────────────────────────────────────────────────
async function main() {
  const now = new Date().toLocaleTimeString();
  console.log(`\n▶  WanVideo I2V Runner  [${now}]`);
  console.log(`   Story  : ${story.meta.title}`);
  console.log(`   Space  : ${CONFIG.hfSpace}`);
  console.log(`   Steps  : ${CONFIG.wan.inferenceSteps}  |  FPS: ${CONFIG.wan.fps}`);
  console.log(`   Output : ${path.resolve(CONFIG.outputDir)}`);
  if (args["dry-run"]) console.log("   Mode   : --dry-run (no API calls)");
  if (args.regen)      console.log("   Mode   : --regen (overwriting existing clips)");
  if (args.scene)      console.log(`   Mode   : single scene ${args.scene}`);

  ensureDir(path.join(CONFIG.outputDir, "clips"));

  // Filter scenes to process
  let scenes = story.scenes.filter(s =>
    s.review.image_approved &&
    !s.review.clip_approved
  );

  if (args.regen) {
    scenes = story.scenes.filter(s => s.review.image_approved);
  }

  if (args.scene) {
    scenes = story.scenes.filter(s => s.scene_id === args.scene);
    if (scenes.length === 0) {
      console.log(`\n  ✗  Scene "${args.scene}" not found in story.json\n`);
      process.exit(1);
    }
  }

  if (scenes.length === 0) {
    const imgApproved = story.scenes.filter(s => s.review.image_approved).length;
    console.log(`\n  Nothing to process.`);
    console.log(`  ${imgApproved}/${story.scenes.length} scenes have image_approved: true`);
    if (imgApproved === 0) {
      console.log(`  → Run flux-runner.js first, then set "image_approved": true in story.json`);
    } else {
      console.log(`  → All image-approved scenes already have clips. Use --regen to force regenerate.`);
    }
    printSummary(0);
    return;
  }

  console.log(`\n  Scenes to process : ${scenes.length}`);
  console.log(`  Estimated time    : ${scenes.length * 10}–${scenes.length * 30} min (depends on HF queue)\n`);
  console.log(`  Tip: HF free queues are shorter during off-peak hours`);
  console.log(`       (early morning IST ≈ late evening UTC)\n`);

  // Connect to HF Space (done once — reuse connection for all scenes)
  console.log(`  Connecting to ${CONFIG.hfSpace} ...`);
  let client;
  try {
    client = await Client.connect(CONFIG.hfSpace, {
      hf_token: process.env.HF_TOKEN,
      events: ["data", "status", "error"],
    });
    console.log(`  ✓  Connected\n`);
  } catch (err) {
    console.error(`  ✗  Failed to connect: ${err.message}`);
    console.error(`     Check that the space is public and the endpoint exists.`);
    console.error(`     Try: https://huggingface.co/spaces/${CONFIG.hfSpace}/api`);
    process.exit(1);
  }

  let processed = 0;
  for (const scene of scenes) {
    processed += await processScene(scene, client);
  }

  printSummary(processed);
}

main().catch(err => {
  console.error("\n✗  Fatal:", err.message);
  if (process.env.DEBUG) console.error(err.stack);
  process.exit(1);
});
