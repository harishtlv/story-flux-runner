#!/usr/bin/env node
/**
 * flux-runner.js — FLUX Batch Image Generator
 * Children's Story Animation Pipeline — Stage 2
 *
 * Phase 1: Generate one canonical reference image per character
 * Phase 2: Generate N image candidates per approved scene
 *
 * Free tier: HuggingFace Inference API (FLUX.1-schnell, no cost)
 * Upgrade:   Swap generateImage() body with ComfyUI + PuLID for
 *            stronger character consistency (see comment in that fn)
 *
 * Usage:
 *   node flux-runner.js               # both phases, all approved scenes
 *   node flux-runner.js --phase=1     # character refs only
 *   node flux-runner.js --phase=2     # scene images only
 *   node flux-runner.js --scene=s03   # single scene (phase 2)
 *   node flux-runner.js --regen       # force regenerate even if files exist
 *
 * Setup:
 *   npm install
 *   export HF_TOKEN=hf_xxxxxxxxxxxxxxx   # free at hf.co/settings/tokens
 *   # Edit story.json: set "json_approved": true for scenes you want to generate
 */

import { InferenceClient } from "@huggingface/inference";
import fs   from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
import dotenv from "dotenv";

// Load environment variables from .env (HF_TOKEN should be set there)
dotenv.config();

// ─────────────────────────────────────────────────────────────────
// CONFIG
// ─────────────────────────────────────────────────────────────────
const CONFIG = {
  storyFile:          "./story.json",
  outputDir:          "./pipeline",

  // Image dimensions — 16:9 matches WanVideo I2V expected input
  imageWidth:         832,
  imageHeight:        480,

  // FLUX.1-schnell: 4 steps is optimal (it's a distilled model)
  inferenceSteps:     4,

  // Candidates per scene — pick the best one during review gate
  candidatesPerScene: 3,

  // Delay between API calls to avoid HF free-tier rate limits
  delayMs:            1800,

  model: "black-forest-labs/FLUX.1-schnell",
  provider: process.env.HF_PROVIDER || "auto",
};

// ─────────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────────
if (!process.env.HF_TOKEN) {
  console.error("\n✗  HF_TOKEN is not set.");
  console.error("   Get a free token at: https://huggingface.co/settings/tokens");
  console.error("   Then run: export HF_TOKEN=hf_your_token_here\n");
  process.exit(1);
}

if (!fs.existsSync(CONFIG.storyFile)) {
  console.error(`\n✗  ${CONFIG.storyFile} not found. Run the story prompt builder first.\n`);
  process.exit(1);
}

const hf    = new InferenceClient(process.env.HF_TOKEN);

// Load and normalize story JSON if it uses a different schema
const rawStory = JSON.parse(fs.readFileSync(CONFIG.storyFile, "utf-8"));
function normalizeStory(raw) {
  const meta = {
    title: raw.title || (raw.meta && raw.meta.title) || "Untitled",
    art_style: raw.art_style || (raw.meta && raw.meta.art_style) || "",
  };

  // Build character list
  const charMap = {};
  if (Array.isArray(raw.characters)) {
    raw.characters.forEach((c, i) => {
      const id = c.id || `c${i + 1}`;
      charMap[c.name] = { id, name: c.name, flux_description: c.description || c.flux_description || "" };
    });
  } else if (Array.isArray(raw.scenes)) {
    raw.scenes.forEach(scene => {
      (scene.characters || []).forEach((c) => {
        if (!charMap[c.name]) {
          const id = `c${Object.keys(charMap).length + 1}`;
          charMap[c.name] = { id, name: c.name, flux_description: c.description || "" };
        }
      });
    });
  }

  const characters = Object.values(charMap);

  const scenes = (raw.scenes || []).map((s, idx) => {
    const sceneNumber = s.scene_number || s.sequence || (idx + 1);
    const scene_id = s.scene_id || `s${String(sceneNumber).padStart(2, "0")}`;
    const characters_present = (s.characters || []).map(c => (charMap[c.name] ? charMap[c.name].id : c.name));

    return {
      scene_id,
      narration_text: s.narration || s.narration_text || "",
      mood: s.emotion || "",
      camera_framing: s.camera_framing || "medium shot",
      characters_present,
      flux_prompt: s.visual_prompt || s.flux_prompt || "",
      flux_seed: null,
      sequence: sceneNumber,
      outputs: { image_candidates: [] },
      review: {
        json_approved: !!s.json_approved,
        image_approved: false,
        image_selected: null,
        clip_approved: false
      }
    };
  });

  return { meta, characters, scenes };
}

const story = (rawStory.meta && rawStory.scenes && rawStory.characters)
  ? rawStory
  : normalizeStory(rawStory);

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

async function saveBlob(blob, filePath) {
  const buf = Buffer.from(await blob.arrayBuffer());
  fs.writeFileSync(filePath, buf);
  return filePath;
}

function saveStory() {
  fs.writeFileSync(CONFIG.storyFile, JSON.stringify(story, null, 2));
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function bar(current, total, width = 20) {
  const filled = Math.round((current / total) * width);
  return "[" + "█".repeat(filled) + "░".repeat(width - filled) + `] ${current}/${total}`;
}

// ─────────────────────────────────────────────────────────────────
// IMAGE GENERATION — HuggingFace Inference API (FLUX.1-schnell)
// ─────────────────────────────────────────────────────────────────
/**
 * UPGRADE PATH → ComfyUI + PuLID (stronger character consistency)
 *
 * Replace the body of this function with:
 *
 *   import { buildPuLIDWorkflow } from "./comfyui-pulid-workflow.js";
 *
 *   const workflow = buildPuLIDWorkflow({
 *     prompt,
 *     referenceImagePath,   // character.reference_image_path
 *     seed,
 *     width:  CONFIG.imageWidth,
 *     height: CONFIG.imageHeight,
 *     steps:  20,           // dev model — use 20 steps
 *   });
 *
 *   const res = await fetch("http://localhost:8188/api/prompt", {
 *     method: "POST",
 *     headers: { "Content-Type": "application/json" },
 *     body: JSON.stringify({ prompt: workflow })
 *   });
 *   const { prompt_id } = await res.json();
 *   const imageBlob = await pollComfyUIResult(prompt_id);  // see comfyui-pulid-workflow.js
 *   return imageBlob;
 *
 * Note: FLUX.1-schnell does not support negative_prompt (distilled model).
 * The flux_negative_prompt in story.json is reserved for FLUX.1-dev / ComfyUI.
 */
async function generateImage(prompt, seed = null) {
  const params = {
    provider: CONFIG.provider,
    model:  CONFIG.model,
    inputs: prompt,
    parameters: {
      num_inference_steps: CONFIG.inferenceSteps,
      width:               CONFIG.imageWidth,
      height:              CONFIG.imageHeight,
      guidance_scale:      0,   // FLUX.1-schnell: no CFG guidance
    }
  };

  if (seed !== null) params.parameters.seed = seed;

  try {
    return await hf.textToImage(params, { outputType: "blob" });
  } catch (err) {
    // Auto-retry on rate limit
    if (err.message?.includes("429") || err.message?.includes("rate")) {
      console.log("\n    ⟳ Rate limited — waiting 45s before retry...");
      await sleep(45000);
      return hf.textToImage(params, { outputType: "blob" });
    }
    throw describeHfError(err);
  }
}

function describeHfError(err) {
  const response = err.httpResponse;
  if (!response) return err;

  const status = response.status ? `HTTP ${response.status}` : "HTTP error";
  const body = typeof response.body === "string"
    ? response.body
    : JSON.stringify(response.body);
  const details = body ? `${status}: ${body}` : status;

  return new Error(`${err.message} (${details})`);
}

// ─────────────────────────────────────────────────────────────────
// PROMPT BUILDERS
// ─────────────────────────────────────────────────────────────────

/**
 * Character reference prompt
 *
 * Strategy: neutral full-body pose, plain background, no scene context.
 * This gives PuLID (or seed-lock) a clean identity anchor to work from.
 * Generate one per character BEFORE any scene images.
 */
function buildCharRefPrompt(char) {
  return [
    story.meta.art_style,
    char.flux_description,
    "standing in a neutral pose facing the viewer",
    "plain soft light background, full body visible",
    "character reference, clean flat even lighting",
    "no other characters, no background clutter"
  ].join(", ");
}

/**
 * Scene prompt
 *
 * Claude already embedded character descriptions verbatim in flux_prompt
 * during story generation. We simply append camera framing.
 *
 * Seed consistency strategy (HF free tier, no PuLID):
 *   - Generate the character reference first with seed A
 *   - Use seed A + scene_sequence as the scene seed
 *   - This gives partial visual continuity across the same character
 *   - For true consistency, upgrade to ComfyUI + PuLID
 */
function buildScenePrompt(scene) {
  return `${scene.flux_prompt}, ${scene.camera_framing}`;
}

// ─────────────────────────────────────────────────────────────────
// PHASE 1 — Character Reference Images
// ─────────────────────────────────────────────────────────────────
async function generateCharacterRefs() {
  console.log("\n━━ Phase 1: Character Reference Images " + "━".repeat(32) + "\n");

  const charDir = path.join(CONFIG.outputDir, "characters");
  ensureDir(charDir);

  let generated = 0;
  let skipped   = 0;

  for (const char of story.characters) {
    const refPath = path.join(charDir, `${char.id}_reference.png`);
    const exists  = char.reference_image_path && fs.existsSync(char.reference_image_path);

    if (exists && !args.regen) {
      console.log(`  ✓  ${char.name.padEnd(18)} reference exists — skipping`);
      skipped++;
      continue;
    }

    console.log(`  ⟳  ${char.name.padEnd(18)} generating reference image...`);
    console.log(`       method: ${char.consistency_method}`);

    const prompt = buildCharRefPrompt(char);
    const seed   = Math.floor(Math.random() * 2147483647);

    try {
      const blob = await generateImage(prompt, seed);
      await saveBlob(blob, refPath);

      // Store seed so we can use it as a consistency anchor in Phase 2
      char.reference_image_path = refPath;
      char._ref_seed = seed;      // internal use for seed-locking strategy
      saveStory();

      generated++;
      console.log(`  ✓  ${char.name.padEnd(18)} → pipeline/characters/${path.basename(refPath)}`);
      console.log(`       seed locked: ${seed}\n`);

      await sleep(CONFIG.delayMs);
    } catch (err) {
      console.error(`  ✗  ${char.name} — ${err.message}\n`);
      if (process.env.DEBUG) console.error(err.stack || err);
    }
  }

  console.log(`  Generated: ${generated}  |  Skipped: ${skipped}`);
  console.log("\n  ⚑  REVIEW GATE — Check pipeline/characters/ before continuing.");
  console.log("     If a character looks wrong: delete the file, edit story.json");
  console.log("     character flux_description, then re-run --phase=1 --regen\n");
}

// ─────────────────────────────────────────────────────────────────
// PHASE 2 — Scene Image Candidates
// ─────────────────────────────────────────────────────────────────
async function generateSceneImages() {
  console.log("━━ Phase 2: Scene Image Candidates " + "━".repeat(36) + "\n");

  // Build character map for seed-locking
  const charMap = Object.fromEntries(story.characters.map(c => [c.id, c]));

  // Filter scenes to process
  let scenesToProcess = story.scenes.filter(s =>
    s.review.json_approved && !s.review.image_approved
  );

  if (args.scene) {
    scenesToProcess = story.scenes.filter(s => s.scene_id === args.scene);
    if (scenesToProcess.length === 0) {
      console.log(`  ✗  Scene "${args.scene}" not found in story.json`);
      return;
    }
  }

  if (scenesToProcess.length === 0) {
    const total    = story.scenes.length;
    const approved = story.scenes.filter(s => s.review.json_approved).length;
    const done     = story.scenes.filter(s => s.review.image_approved).length;

    console.log(`  Nothing to process.\n`);
    console.log(`  ${bar(done, total)} scenes image-approved`);
    console.log(`  ${approved} scene(s) have json_approved: true`);
    console.log(`  → Open story.json and set "json_approved": true on scenes to generate\n`);
    return;
  }

  console.log(`  Processing: ${scenesToProcess.length} scene(s) × ${CONFIG.candidatesPerScene} candidates\n`);

  for (const [idx, scene] of scenesToProcess.entries()) {
    const sceneDir = path.join(CONFIG.outputDir, "scenes", scene.scene_id);
    ensureDir(sceneDir);

    console.log(`  ─── [${scene.scene_id}] ${idx + 1}/${scenesToProcess.length} ${"─".repeat(40)}`);
    console.log(`  narration : "${scene.narration_text.substring(0, 72)}${scene.narration_text.length > 72 ? "…" : ""}"`);
    console.log(`  mood      : ${scene.mood}  |  camera: ${scene.camera_framing}`);
    console.log(`  chars     : ${scene.characters_present.join(", ") || "none (background scene)"}\n`);

    const prompt     = buildScenePrompt(scene);
    const candidates = [...(scene.outputs.image_candidates || [])];

    // Seed-lock strategy: derive scene base seed from character ref seeds
    // This gives partial consistency with character refs even without PuLID
    const primaryChar = scene.characters_present[0]
      ? charMap[scene.characters_present[0]]
      : null;

    const baseSeed = scene.flux_seed !== null
      ? scene.flux_seed
      : (primaryChar?._ref_seed
          ? (primaryChar._ref_seed + scene.sequence * 1000) % 2147483647
          : Math.floor(Math.random() * 2147483647));

    for (let i = 0; i < CONFIG.candidatesPerScene; i++) {
      const seed    = baseSeed + i;
      const imgFile = `c${i + 1}_seed${seed}.png`;
      const imgPath = path.join(sceneDir, imgFile);

      // Skip if already exists and not forcing regen
      if (fs.existsSync(imgPath) && !args.regen) {
        console.log(`     [${i + 1}/${CONFIG.candidatesPerScene}] ${imgFile} — exists, skipping`);
        if (!candidates.find(c => c.path === imgPath)) {
          candidates.push({ path: imgPath, seed });
        }
        continue;
      }

      process.stdout.write(`     [${i + 1}/${CONFIG.candidatesPerScene}] seed ${seed} → `);

      try {
        const blob = await generateImage(prompt, seed);
        await saveBlob(blob, imgPath);
        candidates.push({ path: imgPath, seed });

        // Lock the first seed into the scene for reproducibility
        if (i === 0 && scene.flux_seed === null) {
          scene.flux_seed = baseSeed;
        }

        process.stdout.write(`✓  ${imgFile}\n`);
        await sleep(CONFIG.delayMs);
      } catch (err) {
        process.stdout.write(`✗  ${err.message}\n`);
      }
    }

    scene.outputs.image_candidates = candidates;
    saveStory();

    console.log(`\n     Saved to: ${path.resolve(sceneDir)}`);
    console.log(`     → Review candidates, then set:\n`);
    console.log(`       "image_selected": "c1_seed${baseSeed}.png",`);
    console.log(`       "image_approved": true\n`);
  }
}

// ─────────────────────────────────────────────────────────────────
// STATUS SUMMARY
// ─────────────────────────────────────────────────────────────────
function printSummary() {
  console.log("━━ Pipeline Status " + "━".repeat(52) + "\n");

  const cols = ["Scene", "JSON ✓", "Imgs", "Selected", "Img ✓", "Clip ✓"];
  const widths = [8, 8, 6, 10, 8, 8];
  const header = cols.map((c, i) => c.padEnd(widths[i])).join(" ");

  console.log("  " + header);
  console.log("  " + "─".repeat(header.length));

  for (const s of story.scenes) {
    const row = [
      s.scene_id,
      s.review.json_approved  ? "✓" : "—",
      String(s.outputs.image_candidates?.length || 0),
      s.review.image_selected ? "✓" : "—",
      s.review.image_approved ? "✓" : "—",
      s.review.clip_approved  ? "✓" : "—",
    ].map((v, i) => v.padEnd(widths[i])).join(" ");
    console.log("  " + row);
  }

  const total       = story.scenes.length;
  const jsonDone    = story.scenes.filter(s => s.review.json_approved).length;
  const imgDone     = story.scenes.filter(s => s.review.image_approved).length;
  const clipDone    = story.scenes.filter(s => s.review.clip_approved).length;

  console.log(`\n  JSON approved : ${bar(jsonDone,  total, 15)} scenes`);
  console.log(`  Img approved  : ${bar(imgDone,   total, 15)} scenes`);
  console.log(`  Clip approved : ${bar(clipDone,  total, 15)} scenes`);

  if (imgDone < total) {
    const remaining = story.scenes.filter(s => !s.review.image_approved);
    console.log(`\n  Pending review: ${remaining.map(s => s.scene_id).join(", ")}`);
  }

  console.log(`\n  Next stage: run wan-runner.js for image-to-video (after all images approved)\n`);
}

// ─────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────
async function main() {
  const now = new Date().toLocaleTimeString();
  console.log(`\n▶  FLUX Batch Runner  [${now}]`);
  console.log(`   Story  : ${story.meta.title}`);
  console.log(`   Scenes : ${story.scenes.length} total | ${story.scenes.filter(s => s.review.json_approved).length} approved`);
  console.log(`   Model  : ${CONFIG.model}`);
  console.log(`   Provider: ${CONFIG.provider}`);
  console.log(`   Output : ${path.resolve(CONFIG.outputDir)}`);
  console.log(`   Size   : ${CONFIG.imageWidth}×${CONFIG.imageHeight}`);
  if (args.regen) console.log("   Mode   : --regen (overwriting existing files)");
  if (args.scene) console.log(`   Mode   : single scene ${args.scene}`);

  ensureDir(CONFIG.outputDir);
  ensureDir(path.join(CONFIG.outputDir, "scenes"));
  ensureDir(path.join(CONFIG.outputDir, "characters"));

  const phase = args.phase;

  if (!phase || phase === "1") await generateCharacterRefs();
  if (!phase || phase === "2") await generateSceneImages();

  printSummary();
}

main().catch(err => {
  console.error("\n✗  Fatal:", err.message);
  if (process.env.DEBUG) console.error(err.stack);
  process.exit(1);
});
