# Real-Time Voice Agent — Project Plan

Nov 23, 2026 · @Aryan Barik

## Overview

Build a voice agent that books restaurant reservations by talking to the caller in real time. Make it fast enough to feel like a phone call, then prove it with latency numbers. Plan length: 6 weeks, about $100–150 in compute and API spend.

**Resume pitch:** "Built a streaming voice agent (ASR → LLM with tools → TTS) with p50 voice-to-voice latency of X ms and barge-in under Y ms. Benchmarked hosted against self-hosted models across latency, cost and task success."

**Why restaurant booking:** it needs real tool calls (check availability, book, cancel), messy inputs such as names, dates and party sizes, and confirmation turns. A mock backend keeps it fully testable.

**Success criteria (targets to beat, adjust after week 2):**

| Metric | Target |
| --- | --- |
| Voice-to-voice latency, end of user speech → first agent audio | p50 < 800 ms, p95 < 1,200 ms |
| Barge-in, user starts talking → agent audio stops | < 200 ms |
| Task success on scripted scenarios | ≥ 85% |
| False interruptions, agent cuts in during a pause | < 5% of turns |
| Cost per 3-minute call | Measured and reported per stack |

## Architecture

No stage waits for the one before it to finish. Audio, transcripts, tokens and speech all flow in small chunks, so the agent starts talking while it is still thinking.

&#91;embedded content: voice agent pipeline · 7 stages, 1 barge-in path\]

Audio enters top left and comes back out bottom left. The dashed path is barge-in: as soon as voice activity detection hears the user, playback stops and the TTS and LLM streams are cancelled. The millisecond figures are your starting budget, not measurements; replace them with real numbers after week 2.

## Tech stack

Write the orchestration loop yourself in Python asyncio, and plug in models behind a common interface so each stage can be swapped. Frameworks like [LiveKit Agents](https://livekit.com/blog/best-open-source-voice-and-video-ai-agent-frameworks) and [Pipecat](https://www.forasoft.com/blog/article/pipecat-vs-livekit-agents) would do the plumbing for you. For an infra-leaning portfolio, the plumbing is the part you want to show. Use one of them as a comparison baseline in week 6.

| Stage | Start with (hosted, fast to wire) | Self-hosted swap (week 4) | Why |
| --- | --- | --- | --- |
| Transport | WebSocket, browser mic → 16 kHz PCM frames | WebRTC (aiortc) as a stretch | WebSocket is simple to debug; WebRTC handles jitter and echo better |
| Voice activity | Silero VAD | same | Small, runs on CPU in about 1 ms per frame |
| End of turn | Silence timeout (baseline) | [Smart Turn v3](https://huggingface.co/soniqo/Smart-Turn-v3.2-ONNX), an audio model that predicts if the user is done | The model-vs-timeout comparison is one of your key results |
| Speech-to-text | Deepgram or AssemblyAI streaming | [Parakeet TDT 0.6B or Nemotron streaming](https://openwhispr.com/blog/parakeet-vs-whisper-vs-nemotron) via NVIDIA NeMo | Parakeet beats Whisper large-v3 at a quarter of the size; Whisper streaming is too slow (1–5 s) |
| LLM + tools | A fast hosted model with tool calling | Llama or Qwen 7–8B on vLLM | Time to first token dominates the budget; self-hosting lets you control it |
| Text-to-speech | Cartesia or ElevenLabs Flash | [Kokoro 82M](https://bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models) (Apache 2.0) | Measure time to first audio, not time to first byte |
| Mock backend | FastAPI + SQLite: check availability, book, cancel | same | Keeps every test call deterministic |

Pin versions in week 1. This space moves monthly, so re-check model choices before week 4.

## Week-by-week plan

Each week ends with something that runs. Get a slow agent working end to end in week 1, then spend the rest of the project making it fast and measuring it.

**Week 1 — Slow but complete (turn-based)**

- [ ] Browser page that records mic audio and plays audio back over a WebSocket
- [ ] Mock reservation API (FastAPI + SQLite) with 3 tools: check\_availability, book, cancel
- [ ] Turn-based loop: record → transcribe → LLM with tools → speak, all hosted APIs
- [ ] Tracing from day one: timestamp every stage boundary for every turn, logged as JSON

**Week 2 — Streaming everywhere**

- [ ] Streaming ASR with partial transcripts
- [ ] Stream LLM tokens into a sentence chunker; send each clause to TTS as soon as it is complete
- [ ] Stream TTS audio frames to the browser with a small playback buffer
- [ ] First latency report: voice-to-voice p50/p95 and a per-stage breakdown

**Week 3 — Turn-taking and interruptions**

- [ ] Silero VAD on incoming frames; barge-in cancels the LLM stream, TTS and queued audio
- [ ] Track what the agent actually said before it was cut off, so conversation history stays truthful
- [ ] End-of-turn: silence timeout vs Smart Turn model; tune the trade-off between speed and cutting people off
- [ ] Filler and backchannel handling ("um", "okay") so they don't trigger interruptions

**Week 4 — Self-hosted stack**

- [ ] Rent one GPU; serve Parakeet (ASR), a 7–8B LLM on vLLM, and Kokoro (TTS)
- [ ] Same interface as the hosted stack, so you can switch with a config flag
- [ ] Optimizations: warm models, prompt prefix caching, speculative "pre-generation" while the user is still finishing
- [ ] Compare hosted vs self-hosted on latency, cost per call and accuracy

**Week 5 — Evaluation harness**

- [ ] 50+ scripted caller scenarios, played as synthetic audio (see Evaluation)
- [ ] Automated grading of task success from the mock database state
- [ ] Load test: 1, 5, 10, 20 simultaneous calls; plot how latency degrades

**Week 6 — Polish and publish**

- [ ] Run a framework baseline (Pipecat or LiveKit) on the same scenarios for comparison
- [ ] README with charts, architecture diagram and a 2-minute demo video
- [ ] Blog post on what made the biggest latency difference
- [ ] Deploy a public demo with a rate limit and spend cap

## Evaluation and benchmarks

The eval harness is what separates this from a demo. Build it so one command replays every scenario against any stack and writes a results file.

**How test calls work:** write each caller scenario as a script with a goal, such as "party of 4, Friday 7pm, name is Priya Raman, accept 7:30 if 7 is full". A simulated caller (an LLM plus TTS) speaks the caller lines as audio into the agent, with realistic pauses. Grade success by checking the mock database at the end, not by reading the transcript.

**Scenario mix (50+):** simple bookings, unavailable time with a counter-offer, cancellations, spelling hard names, changing your mind mid-sentence, interrupting the agent, long thinking pauses, and background noise mixed in at 2–3 levels.

**What to measure:**

| Metric | How |
| --- | --- |
| Voice-to-voice latency, p50/p95 | Last user audio frame → first agent audio frame, from your stage traces |
| Per-stage breakdown | Endpoint wait, ASR finalize, LLM first token, TTS first audio, network |
| Barge-in latency | User speech start → agent audio stopped |
| End-of-turn quality | False cut-ins vs added wait, timeout vs Smart Turn |
| Task success rate | Correct row in the mock database |
| ASR word error rate on names and numbers | Against the scenario script |
| Cost per call | API usage, or GPU-hours divided by calls served |

**Charts for the README:** a stacked bar of the latency budget for each version (week 1, 2, 3, 4) showing what each change removed; latency vs number of simultaneous calls; and a speed-vs-interruptions curve for end-of-turn settings.

## Stretch goals

Pick at most one or two, and only after week 5 is done.

- **Phone number:** connect a Twilio number so people can actually call it. Telephony audio is 8 kHz, which makes ASR harder and the demo more convincing.
- **Fine-tune your own turn detector:** train a small classifier on audio from your scenarios and compare it with Smart Turn. This is a nice ML training story inside a systems project.
- **Speech-to-speech comparison:** run the same scenarios through a native speech-to-speech model and compare latency, cost and tool-call reliability with your pipeline.
- **WebRTC transport:** replace the WebSocket with aiortc and measure the effect on jitter and latency over a bad network.
- **Multilingual:** add Spanish and measure how much each stage degrades.

## Budget and risks

Expect roughly $100–150 in total. Most of it goes to GPU rental in weeks 4–5 and hosted API calls during eval runs. These are estimates; check current prices before you start.

| Item | Estimate |
| --- | --- |
| Hosted ASR, LLM and TTS during weeks 1–3 | $20–40 (most have free credits) |
| One 24–48 GB GPU for weeks 4–5, about 60–80 hours | $40–80 |
| Eval runs and load tests | $20–30 |
| Public demo hosting | Under $10/month with a spend cap |

**Risks and what to do about them:**

- **Audio plumbing eats week 1.** Resampling, frame sizes and browser autoplay rules are fiddly. Fall back to a framework's transport layer and keep your own orchestration on top.
- **Echo: the agent hears itself and interrupts itself.** Use headphones while developing, and turn on the browser's echo cancellation.
- **Self-hosted stack is slower than hosted.** That is still a valid result. Explain where the time goes; an honest breakdown impresses more than a win.
- **GPU bills creep.** Script start/stop for the GPU box and set a provider spend limit.
- **Scope creep.** Week 3 (interruptions) is the hardest. If it slips, cut a stretch goal, not the eval harness.

## Deliverables

Ship these four things at the end of week 6.

1. **Public GitHub repo** with this layout:

```
voice-agent/
  server/
    pipeline.py      # asyncio orchestrator, cancellation, state
    stages/          # asr/, llm/, tts/, vad/, turn/ — hosted + self-hosted behind one interface
    tools/           # mock reservation API client
    tracing.py       # per-turn stage timestamps
  client/            # browser mic + playback page
  backend/           # FastAPI + SQLite mock restaurant
  eval/
    scenarios/       # caller scripts (YAML)
    simulate.py      # synthetic caller
    run_eval.py      # one command → results.json + charts
  infra/             # GPU box start/stop, Dockerfiles
```

2. **README** that opens with the headline numbers and the latency chart, then the architecture diagram, how to run it, and the results tables.
3. **2-minute demo video** of a real conversation, including an interruption and a counter-offer, with a latency overlay.
4. **Blog post**, for example "Getting a voice agent from 3 seconds to under 1 second", walking through what each change bought.

**Draft resume bullets (fill in your real numbers):**

- Built a real-time voice agent (streaming ASR → tool-calling LLM → streaming TTS) in Python asyncio, cutting voice-to-voice latency from X s to Y ms p50 through streaming, clause-level TTS and speculative generation
- Implemented barge-in with full pipeline cancellation (Z ms) and model-based end-of-turn detection, reducing false interruptions by N% vs silence timeouts
- Built an eval harness of 50+ synthetic-caller scenarios with database-verified task success (M%); benchmarked hosted vs self-hosted (vLLM, Parakeet, Kokoro) stacks on latency, cost per call and concurrency
