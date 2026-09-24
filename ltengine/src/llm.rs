use llama_cpp_2::context::params::{LlamaContextParams, LlamaContextType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaModel, LlamaChatMessage};
use llama_cpp_2::vocab::LlamaVocab;
use llama_cpp_common::chat::LlamaMinjaChatTemplate;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_common::speculative::{MtpSpeculative, MtpSpeculativeParams};
use llama_cpp_2::{send_logs_to_tracing, LogOptions};
use std::num::NonZeroU32;
use std::path::PathBuf;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use anyhow::{Result, Context};

#[derive(Debug, thiserror::Error)]
pub enum LLMError {
    #[error("Server busy, please try again later")]
    Busy,
    #[error("Generation was cancelled")]
    Cancelled,
    /// The requested `max_output_tokens` needs more context positions than the
    /// model's training context (`RD-29` 3.3). The route maps this to HTTP 400.
    #[error("{0}")]
    OutputTokensExceedContext(String),
}

/// The sampler temperature that predates `RD-29`. `temp_ext(0.0, ...)` selects
/// the greedy path.
pub const DEFAULT_TEMPERATURE: f32 = 0.0;

/// The sampler `top_p` that predates `RD-29`.
pub const DEFAULT_TOP_P: f32 = 0.95;

/// The generation controls of one request (`RD-29`).
///
/// [`Generation::default`] is the behavior that predates `RD-29`, so a request
/// that carries no control keeps the current sampler and context size
/// (`INVARIANT PH8-1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Generation {
    /// The `temp_ext` temperature. `0.0` is greedy.
    pub temperature: f32,
    /// The `top_p` entry.
    pub top_p: f32,
    /// The hard ceiling on generated tokens, or `None` for the `3 × prompt`
    /// budget of `PH-1` through `PH-7`.
    pub max_output_tokens: Option<u32>,
}

impl Default for Generation {
    fn default() -> Self {
        Self {
            temperature: DEFAULT_TEMPERATURE,
            top_p: DEFAULT_TOP_P,
            max_output_tokens: None,
        }
    }
}

/// The last absolute position that the decode loop may generate.
///
/// The loop condition is inclusive (`while n_cur <= bound`), so the last
/// position is one less than the exclusive ceiling. With `max_output_tokens`
/// present, positions `P..P+M` are generated, which is exactly `M` tokens
/// (`PH8-05`). `None` keeps the pre-`RD-29` `3 × prompt_tokens` bound.
fn output_ceiling(prompt_tokens: i32, max_output_tokens: Option<u32>) -> i32 {
    match max_output_tokens {
        Some(max) => prompt_tokens
            .saturating_add(i32::try_from(max).unwrap_or(i32::MAX))
            .saturating_sub(1),
        None => prompt_tokens.saturating_mul(3),
    }
}

/// The context positions that a request needs: the exclusive ceiling plus the
/// MTP margin (`RD-29` 3.2).
fn context_positions(prompt_tokens: i32, max_output_tokens: Option<u32>, margin: i32) -> i32 {
    let budget = match max_output_tokens {
        Some(max) => prompt_tokens.saturating_add(i32::try_from(max).unwrap_or(i32::MAX)),
        None => prompt_tokens.saturating_mul(3),
    };
    budget.saturating_add(margin)
}

/// The `RD-29` 3.3 rejection, or `None` when the request fits. The check needs
/// the tokenized prompt, so it runs after tokenization.
fn context_rejection(
    prompt_tokens: i32,
    max_output_tokens: Option<u32>,
    margin: i32,
    n_ctx_train: u32,
) -> Option<String> {
    let max = max_output_tokens?;
    let needed = context_positions(prompt_tokens, Some(max), margin);
    if i64::from(needed) > i64::from(n_ctx_train) {
        return Some(format!(
            "`max_output_tokens` {max} needs {needed} context positions (prompt {prompt_tokens} plus margin {margin}), but the model's training context is {n_ctx_train}"
        ));
    }
    None
}

/// True when the optional cancellation flag is set (`PH-6b`, `RD-24`).
fn is_cancelled(cancel: Option<&AtomicBool>) -> bool {
    cancel.is_some_and(|flag| flag.load(Ordering::Relaxed))
}

/// Exact token counts of one generation (RD-18).
///
/// `input_tokens` counts the token IDs of the fully chat-templated prompt,
/// including the beginning-of-sequence token. `output_tokens` counts the
/// generated token IDs, excluding the end-of-generation token and including the
/// accepted MTP draft tokens and the sampled tokens. `reasoning_tokens` counts
/// the generated token IDs that the reasoning-trace cleanup removed (`RD-38`);
/// it is a subset of `output_tokens`. No field is an estimate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// `RD-38`: the generated tokens removed with the reasoning trace.
    pub reasoning_tokens: u32,
}

impl TokenUsage {
    /// The exact sum of the two counts (RD-18).
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }

    /// Count one generated token ID. The end-of-generation token is never
    /// counted (RD-18).
    fn record_output(&mut self, is_eog: bool) {
        if !is_eog {
            self.output_tokens += 1;
        }
    }
}

/// Query (free_mib, total_mib) on device 0. Works for CUDA and Vulkan; None for CPU builds.
#[allow(unused_mut)]
fn vram_mib() -> Option<(u64, u64)> {
    let mut free = 0usize;
    let mut total = 0usize;

    #[cfg(feature = "cuda")]
    {
        unsafe extern "C" {
            fn ggml_backend_cuda_get_device_memory(device: i32, free: *mut usize, total: *mut usize);
        }
        unsafe { ggml_backend_cuda_get_device_memory(0, &mut free, &mut total) };
        if total > 0 {
            return Some((free as u64 / (1024 * 1024), total as u64 / (1024 * 1024)));
        }
    }

    #[cfg(feature = "vulkan")]
    {
        unsafe extern "C" {
            fn ggml_backend_vk_get_device_memory(device: i32, free: *mut usize, total: *mut usize);
        }
        unsafe { ggml_backend_vk_get_device_memory(0, &mut free, &mut total) };
        if total > 0 {
            return Some((free as u64 / (1024 * 1024), total as u64 / (1024 * 1024)));
        }
    }

    let _ = (free, total);
    None
}

/// Pick n_ubatch based on total VRAM of the primary GPU device.
///
/// On cards with limited VRAM the default n_ubatch can exhaust memory when
/// combined with KV cache and GPU compute buffers, causing an OOM crash.
/// We use n_ubatch=128 on cards with less than 6 GB of total VRAM.
fn pick_n_ubatch(use_gpu: bool) -> u32 {
    let default = LlamaContextParams::default().n_ubatch();
    if use_gpu {
        if let Some((_, total_mib)) = vram_mib() {
            let n = if total_mib >= 6 * 1024 { default } else { 128 };
            eprintln!("ltengine: {} MiB total VRAM, n_ubatch={}", total_mib, n);
            return n;
        }
    }
    default
}

/// The reasoning controls that reach the model's chat template (`RD-28`).
///
/// [`Reasoning::default`] is the behavior that predates `RD-28`: the template's
/// `enable_thinking` variable is false and no `reasoning_effort` is set.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Reasoning<'a> {
    /// The template's `enable_thinking` variable.
    pub thinking: bool,
    /// The `reasoning_effort` template variable, as a plain value such as
    /// `"low"`. `None` leaves the template's own default in place.
    pub effort: Option<&'a str>,
}

/// The template keywords for `reasoning` (`RD-28`).
///
/// The binding takes JSON text, so the plain effort value is encoded here.
fn reasoning_kwargs(reasoning: &Reasoning<'_>) -> Vec<(&'static str, String)> {
    match reasoning.effort {
        Some(effort) => vec![(
            "reasoning_effort",
            serde_json::Value::String(effort.to_owned()).to_string(),
        )],
        None => Vec::new(),
    }
}

pub struct LLM {
    backend: LlamaBackend,
    model: LlamaModel,
    mtp_model: Option<LlamaModel>,
    self_mtp: bool,
    mtp_n_max: i32,
    prompt_lock: Mutex<()>,
    n_ubatch: u32,
}

pub struct LLMContext<'a>{
    llm: &'a LLM,
    ctx: LlamaContext<'a>,
    ctx_size: i32
}

impl LLM {
    pub fn new(
        model_path: PathBuf,
        mtp_model_path: Option<PathBuf>,
        mtp_n_max: i32,
        cpu: bool,
        verbose: bool,
    ) -> Result<Self> {
        let has_draft_file = mtp_model_path.is_some();
        // A target GGUF can carry its own nextn/MTP head. The loader skips those
        // tensors unless `load_mtp` is set, so request them when no draft file
        // is configured. Architectures without the head ignore the flag.
        let load_mtp = !has_draft_file;
        if has_draft_file {
            validate_mtp_n_max(mtp_n_max)?;
        }
        if !verbose {
            send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
        }

        let backend = LlamaBackend::init()?;
        let use_gpu = !cpu && cfg!(any(feature = "cuda", feature = "vulkan"));

        let (model, gpu_layers) = if use_gpu {
            let mut n_gpu = 9999u32;
            let model = loop {
                let model = match LlamaModel::load_from_file(
                    &backend, &model_path,
                    &LlamaModelParams::default().with_n_gpu_layers(n_gpu).with_load_mtp(load_mtp),
                ) {
                    Ok(m) => m,
                    Err(_) => {
                        // Load failed (likely GPU OOM before probe). On the first failure
                        // jump to 64 (covers most models); after that halve to converge fast.
                        let next = if n_gpu >= 9999 { 64 } else { n_gpu / 2 };
                        eprintln!("ltengine: model load failed at {} GPU layers, retrying with {}", n_gpu, next);
                        n_gpu = next;
                        if n_gpu == 0 {
                            return Err(anyhow::anyhow!("Unable to load model even with 0 GPU layers"));
                        }
                        continue;
                    }
                };

                // Probe: create a minimal context and decode one token to confirm
                // the GPU has enough VRAM for compute scratch buffers.
                let probe_ok = model.new_context(
                    &backend,
                    LlamaContextParams::default()
                        .with_n_ctx(Some(NonZeroU32::new(8).unwrap()))
                        .with_n_ubatch(1),
                ).ok().and_then(|mut ctx| {
                    let mut batch = LlamaBatch::new(8, 1);
                    batch.add(LlamaToken(0), 0, &[0], true).ok()?;
                    ctx.decode(&mut batch).ok()
                }).is_some();

                if probe_ok {
                    break model;
                }

                let actual = model.n_layer() as u32;
                let current = n_gpu.min(actual);
                let next = current.saturating_sub((current / 10).max(1));
                eprintln!("ltengine: GPU probe failed at {} layers, retrying with {}", current, next);
                n_gpu = next;
                drop(model);

                if n_gpu == 0 {
                    return Err(anyhow::anyhow!("GPU inference failed even with 0 layers"));
                }
            };

            let actual = model.n_layer() as u32;
            let on_gpu = n_gpu.min(actual);
            let gpu_layers = if on_gpu < actual { Some(on_gpu) } else { None };
            (model, gpu_layers)
        } else {
            let model = LlamaModel::load_from_file(
                &backend, model_path,
                &LlamaModelParams::default().with_n_gpu_layers(0).with_load_mtp(load_mtp),
            ).with_context(|| "Unable to load model")?;
            (model, None)
        };

        let mtp_model = mtp_model_path
            .map(|path| load_mtp_model(&backend, &path, use_gpu))
            .transpose()?;
        let n_ubatch = pick_n_ubatch(use_gpu);

        // Self-speculative decoding: the target model drafts with its own
        // nextn/MTP head, so no second model file is needed.
        let nextn_layers = if has_draft_file { 0 } else { nextn_predict_layers(&model) };
        let self_mtp = match mtp_mode(has_draft_file, nextn_layers) {
            MtpMode::SelfMtp => {
                validate_mtp_n_max(mtp_n_max)?;
                probe_mtp(&backend, &model, &model, mtp_n_max, n_ubatch)?;
                eprintln!(
                    "ltengine: MTP head found in the target model ({} nextn layer(s)), self-speculative decoding enabled",
                    nextn_layers
                );
                true
            }
            _ => false,
        };

        if let Some(draft) = &mtp_model {
            validate_mtp_pair(&model, draft)?;
            probe_mtp(&backend, &model, draft, mtp_n_max, n_ubatch)?;
            eprintln!("ltengine: MTP draft model loaded");
        }

        match (use_gpu, gpu_layers) {
            (false, _) => eprintln!("ltengine: {} model layers, CPU only", model.n_layer()),
            (true, None) => eprintln!("ltengine: {} model layers, all offloaded to GPU", model.n_layer()),
            (true, Some(n)) => eprintln!("ltengine: {}/{} model layers on GPU, rest on CPU", n, model.n_layer()),
        }

        Ok(LLM {
            backend,
            model,
            mtp_model,
            self_mtp,
            mtp_n_max,
            prompt_lock: Mutex::new(()),
            n_ubatch,
        })
    }

    pub fn create_context(&self, ctx_size: i32) -> Result<LLMContext<'_>>{
        let ctx_params =
            LlamaContextParams::default()
                .with_n_ctx(Some(NonZeroU32::new(ctx_size as u32).unwrap()))
                .with_n_ubatch(self.n_ubatch);

        // Use all threads
        // ctx_params = ctx_params.with_n_threads(threads);
        // ctx_params = ctx_params.with_n_threads_batch(threads_batch);

        let ctx = self.model
            .new_context(&self.backend, ctx_params)
            .with_context(|| "Unable to create the llama context")?;
        Ok(LLMContext{ llm: self, ctx, ctx_size })
    }

    /// Run one generation and return the cleaned text.
    ///
    /// Kept for the existing callers (`/translate`). It delegates to
    /// [`LLM::run_prompt_usage`], so every caller uses one decode path
    /// (`CC-4`).
    pub fn run_prompt(&self, system: String, user: String) -> Result<String>{
        self.run_prompt_usage(system, user).map(|(text, _usage)| text)
    }

    /// Run one generation and return the cleaned text with the exact token
    /// counts of RD-18, under explicit reasoning controls (`RD-28`).
    pub fn run_prompt_usage(&self, system: String, user: String) -> Result<(String, TokenUsage)>{
        self.run_prompt_usage_grammar(system, user, None, &Reasoning::default(), &Generation::default())
            .map(|(text, _trace, usage)| (text, usage))
    }

    /// Run one generation with an optional GBNF grammar that constrains decode
    /// (PH-4a). `None` keeps the free-text behavior of [`LLM::run_prompt_usage`].
    /// The grammar is the only difference: both callers share one decode path
    /// (`CC-4`).
    pub fn run_prompt_usage_grammar(
        &self,
        system: String,
        user: String,
        grammar: Option<&str>,
        reasoning: &Reasoning<'_>,
        generation: &Generation,
    ) -> Result<(String, String, TokenUsage)>{
        self.run_prompt_usage_grammar_cancellable(system, user, grammar, None, reasoning, generation)
    }

    /// The same decode path with a cancellation flag (`PH-6b`, `RD-24`).
    ///
    /// The flag is checked before the prompt lock and at each decoded token. A
    /// set flag returns [`LLMError::Cancelled`]. `run_prompt_usage_grammar`
    /// delegates with `None`, so the foreground path is unchanged.
    pub fn run_prompt_usage_grammar_cancellable(
        &self,
        system: String,
        user: String,
        grammar: Option<&str>,
        cancel: Option<&AtomicBool>,
        reasoning: &Reasoning<'_>,
        generation: &Generation,
    ) -> Result<(String, String, TokenUsage)>{
        let messages = [
            LlamaChatMessage::new("user".to_string(), format!("{system}\n\n{user}"))
                .context("Failed to build chat message")?
        ];

        // Render with the model's embedded Jinja template through llama.cpp's
        // Minja engine. Unlike apply_chat_template, Minja handles templates that
        // the built-in name list cannot (e.g. Gemma 4).
        let template = LlamaMinjaChatTemplate::from_model(&self.model)
            .with_context(|| "Model has no usable embedded chat template")?;
        let kwargs = reasoning_kwargs(reasoning);
        let kwargs_refs: Vec<(&str, &str)> = kwargs
            .iter()
            .map(|(key, value)| (*key, value.as_str()))
            .collect();
        let llm_input = template
            .render_with_kwargs(&messages, true, reasoning.thinking, &kwargs_refs)
            .with_context(|| "Failed to apply the model's chat template")?;

        // llama.cpp strips a leading BOS from the rendered text whenever the
        // vocabulary enables add_bos (common/chat.cpp), so the rendered prompt
        // cannot be assumed to carry one. `add_special = true` lets the
        // vocabulary decide, and `parse_special = true` keeps the template's
        // special tokens as tokens.
        let tokens_list = self.model.vocab().tokenize(llm_input.as_bytes(), true, true);
        // for token in &tokens_list {
        //     eprint!("{} {} | ", self.model.token_to_str(*token, Special::Tokenize)?, token);
        // }
        // RD-18: the tokens that reach decode are the count, BOS included.
        let input_tokens = u32::try_from(tokens_list.len())
            .context("prompt token count does not fit in u32")?;
        let prompt_tokens = tokens_list.len() as i32;
        let mtp_enabled = self.mtp_model.is_some() || self.self_mtp;
        // The MTP path reserves `mtp_n_max + 1` positions beyond the decode
        // ceiling for its draft verification (`RD-29` 3.2).
        let margin = if mtp_enabled { self.mtp_n_max + 1 } else { 0 };
        if let Some(message) = context_rejection(
            prompt_tokens,
            generation.max_output_tokens,
            margin,
            self.model.n_ctx_train(),
        ) {
            return Err(LLMError::OutputTokensExceedContext(message).into());
        }
        let ceiling = output_ceiling(prompt_tokens, generation.max_output_tokens);
        let ctx_size = context_positions(prompt_tokens, generation.max_output_tokens, margin);
        // Lock before create_context: context allocation uses GPU resources and
        // two concurrent allocations corrupt each other even before inference starts.
        // TODO: The llama bindings (or llama itself?) do not appear to be totally thread-safe
        // as garbage starts to come out when we run inference in parallel
        // this might need to be investigated and fixed. For now we lock and process requests
        // one at a time.
        let _lock = self.lock_prompt(cancel)?;
        let (text, trace, output_tokens, reasoning_tokens) = if mtp_enabled {
            self.process_mtp(
                tokens_list,
                ctx_size,
                ceiling,
                grammar,
                cancel,
                reasoning.thinking,
                generation,
            )?
        } else {
            let mut ctx = self.create_context(ctx_size)?;
            ctx.process(tokens_list, ceiling, grammar, cancel, reasoning.thinking, generation)?
        };
        Ok((
            text,
            trace,
            TokenUsage {
                input_tokens,
                output_tokens,
                reasoning_tokens,
            },
        ))
    }

    /// Acquire the prompt lock and observe the cancellation flag (`RD-24`).
    fn lock_prompt(&self, cancel: Option<&AtomicBool>) -> Result<parking_lot::MutexGuard<'_, ()>> {
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            if is_cancelled(cancel) {
                return Err(LLMError::Cancelled.into());
            }
            if let Some(guard) = self.prompt_lock.try_lock_for(Duration::from_millis(50)) {
                return Ok(guard);
            }
            if Instant::now() >= deadline {
                return Err(LLMError::Busy.into());
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_mtp(
        &self,
        tokens_list: Vec<LlamaToken>,
        context_size: i32,
        output_limit: i32,
        grammar: Option<&str>,
        cancel: Option<&AtomicBool>,
        thinking: bool,
        generation: &Generation,
    ) -> Result<(String, String, u32, u32)> {
        // The draft model is the separate `--mtp-model-file` model when present,
        // and otherwise the target model's own nextn/MTP head.
        let mtp_model = self.mtp_model.as_ref().unwrap_or(&self.model);
        let n_ctx = NonZeroU32::new(context_size.try_into()?).context("Invalid context size")?;
        let n_rs_seq = self.mtp_n_max.max(4).try_into()?;

        let target = self.model.new_context(
            &self.backend,
            LlamaContextParams::default()
                .with_n_ctx(Some(n_ctx))
                .with_n_ubatch(self.n_ubatch)
                .with_n_rs_seq(n_rs_seq),
        ).context("Unable to create MTP target context")?;
        let draft = mtp_model.new_context_with_ctx_other(
            &self.backend,
            LlamaContextParams::default()
                .with_n_ctx(Some(n_ctx))
                .with_n_ubatch(self.n_ubatch)
                .with_n_rs_seq(n_rs_seq)
                .with_context_type(LlamaContextType::Mtp),
            &target,
        ).context("Unable to create MTP draft context")?;
        let mut mtp = MtpSpeculative::new(
            target,
            draft,
            MtpSpeculativeParams {
                n_max: self.mtp_n_max,
                ..Default::default()
            },
        ).context("Unable to initialize MTP speculative decoding")?;

        let mut batch = LlamaBatch::new(context_size.try_into()?, 1);
        let last_index = tokens_list.len() - 1;
        for (i, token) in tokens_list.iter().copied().enumerate() {
            batch.add(token, i.try_into()?, &[0], i == last_index)?;
        }
        mtp.target_context_mut().decode(&mut batch)
            .context("MTP target prefill failed")?;
        mtp.process(&batch).context("MTP draft prefill failed")?;
        mtp.begin(&tokens_list).context("MTP generation setup failed")?;

        let mut sampler = create_sampler(&self.model, grammar, generation)?;
        // `LlamaSampler::sample` accepts the sampled token inside llama.cpp;
        // the loop must not accept it again (PH-4a).
        let mut token = sampler.sample(mtp.target_context(), batch.n_tokens() - 1);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let mut n_past = batch.n_tokens();
        let mut proposed = 0_usize;
        let mut accepted_total = 0_usize;
        let mut usage = TokenUsage::default();

        while n_past <= output_limit {
            if is_cancelled(cancel) {
                return Err(LLMError::Cancelled.into());
            }
            if self.model.vocab().is_eog(token) {
                break;
            }
            // The check above excluded the end-of-generation token.
            usage.record_output(false);
            append_token(&self.model.vocab(), token, &mut decoder, &mut output)?;

            let mut drafts = mtp.draft(n_past, token, &tokens_list)
                .context("MTP draft failed")?;
            let draft_pending = !drafts.is_empty();
            drafts.truncate((output_limit - n_past).max(0).try_into()?);
            proposed += drafts.len();

            batch.clear();
            batch.add(token, n_past, &[0], true)?;
            for (i, draft_token) in drafts.iter().copied().enumerate() {
                batch.add(draft_token, n_past + 1 + i32::try_from(i)?, &[0], true)?;
            }

            let draft_rollback = u32::try_from(n_past)?;
            if !mtp.draft_context_mut().clear_kv_cache_seq(
                Some(0),
                Some(draft_rollback),
                None,
            )? {
                anyhow::bail!("MTP draft context refused cache rollback at {n_past}");
            }
            mtp.target_context_mut().decode(&mut batch)
                .context("MTP target verification failed")?;
            mtp.process(&batch).context("MTP draft verification failed")?;

            let mut accepted = 0_usize;
            let mut next = sampler.sample(mtp.target_context(), 0);
            for (i, draft_token) in drafts.iter().copied().enumerate() {
                if next != draft_token {
                    break;
                }
                accepted = i + 1;
                if self.model.vocab().is_eog(next) {
                    break;
                }
                next = sampler.sample(mtp.target_context(), i32::try_from(i + 1)?);
            }

            let new_n_past = n_past + 1 + i32::try_from(accepted)?;
            if accepted < drafts.len() {
                let rollback = Some(u32::try_from(new_n_past)?);
                if !mtp.target_context_mut().clear_kv_cache_seq(Some(0), rollback, None)? {
                    anyhow::bail!("MTP target context refused cache rollback at {new_n_past}");
                }
                if !mtp.draft_context_mut().clear_kv_cache_seq(Some(0), rollback, None)? {
                    anyhow::bail!("MTP draft context refused cache rollback at {new_n_past}");
                }
            }
            if draft_pending {
                mtp.accept(u16::try_from(accepted)?)
                    .context("MTP acceptance update failed")?;
            }

            for draft_token in drafts.iter().copied().take(accepted) {
                append_token(&self.model.vocab(), draft_token, &mut decoder, &mut output)?;
                // An accepted draft counts, unless it is the end-of-generation
                // token, which is never counted (RD-18).
                usage.record_output(self.model.vocab().is_eog(draft_token));
            }
            accepted_total += accepted;
            token = next;
            n_past = new_n_past;
        }

        eprintln!(
            "ltengine: MTP proposed {proposed} tokens, accepted {accepted_total}"
        );
        let (text, trace) = clean_output(output, thinking)?;
        let reasoning_tokens =
            count_reasoning_tokens(&self.model.vocab(), &trace, usage.output_tokens);
        Ok((text, trace, usage.output_tokens, reasoning_tokens))
    }
}

fn load_mtp_model(
    backend: &LlamaBackend,
    path: &PathBuf,
    use_gpu: bool,
) -> Result<LlamaModel> {
    let mut n_gpu = if use_gpu { 9999 } else { 0 };
    loop {
        match LlamaModel::load_from_file(
            backend,
            path,
            &LlamaModelParams::default().with_n_gpu_layers(n_gpu),
        ) {
            Ok(model) => return Ok(model),
            Err(error) if use_gpu && n_gpu > 0 => {
                n_gpu = if n_gpu >= 9999 { 64 } else { n_gpu / 2 };
                eprintln!("ltengine: MTP model load failed, retrying with {n_gpu} GPU layers");
                let _ = error;
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("Unable to load MTP model: {}", path.display()));
            }
        }
    }
}

fn validate_mtp_pair(target: &LlamaModel, draft: &LlamaModel) -> Result<()> {
    let target_vocab = target.vocab();
    let draft_vocab = draft.vocab();
    if target.n_vocab() != draft.n_vocab()
        || target_vocab.vocab_type() != draft_vocab.vocab_type()
        || target_vocab.bos() != draft_vocab.bos()
        || target.n_embd_out() != draft.n_embd_out()
    {
        anyhow::bail!(
            "MTP model is not compatible with the target model: \
             vocab={}/{}, type={:?}/{:?}, BOS={:?}/{:?}, embedding={}/{}",
            target.n_vocab(),
            draft.n_vocab(),
            target_vocab.vocab_type(),
            draft_vocab.vocab_type(),
            target_vocab.bos(),
            draft_vocab.bos(),
            target.n_embd_out(),
            draft.n_embd_out(),
        );
    }
    Ok(())
}

fn probe_mtp(
    backend: &LlamaBackend,
    target_model: &LlamaModel,
    draft_model: &LlamaModel,
    n_max: i32,
    n_ubatch: u32,
) -> Result<()> {
    let n_ctx = NonZeroU32::new(8).unwrap();
    let n_rs_seq = n_max.max(4).try_into()?;
    let target = target_model.new_context(
        backend,
        LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx))
            .with_n_ubatch(n_ubatch)
            .with_n_rs_seq(n_rs_seq),
    ).context("Unable to create MTP target probe context")?;
    let draft = draft_model.new_context_with_ctx_other(
        backend,
        LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx))
            .with_n_ubatch(n_ubatch)
            .with_n_rs_seq(n_rs_seq)
            .with_context_type(LlamaContextType::Mtp),
        &target,
    ).context("Unable to create MTP draft probe context")?;
    MtpSpeculative::new(
        target,
        draft,
        MtpSpeculativeParams {
            n_max,
            ..Default::default()
        },
    ).context("Unable to initialize MTP probe")?;
    Ok(())
}

/// Build the sampler chain. `grammar` prepends a GBNF grammar sampler so every
/// sampled token follows the grammar (PH-4a, RD-19).
fn create_sampler(
    model: &LlamaModel,
    grammar: Option<&str>,
    generation: &Generation,
) -> Result<LlamaSampler> {
    let seq_breakers = vec![b"\n", b":", b"\"", b"*"];
    let mut samplers = Vec::with_capacity(10);
    if let Some(grammar) = grammar {
        samplers.push(LlamaSampler::grammar(model, grammar, "root")?);
    }
    samplers.extend([
        LlamaSampler::penalties(model.n_vocab(), 64, 1.0, 0.0, 0.0),
        LlamaSampler::dry(model, 0.0, 1.75, 2, -1, seq_breakers),
        LlamaSampler::top_k(40),
        LlamaSampler::typical(1.0, 0),
        LlamaSampler::top_p(generation.top_p, 0),
        LlamaSampler::min_p(0.05, 0),
        LlamaSampler::xtc(0.0, 0.1, 0, 42),
        LlamaSampler::temp_ext(generation.temperature, 0.0, 1.0),
        LlamaSampler::dist(42),
    ]);
    Ok(LlamaSampler::chain_simple(samplers))
}

fn append_token(
    vocab: &LlamaVocab<'_>,
    token: LlamaToken,
    decoder: &mut encoding_rs::Decoder,
    output: &mut String,
) -> Result<()> {
    if !vocab.is_eog(token) {
        output.push_str(&decode_piece(decoder, &vocab.token_to_piece(token, true, None)));
    }
    Ok(())
}

/// Decode one token's bytes with the incremental UTF-8 decoder.
///
/// `decode_to_string` never grows its destination. The decoder's bound also
/// accounts for an incomplete UTF-8 sequence retained from the previous token.
fn decode_piece(decoder: &mut encoding_rs::Decoder, bytes: &[u8]) -> String {
    let mut output = String::with_capacity(
        decoder
            .max_utf8_buffer_length(bytes.len())
            .expect("token output is too large to decode"),
    );
    let (result, read, _) = decoder.decode_to_string(bytes, &mut output, false);
    assert!(
        matches!(result, encoding_rs::CoderResult::InputEmpty) && read == bytes.len(),
        "UTF-8 decoder capacity bound must consume the complete token"
    );
    output
}

/// Clean one generated answer and return the removed reasoning trace with it
/// (`RD-38`). The trace is the `SP-NEVER-003` material that must not reach the
/// response text.
fn clean_output(output: String, thinking: bool) -> Result<(String, String)> {
    let (output, mut trace) = strip_thinking_block(output, thinking);
    let output = if let Some(pos) = output.find("<channel|>") {
        // Everything through the channel close marker is trace material.
        trace.push_str(&output[..pos + "<channel|>".len()]);
        output[pos + "<channel|>".len()..].to_owned()
    } else if let Some(rest) = output.strip_prefix("<|channel>thought") {
        trace.push_str("<|channel>thought");
        rest.trim_start_matches(['\n', ' ']).to_owned()
    } else {
        output
    };
    let output = output.replace("<end_of_turn>", "");
    let output = output.trim().to_owned();
    if output.is_empty() {
        anyhow::bail!("Model produced empty output");
    }
    Ok((output, trace))
}

/// Count the tokens of a removed reasoning trace with the loaded model
/// tokenizer (`RD-38`). The count is clamped to the generated count, so it is
/// always a subset of `output_tokens`.
fn count_reasoning_tokens(vocab: &LlamaVocab<'_>, trace: &str, output_tokens: u32) -> u32 {
    if trace.is_empty() {
        return 0;
    }
    let count = vocab.tokenize(trace.as_bytes(), false, true).len();
    u32::try_from(count).unwrap_or(u32::MAX).min(output_tokens)
}

/// Remove a reasoning trace, so it never reaches the response text
/// (`SP-NEVER-003`, `RD-28`).
///
/// Two template shapes produce a trace. A template that leaves the thinking
/// tags to the model emits `<think>...</think>` in the output, which is the
/// leading form. A template that opens the block in its generation prompt
/// (`<|im_start|>assistant\n<think>\n`, the Qwen3.5/3.6 family) leaves the
/// opening tag in the prompt, so the output starts with the reasoning text and
/// ends the trace at a bare `</think>`; that form is removed only when the
/// request asked for thinking. A request without thinking gets a closed block
/// from the same template, so a `</think>` in its output is answer text.
///
/// An unterminated leading block leaves nothing behind, and [`clean_output`]
/// then reports the empty output.
fn strip_thinking_block(output: String, thinking: bool) -> (String, String) {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let trimmed = output.trim_start();
    if let Some(rest) = trimmed.strip_prefix(OPEN) {
        return match rest.find(CLOSE) {
            Some(end) => (
                rest[end + CLOSE.len()..].trim_start().to_owned(),
                format!("{OPEN}{}", &rest[..end + CLOSE.len()]),
            ),
            None => (String::new(), trimmed.to_owned()),
        };
    }
    if thinking
        && let Some(end) = output.find(CLOSE)
    {
        return (
            output[end + CLOSE.len()..].trim_start().to_owned(),
            output[..end + CLOSE.len()].to_owned(),
        );
    }
    (output, String::new())
}

/// Which MTP decode path `LLM::new` selects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MtpMode {
    /// No MTP. The single-token decode path.
    Off,
    /// `--mtp-model-file` names a separate MTP draft model.
    DraftFile,
    /// The target GGUF carries its own nextn/MTP head.
    SelfMtp,
}

/// The MTP decision rule: a draft file wins; otherwise a target that reports a
/// nextn head drafts for itself; otherwise MTP stays off.
fn mtp_mode(has_draft_file: bool, nextn_layers: i32) -> MtpMode {
    if has_draft_file {
        MtpMode::DraftFile
    } else if nextn_layers > 0 {
        MtpMode::SelfMtp
    } else {
        MtpMode::Off
    }
}

/// Parse a `<architecture>.nextn_predict_layers` metadata value. llama.cpp
/// renders an integer metadata value as its decimal string. A missing or
/// unusable value reports `0`.
fn parse_nextn_layers(value: Option<&str>) -> i32 {
    value
        .and_then(|value| value.trim().parse::<i32>().ok())
        .unwrap_or(0)
}

/// Read `<architecture>.nextn_predict_layers` from the model metadata.
fn nextn_predict_layers(model: &LlamaModel) -> i32 {
    let architecture = match model.meta_val_str("general.architecture") {
        Ok(value) => value,
        Err(_) => return 0,
    };
    let key = format!("{}.nextn_predict_layers", architecture.trim());
    parse_nextn_layers(model.meta_val_str(&key).ok().as_deref())
}

fn validate_mtp_n_max(value: i32) -> Result<()> {
    if !(1..=16).contains(&value) {
        anyhow::bail!("MTP draft token count must be between 1 and 16");
    }
    Ok(())
}

impl LLMContext<'_>{
    /// Decode a prompt and return the cleaned text with the number of generated
    /// token IDs, excluding the end-of-generation token (RD-18). `grammar`
    /// constrains the decode when present (PH-4a).
    pub fn process(
        &mut self,
        tokens_list: Vec<LlamaToken>,
        output_limit: i32,
        grammar: Option<&str>,
        cancel: Option<&AtomicBool>,
        thinking: bool,
        generation: &Generation,
    ) -> Result<(String, String, u32, u32)>{
        // We use this object to submit token data for decoding
        let mut batch = LlamaBatch::new(self.ctx_size.try_into()?, 1);

        let last_index: i32 = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            // llama_decode will output logits only for the last token of the prompt
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }

        self.ctx.decode(&mut batch)
            .with_context(|| "llama_decode() failed")?;

        let mut n_cur = batch.n_tokens();

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut sampler = create_sampler(&self.llm.model, grammar, generation)?;

        let mut output = String::new();
        let mut usage = TokenUsage::default();

        while n_cur <= output_limit {
            if is_cancelled(cancel) {
                return Err(LLMError::Cancelled.into());
            }

            // sample the next token
            {
                let token = sampler.sample(&self.ctx, batch.n_tokens() - 1);

                // `llama_sampler_sample` accepts the token inside llama.cpp,
                // so the loop must not accept it a second time. A second accept
                // advances a grammar past its end and exhausts it (PH-4a).
                // is it an end of stream?
                if self.llm.model.vocab().is_eog(token) {
                    break;
                }

                // RD-18: count the generated token, end-of-generation excluded.
                usage.record_output(false);
                append_token(&self.llm.model.vocab(), token, &mut decoder, &mut output)?;

                batch.clear();
                batch.add(token, n_cur, &[0], true)?;
            }

            n_cur += 1;

            self.ctx.decode(&mut batch).with_context(|| "Failed to eval")?;
        }

        let (text, trace) = clean_output(output, thinking)?;
        let reasoning_tokens =
            count_reasoning_tokens(&self.llm.model.vocab(), &trace, usage.output_tokens);
        Ok((text, trace, usage.output_tokens, reasoning_tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Generation, clean_output, context_positions, context_rejection, mtp_mode,
        output_ceiling, parse_nextn_layers, reasoning_kwargs, validate_mtp_n_max, MtpMode, Reasoning,
        TokenUsage,
    };

    /// `PH8-06`, preserved behavior: with no `max_output_tokens`, the ceiling
    /// and the context positions stay the pre-`RD-29` `3 × prompt`.
    #[test]
    fn ph8_06_absent_ceiling_is_three_times_the_prompt() {
        assert_eq!(output_ceiling(100, None), 300);
        assert_eq!(context_positions(100, None, 0), 300);
        assert_eq!(context_positions(100, None, 4), 304);
        assert_eq!(Generation::default().max_output_tokens, None);
    }

    /// `PH8-05`, changed behavior: the inclusive loop bound generates exactly
    /// `max_output_tokens` positions.
    #[test]
    fn ph8_05_ceiling_is_one_less_than_the_exclusive_budget() {
        assert_eq!(output_ceiling(100, Some(1)), 100);
        assert_eq!(output_ceiling(100, Some(50)), 149);
        assert_eq!(context_positions(100, Some(1), 0), 101);
    }

    /// `PH8-04`, changed behavior: a ceiling whose positions exceed the model's
    /// training context is rejected, and the value is not reduced. A ceiling
    /// that fits is accepted.
    #[test]
    fn ph8_04_context_rejection_names_max_output_tokens() {
        assert_eq!(output_ceiling(100, Some(50)), 149);
        assert_eq!(context_positions(100, Some(50), 4), 154);
        assert!(context_rejection(100, Some(50), 0, 150).is_none());
        let message = context_rejection(100, Some(50), 0, 149).expect("over the training context");
        assert!(message.contains("max_output_tokens"), "{message}");
        assert!(message.contains("150"), "{message}");
        // The MTP margin counts toward the requirement.
        assert!(context_rejection(100, Some(50), 4, 150).is_some());
        // No ceiling is never rejected.
        assert!(context_rejection(100, None, 4, 1).is_none());
    }

    #[test]
    fn selects_self_mtp_only_for_a_baked_in_head() {
        assert_eq!(mtp_mode(true, 0), MtpMode::DraftFile);
        assert_eq!(mtp_mode(true, 4), MtpMode::DraftFile);
        assert_eq!(mtp_mode(false, 1), MtpMode::SelfMtp);
        assert_eq!(mtp_mode(false, 0), MtpMode::Off);
        assert_eq!(mtp_mode(false, -1), MtpMode::Off);
    }

    #[test]
    fn parses_the_nextn_layer_count() {
        assert_eq!(parse_nextn_layers(Some("1")), 1);
        assert_eq!(parse_nextn_layers(Some(" 2 ")), 2);
        assert_eq!(parse_nextn_layers(Some("junk")), 0);
        assert_eq!(parse_nextn_layers(Some("")), 0);
        assert_eq!(parse_nextn_layers(None), 0);
    }

    #[test]
    fn usage_total_is_the_exact_sum_and_eog_is_not_counted() {
        let mut usage = TokenUsage {
            input_tokens: 5,
            output_tokens: 0,
            ..Default::default()
        };
        assert_eq!(usage.total_tokens(), 5);
        usage.record_output(false);
        usage.record_output(true);
        assert_eq!(usage.output_tokens, 1);
        assert_eq!(usage.total_tokens(), 6);
    }

    #[test]
    fn validates_mtp_draft_count() {
        assert!(validate_mtp_n_max(1).is_ok());
        assert!(validate_mtp_n_max(16).is_ok());
        assert!(validate_mtp_n_max(0).is_err());
        assert!(validate_mtp_n_max(17).is_err());
    }

    #[test]
    fn removes_gemma_thinking_output() {
        assert_eq!(
            clean_output("<|channel>thought\nreason<channel|>answer".into(), false).unwrap().0,
            "answer"
        );
        assert_eq!(
            clean_output("<|channel>thought answer<end_of_turn>".into(), false).unwrap().0,
            "answer"
        );
    }

    /// `RD-28`, `SP-NEVER-003`, preserved behavior (AC-6): the reasoning trace
    /// never reaches the text.
    #[test]
    fn removes_a_leading_thinking_block() {
        assert_eq!(
            clean_output("<think>\nreasoning here\n</think>\nThe answer.".into(), false).unwrap().0,
            "The answer."
        );
        assert_eq!(
            clean_output("<think>reasoning</think>Answer".into(), false).unwrap().0,
            "Answer"
        );
    }

    /// An unterminated block leaves no answer behind, which is an error rather
    /// than a trace in the response text (AC-7).
    #[test]
    fn rejects_output_that_is_only_a_thinking_block() {
        assert!(clean_output("<think>reasoning without an end".into(), false).is_err());
    }

    /// `RD-28`, preserved behavior: an output with no thinking block is
    /// returned unchanged (AC-8).
    #[test]
    fn keeps_output_without_a_thinking_block() {
        assert_eq!(
            clean_output("plain answer".into(), false).unwrap().0,
            "plain answer"
        );
    }

    /// AC-1: a template that opens the thinking block in its generation prompt
    /// leaves the marker in the prompt, so the trace arrives with no leading
    /// `<think>` and ends at a bare `</think>`.
    #[test]
    fn ac1_strips_a_trace_ended_by_a_bare_close_marker() {
        assert_eq!(
            clean_output("The user asks X.\n</think>\n\nOK".into(), true).unwrap().0,
            "OK"
        );
    }

    /// AC-2: a trace that is empty is a leading close marker.
    #[test]
    fn ac2_strips_a_leading_close_marker() {
        assert_eq!(clean_output("</think>\n\nOK".into(), true).unwrap().0, "OK");
    }

    /// AC-3: a response that carries only a trace is an error, exactly as for
    /// the leading `<think>` form (AC-7).
    #[test]
    fn ac3_rejects_output_that_is_only_a_trace() {
        assert!(clean_output("reasoning\n</think>".into(), true).is_err());
    }

    /// AC-4: the first close marker ends the trace, so a later one is text.
    #[test]
    fn ac4_strips_only_through_the_first_close_marker() {
        assert_eq!(
            clean_output("trace</think>answer</think>more".into(), true).unwrap().0,
            "answer</think>more"
        );
    }

    /// AC-5: a request that did not ask for thinking gets a closed block from
    /// the template, so a close marker in the output is answer text.
    #[test]
    fn ac5_keeps_a_close_marker_when_thinking_was_not_requested() {
        assert_eq!(
            clean_output("text </think> more".into(), false).unwrap().0,
            "text </think> more"
        );
    }

    /// AC-6: the leading `<think>...</think>` form is stripped under both
    /// values, so gating the bare-marker rule does not change it.
    #[test]
    fn ac6_strips_a_leading_block_for_both_values() {
        for thinking in [false, true] {
            assert_eq!(
                clean_output("<think>\nreasoning\n</think>\nAnswer.".into(), thinking).unwrap().0,
                "Answer."
            );
        }
    }

    /// `RD-28`: the effort reaches the template as JSON text, and an absent
    /// effort adds no keyword.
    #[test]
    fn encodes_the_reasoning_effort_as_json() {
        assert_eq!(
            reasoning_kwargs(&Reasoning {
                thinking: true,
                effort: Some("low"),
            }),
            vec![("reasoning_effort", "\"low\"".to_string())]
        );
        assert!(reasoning_kwargs(&Reasoning {
            thinking: true,
            effort: None,
        })
        .is_empty());
    }
}
