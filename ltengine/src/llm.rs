use llama_cpp_2::context::params::{LlamaContextParams, LlamaContextType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, LlamaChatMessage};
use llama_cpp_2::chat::LlamaMinjaChatTemplate;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::speculative::{MtpSpeculative, MtpSpeculativeParams};
use llama_cpp_2::{send_logs_to_tracing, LogOptions};
use std::num::NonZeroU32;
use std::path::PathBuf;
use parking_lot::Mutex;
use std::time::Duration;
use anyhow::{Result, Context};

#[derive(Debug, thiserror::Error)]
pub enum LLMError {
    #[error("Server busy, please try again later")]
    Busy,
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

    pub fn run_prompt(&self, system: String, user: String) -> Result<String>{
        self.run_prompt_reasoning(system, user, &Reasoning::default())
    }

    /// Run one generation with explicit reasoning controls (`RD-28`).
    ///
    /// [`LLM::run_prompt`] delegates with [`Reasoning::default`], so every
    /// caller that predates `RD-28` keeps its behavior.
    pub fn run_prompt_reasoning(
        &self,
        system: String,
        user: String,
        reasoning: &Reasoning<'_>,
    ) -> Result<String> {
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
        // cannot be assumed to carry one. AddBos::Always maps to
        // add_special=true and lets the vocabulary decide.
        let tokens_list = self.model
            .str_to_token(&llm_input, AddBos::Always)
            .with_context(|| "Failed to tokenize prompt")?;
        // for token in &tokens_list {
        //     eprint!("{} {} | ", self.model.token_to_str(*token, Special::Tokenize)?, token);
        // }
        let ctx_size: i32 = tokens_list.len() as i32 * 3;
        // Lock before create_context: context allocation uses GPU resources and
        // two concurrent allocations corrupt each other even before inference starts.
        // TODO: The llama bindings (or llama itself?) do not appear to be totally thread-safe
        // as garbage starts to come out when we run inference in parallel
        // this might need to be investigated and fixed. For now we lock and process requests
        // one at a time.
        let _lock = self.prompt_lock.try_lock_for(Duration::from_secs(120))
            .ok_or(LLMError::Busy)?;
        if self.mtp_model.is_some() || self.self_mtp {
            self.process_mtp(tokens_list, ctx_size)
        } else {
            let mut ctx = self.create_context(ctx_size)?;
            ctx.process(tokens_list)
        }
    }

    fn process_mtp(&self, tokens_list: Vec<LlamaToken>, output_limit: i32) -> Result<String> {
        // The draft model is the separate `--mtp-model-file` model when present,
        // and otherwise the target model's own nextn/MTP head.
        let mtp_model = self.mtp_model.as_ref().unwrap_or(&self.model);
        let context_size = output_limit + self.mtp_n_max + 1;
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

        let mut sampler = create_sampler(&self.model);
        let mut token = sampler.sample(mtp.target_context(), batch.n_tokens() - 1);
        sampler.accept(token);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let mut n_past = batch.n_tokens();
        let mut proposed = 0_usize;
        let mut accepted_total = 0_usize;

        while n_past <= output_limit {
            if self.model.is_eog_token(token) {
                break;
            }
            append_token(&self.model, token, &mut decoder, &mut output)?;

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
            sampler.accept(next);
            for (i, draft_token) in drafts.iter().copied().enumerate() {
                if next != draft_token {
                    break;
                }
                accepted = i + 1;
                if self.model.is_eog_token(next) {
                    break;
                }
                next = sampler.sample(mtp.target_context(), i32::try_from(i + 1)?);
                sampler.accept(next);
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
                append_token(&self.model, draft_token, &mut decoder, &mut output)?;
            }
            accepted_total += accepted;
            token = next;
            n_past = new_n_past;
        }

        eprintln!(
            "ltengine: MTP proposed {proposed} tokens, accepted {accepted_total}"
        );
        clean_output(output)
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
    if target.n_vocab() != draft.n_vocab()
        || target.vocab_type() != draft.vocab_type()
        || target.token_bos() != draft.token_bos()
        || target.n_embd_out() != draft.n_embd_out()
    {
        anyhow::bail!(
            "MTP model is not compatible with the target model: \
             vocab={}/{}, type={:?}/{:?}, BOS={:?}/{:?}, embedding={}/{}",
            target.n_vocab(),
            draft.n_vocab(),
            target.vocab_type(),
            draft.vocab_type(),
            target.token_bos(),
            draft.token_bos(),
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

fn create_sampler(model: &LlamaModel) -> LlamaSampler {
    let seq_breakers = vec![b"\n", b":", b"\"", b"*"];
    LlamaSampler::chain_simple([
        LlamaSampler::penalties(model.n_vocab(), 64, 1.0, 0.0, 0.0),
        LlamaSampler::dry(model, 0.0, 1.75, 2, -1, seq_breakers),
        LlamaSampler::top_k(40),
        LlamaSampler::typical(1.0, 0),
        LlamaSampler::top_p(0.95, 0),
        LlamaSampler::min_p(0.05, 0),
        LlamaSampler::xtc(0.0, 0.1, 0, 42),
        LlamaSampler::temp_ext(0.0, 0.0, 1.0),
        LlamaSampler::dist(42),
    ])
}

fn append_token(
    model: &LlamaModel,
    token: LlamaToken,
    decoder: &mut encoding_rs::Decoder,
    output: &mut String,
) -> Result<()> {
    if !model.is_eog_token(token) {
        output.push_str(&model.token_to_piece(token, decoder, true, None)?);
    }
    Ok(())
}

fn clean_output(output: String) -> Result<String> {
    let output = strip_thinking_block(output);
    let output = if let Some(pos) = output.find("<channel|>") {
        output[pos + "<channel|>".len()..].to_owned()
    } else if let Some(rest) = output.strip_prefix("<|channel>thought") {
        rest.trim_start_matches(['\n', ' ']).to_owned()
    } else {
        output
    };
    let output = output.replace("<end_of_turn>", "");
    let output = output.trim().to_owned();
    if output.is_empty() {
        anyhow::bail!("Model produced empty output");
    }
    Ok(output)
}

/// Remove a leading thinking block, so a reasoning trace never reaches the
/// response text (`SP-NEVER-003`, `RD-28`).
///
/// An unterminated block leaves nothing behind, and [`clean_output`] then
/// reports the empty output. An output with no leading block is unchanged.
fn strip_thinking_block(output: String) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let Some(rest) = output.trim_start().strip_prefix(OPEN) else {
        return output;
    };
    match rest.find(CLOSE) {
        Some(end) => rest[end + CLOSE.len()..].trim_start().to_owned(),
        None => String::new(),
    }
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
    pub fn process(&mut self, tokens_list: Vec<LlamaToken>) -> Result<String>{
        // let ctx_size: i32 = tokens_list.len() as i32 * 3;
        
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
        let mut sampler = create_sampler(&self.llm.model);

        let mut output = String::new();

        while n_cur <= self.ctx_size {

            // sample the next token
            {
                let token = sampler.sample(&self.ctx, batch.n_tokens() - 1);

                sampler.accept(token);

                // is it an end of stream?
                if self.llm.model.is_eog_token(token) {
                    break;
                }
                    
                append_token(&self.llm.model, token, &mut decoder, &mut output)?;

                batch.clear();
                batch.add(token, n_cur, &[0], true)?;
            }

            n_cur += 1;

            self.ctx.decode(&mut batch).with_context(|| "Failed to eval")?;
        }

        clean_output(output)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        clean_output, mtp_mode, parse_nextn_layers, reasoning_kwargs, validate_mtp_n_max, MtpMode,
        Reasoning,
    };

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
    fn validates_mtp_draft_count() {
        assert!(validate_mtp_n_max(1).is_ok());
        assert!(validate_mtp_n_max(16).is_ok());
        assert!(validate_mtp_n_max(0).is_err());
        assert!(validate_mtp_n_max(17).is_err());
    }

    #[test]
    fn removes_gemma_thinking_output() {
        assert_eq!(
            clean_output("<|channel>thought\nreason<channel|>answer".into()).unwrap(),
            "answer"
        );
        assert_eq!(
            clean_output("<|channel>thought answer<end_of_turn>".into()).unwrap(),
            "answer"
        );
    }

    /// `RD-28`, `SP-NEVER-003`: the reasoning trace never reaches the text.
    #[test]
    fn removes_a_leading_thinking_block() {
        assert_eq!(
            clean_output("<think>\nreasoning here\n</think>\nThe answer.".into()).unwrap(),
            "The answer."
        );
        assert_eq!(
            clean_output("<think>reasoning</think>Answer".into()).unwrap(),
            "Answer"
        );
    }

    /// An unterminated block leaves no answer behind, which is an error rather
    /// than a trace in the response text.
    #[test]
    fn rejects_output_that_is_only_a_thinking_block() {
        assert!(clean_output("<think>reasoning without an end".into()).is_err());
    }

    /// `RD-28`, preserved behavior: an output with no thinking block is
    /// returned unchanged.
    #[test]
    fn keeps_output_without_a_thinking_block() {
        assert_eq!(clean_output("plain answer".into()).unwrap(), "plain answer");
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
