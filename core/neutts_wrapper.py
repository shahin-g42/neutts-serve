"""
NeuTTSAir Wrapper for integration with the TTS service.
Supports multiple backends: Transformers, vLLM, GGUF with streaming.
"""

import re
import numpy as np
import torch
from typing import Generator, AsyncGenerator, Optional
from pathlib import Path

from core.logging import app_logger
from core.settings import settings


def _linear_overlap_add(frames: list[np.ndarray], stride: int) -> np.ndarray:
    """
    Linear overlap-add for smooth audio chunk concatenation.
    Original implementation from encodec: https://github.com/facebookresearch/encodec
    """
    assert len(frames)
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]

    total_size = 0
    for i, frame in enumerate(frames):
        frame_end = stride * i + frame.shape[-1]
        total_size = max(total_size, frame_end)

    sum_weight = np.zeros(total_size, dtype=dtype)
    out = np.zeros(*shape, total_size, dtype=dtype)

    offset: int = 0
    for frame in frames:
        frame_length = frame.shape[-1]
        t = np.linspace(0, 1, frame_length + 2, dtype=dtype)[1:-1]
        weight = np.abs(0.5 - (t - 0.5))

        out[..., offset : offset + frame_length] += weight * frame
        sum_weight[offset : offset + frame_length] += weight
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight


class NeuTTSAirWrapper:
    """
    Wrapper for NeuTTS-Air model with streaming support.
    Integrates NeuCodec for audio encoding/decoding and phonemizer for text preprocessing.
    Supports multiple inference backends: Transformers, vLLM AsyncLLMEngine, and GGUF.
    """
    
    def __init__(
        self,
        backbone_repo: str = "neuphonic/neutts-air",
        backbone_device: str = "cuda",
        codec_repo: str = "neuphonic/neucodec",
        codec_device: str = "cuda",
        use_vllm: bool = False,
        use_gguf: bool = False,
        use_async_engine: bool = False,
        max_context: int = 2048,
        gpu_memory_utilization: float = 0.7
    ):
        """
        Initialize the NeuTTSAir wrapper.
        
        Args:
            backbone_repo: Model repository for NeuTTS-Air backbone
            backbone_device: Device for backbone (cpu/cuda)
            codec_repo: Repository for NeuCodec
            codec_device: Device for codec (cpu/cuda)
            use_vllm: Enable vLLM acceleration
            use_gguf: Enable GGUF quantization
            use_async_engine: Use AsyncLLMEngine for vLLM streaming
            max_context: Maximum context length
            gpu_memory_utilization: GPU memory utilization for vLLM
        """
        # Constants
        self.sample_rate = 24_000
        self.max_context = max_context
        self.hop_length = settings.hop_length
        
        # Streaming parameters
        self.streaming_overlap_frames = settings.streaming_overlap_frames
        self.streaming_frames_per_chunk = settings.streaming_frames_per_chunk
        self.streaming_lookforward = settings.streaming_lookforward
        self.streaming_lookback = settings.streaming_lookback
        self.streaming_stride_samples = self.streaming_frames_per_chunk * self.hop_length
        
        # Backend flags
        self._is_quantized_model = use_gguf
        self._use_vllm = use_vllm and not use_gguf
        self._use_async_engine = use_async_engine and self._use_vllm
        
        # Model components
        self.tokenizer = None
        self.backbone = None
        self.codec = None
        self.phonemizer = None
        
        # Load components
        app_logger.info("Initializing NeuTTSAir wrapper...")
        self._load_phonemizer()
        self._load_backbone(backbone_repo, backbone_device, gpu_memory_utilization)
        self._load_codec(codec_repo, codec_device)
        
        # Load watermarker (optional - install perth if available)
        try:
            import perth
            if hasattr(perth, 'PerthImplicitWatermarker'):
                self.watermarker = perth.PerthImplicitWatermarker()
                app_logger.info("Watermarker loaded successfully")
            else:
                app_logger.warning("perth module found but PerthImplicitWatermarker not available")
                self.watermarker = None
        except (ImportError, AttributeError) as e:
            app_logger.warning(f"perth watermarker not available - audio will not be watermarked: {e}")
            self.watermarker = None
        
        app_logger.info("✓ NeuTTSAir wrapper initialized successfully")
    
    def _load_phonemizer(self):
        """Load the phonemizer backend for text-to-phoneme conversion."""
        if not settings.enable_phonemization:
            app_logger.info("Phonemization disabled - text will be passed directly")
            self.phonemizer = None
            return
        
        app_logger.info(f"Loading phonemizer with language={settings.phonemizer_language}...")
        from phonemizer.backend import EspeakBackend
        
        self.phonemizer = EspeakBackend(
            language=settings.phonemizer_language,
            preserve_punctuation=settings.phonemizer_preserve_punctuation,
            with_stress=settings.phonemizer_with_stress
        )
        app_logger.info(f"✓ Phonemizer loaded (language: {settings.phonemizer_language})")
    
    def _load_backbone(self, backbone_repo: str, backbone_device: str, gpu_memory_utilization: float):
        """Load the NeuTTS-Air backbone model."""
        app_logger.info(f"Loading backbone from: {backbone_repo} on {backbone_device}...")
        
        # GGUF loading
        if self._is_quantized_model or backbone_repo.endswith("gguf"):
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "Failed to import `llama_cpp`. "
                    "Please install it with: pip install llama-cpp-python"
                ) from e
            
            self.backbone = Llama.from_pretrained(
                repo_id=backbone_repo,
                filename="*.gguf",
                verbose=False,
                n_gpu_layers=-1 if backbone_device == "cuda" else 0,
                n_ctx=self.max_context,
                mlock=True,
                flash_attn=True if backbone_device == "cuda" else False,
            )
            self._is_quantized_model = True
            app_logger.info("✓ GGUF backbone loaded")
        
        # vLLM loading (AsyncLLMEngine or standard LLM)
        elif self._use_vllm:
            import vllm
            from vllm.engine.arg_utils import AsyncEngineArgs
            
            if self._use_async_engine:
                # Use AsyncLLMEngine for streaming
                app_logger.info("Loading vLLM AsyncLLMEngine...")
                engine_args = AsyncEngineArgs(
                    model=backbone_repo,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=self.max_context,
                    enforce_eager=False,  # Enable CUDA graphs
                )
                self.backbone = vllm.AsyncLLMEngine.from_engine_args(engine_args)
                app_logger.info("✓ vLLM AsyncLLMEngine loaded")
            else:
                # Use standard vLLM.LLM for non-streaming
                app_logger.info("Loading vLLM.LLM...")
                self.backbone = vllm.LLM(
                    model=backbone_repo,
                    seed=settings.seed,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_model_len=self.max_context,
                )
                app_logger.info("✓ vLLM.LLM loaded")
            
            # Load tokenizer for vLLM
            app_logger.info("Loading tokenizer from transformers...")
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo)
                app_logger.info("✓ Tokenizer loaded")
            except Exception as e:
                app_logger.error(f"Failed to load tokenizer: {e}")
                raise
        
        # Standard Transformers loading
        else:
            app_logger.info("Loading standard transformers model...")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_repo)
            self.backbone = AutoModelForCausalLM.from_pretrained(backbone_repo).to(
                torch.device(backbone_device)
            )
            self.backbone.eval()
            app_logger.info("✓ Transformers backbone loaded")
    
    def _load_codec(self, codec_repo: str, codec_device: str):
        """Load the NeuCodec model for audio encoding/decoding."""
        app_logger.info(f"Loading codec from: {codec_repo} on {codec_device}...")
        from neucodec import NeuCodec
        
        self.codec = NeuCodec.from_pretrained(codec_repo)
        self.codec.eval().to(torch.device(codec_device))
        app_logger.info("✓ NeuCodec loaded")
    
    def encode_reference(self, ref_audio_path: str | Path) -> np.ndarray:
        """
        Encode reference audio to NeuCodec FSQ codes.
        
        Args:
            ref_audio_path: Path to reference audio file
            
        Returns:
            FSQ codes as numpy array
        """
        import librosa
        import torchaudio
        from torchaudio import transforms as T
        
        # Load audio at 16kHz (NeuCodec encoding sample rate)
        wav, sr = torchaudio.load(str(ref_audio_path))
        if sr != 16_000:
            wav = T.Resample(sr, 16_000)(wav)
        
        # Ensure mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Add batch dimension: [1, 1, T]
        wav_tensor = wav.unsqueeze(0) if wav.dim() == 2 else wav
        
        with torch.no_grad():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor)
            ref_codes = ref_codes.squeeze(0).squeeze(0)  # Remove batch and channel dims
        
        return ref_codes.cpu().numpy()
    
    def _to_phones(self, text: str) -> str:
        """Convert text to phonemes (or return as-is if phonemization disabled)."""
        if not settings.enable_phonemization or self.phonemizer is None:
            # Return text directly without phonemization
            return text
        
        # Phonemize the text
        phones = self.phonemizer.phonemize([text])
        phones = phones[0].split()
        phones = " ".join(phones)
        return phones
    
    def _decode(self, codes_str: str) -> np.ndarray:
        """
        Decode speech tokens to waveform.
        
        Args:
            codes_str: String containing speech tokens like "<|speech_123|>"
            
        Returns:
            Audio waveform as numpy array with shape [1, samples] (matching NeuCodec reference)
        """
        # Extract speech token IDs using regex
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes_str)]
        
        if len(speech_ids) == 0:
            raise ValueError("No valid speech tokens found in the output.")
        
        with torch.no_grad():
            codes = torch.tensor(speech_ids, dtype=torch.long)[None, None, :].to(
                self.codec.device
            )
            recon = self.codec.decode_code(codes).cpu().numpy()  # Shape: [B, 1, T_24]
        
        # Return [1, samples] format matching NeuCodec reference: recon[0, :, :]
        return recon[0, :, :]
    
    def _apply_watermark(self, wav: np.ndarray) -> np.ndarray:
        """Apply watermark if available."""
        if self.watermarker is not None:
            return self.watermarker.apply_watermark(wav, sample_rate=self.sample_rate)
        return wav
    
    def _apply_chat_template(self, ref_codes: np.ndarray | list, ref_text: str, input_text: str) -> list[int]:
        """
        Build chat template prompt for NeuTTS-Air.
        
        Args:
            ref_codes: Reference audio codes
            ref_text: Reference text transcription
            input_text: Text to synthesize
            
        Returns:
            Token IDs for the prompt
        """
        if isinstance(ref_codes, np.ndarray):
            ref_codes = ref_codes.tolist()
        
        # Phonemize texts
        input_text = self._to_phones(ref_text) + " " + self._to_phones(input_text)
        
        # Get special token IDs
        speech_replace = self.tokenizer.convert_tokens_to_ids("<|SPEECH_REPLACE|>")
        speech_gen_start = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        text_replace = self.tokenizer.convert_tokens_to_ids("<|TEXT_REPLACE|>")
        text_prompt_start = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
        text_prompt_end = self.tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")
        
        # Tokenize input text
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        
        # Build base chat template
        chat = """user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"""
        ids = self.tokenizer.encode(chat)
        
        # Replace text placeholder
        text_replace_idx = ids.index(text_replace)
        ids = (
            ids[:text_replace_idx]
            + [text_prompt_start]
            + input_ids
            + [text_prompt_end]
            + ids[text_replace_idx + 1:]
        )
        
        # Replace speech placeholder with reference codes
        speech_replace_idx = ids.index(speech_replace)
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
        codes = self.tokenizer.encode(codes_str, add_special_tokens=False)
        ids = ids[:speech_replace_idx] + [speech_gen_start] + list(codes)
        
        return ids
    
    async def infer_async(self, text: str, ref_codes: np.ndarray | list, ref_text: str) -> np.ndarray:
        """
        Perform async inference to generate speech from text (for AsyncLLMEngine).
        
        Args:
            text: Input text to synthesize
            ref_codes: Encoded reference audio codes
            ref_text: Reference text transcription
            
        Returns:
            Generated speech waveform
        """
        prompt_ids = self._apply_chat_template(ref_codes, ref_text, text)
        output_str = await self._infer_vllm_async(prompt_ids)
        
        # Decode to audio
        wav = self._decode(output_str)
        watermarked_wav = self._apply_watermark(wav)
        
        return watermarked_wav
    
    def infer(self, text: str, ref_codes: np.ndarray | list, ref_text: str) -> np.ndarray:
        """
        Perform inference to generate speech from text (synchronous wrapper).
        
        Args:
            text: Input text to synthesize
            ref_codes: Encoded reference audio codes
            ref_text: Reference text transcription
            
        Returns:
            Generated speech waveform
        """
        if self._is_quantized_model:
            output_str = self._infer_ggml(ref_codes, ref_text, text)
        elif self._use_vllm and not self._use_async_engine:
            # Standard vLLM.LLM (synchronous)
            prompt_ids = self._apply_chat_template(ref_codes, ref_text, text)
            output_str = self._infer_vllm_sync(prompt_ids)
        else:
            # Standard Transformers (synchronous)
            prompt_ids = self._apply_chat_template(ref_codes, ref_text, text)
            output_str = self._infer_torch(prompt_ids)
        
        # Decode to audio
        wav = self._decode(output_str)
        watermarked_wav = self._apply_watermark(wav)
        
        return watermarked_wav
    
    def _infer_torch(self, prompt_ids: list[int]) -> str:
        """
        Inference using standard PyTorch transformers backend (non-vLLM).
        
        Args:
            prompt_ids: Token IDs for the prompt
            
        Returns:
            Generated text containing speech tokens
        """
        # Standard transformers - backbone has .device attribute
        prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self.backbone.device)
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        
        with torch.no_grad():
            output_tokens = self.backbone.generate(
                prompt_tensor,
                max_length=self.max_context,
                eos_token_id=speech_end_id,
                do_sample=True,
                temperature=settings.temperature,
                top_k=settings.top_k,
                use_cache=True,
                min_new_tokens=settings.min_tokens,
            )
        
        input_length = prompt_tensor.shape[-1]
        output_str = self.tokenizer.decode(
            output_tokens[0, input_length:].cpu().numpy().tolist(),
            skip_special_tokens=False
        )
        
        return output_str
    
    def _infer_vllm_sync(self, prompt_ids: list[int]) -> str:
        """
        Inference using standard vLLM.LLM (synchronous).
        
        Args:
            prompt_ids: Token IDs for the prompt
            
        Returns:
            Generated text containing speech tokens
        """
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        
        # vLLM sampling parameters
        sampling_params = SamplingParams(
            max_tokens=self.max_context,
            min_tokens=settings.min_tokens,
            temperature=settings.temperature,
            top_k=settings.top_k,
            top_p=settings.top_p,
            stop_token_ids=[speech_end_id],
        )
        
        # Standard vLLM.LLM requires TokensPrompt type
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)
        outputs = self.backbone.generate(
            prompts=[prompt],
            sampling_params=sampling_params
        )
        output_str = outputs[0].outputs[0].text
        
        return output_str
    
    async def _infer_vllm_async(self, prompt_ids: list[int]) -> str:
        """
        Inference using vLLM AsyncLLMEngine (asynchronous).
        
        Args:
            prompt_ids: Token IDs for the prompt
            
        Returns:
            Generated text containing speech tokens
        """
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        
        # vLLM sampling parameters
        sampling_params = SamplingParams(
            max_tokens=self.max_context,
            min_tokens=settings.min_tokens,
            temperature=settings.temperature,
            top_k=settings.top_k,
            top_p=settings.top_p,
            stop_token_ids=[speech_end_id],
        )
        
        # AsyncLLMEngine requires TokensPrompt type
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)
        request_id = f"req-{id(prompt_ids)}"
        final_output = None
        
        async for output in self.backbone.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id
        ):
            final_output = output
        
        return final_output.outputs[0].text if final_output else ""
    
    def _infer_ggml(self, ref_codes: np.ndarray | list, ref_text: str, input_text: str) -> str:
        """
        Inference using GGUF/llama-cpp backend.
        
        Args:
            ref_codes: Reference audio codes
            ref_text: Reference text
            input_text: Text to synthesize
            
        Returns:
            Generated text containing speech tokens
        """
        if isinstance(ref_codes, np.ndarray):
            ref_codes = ref_codes.tolist()
        
        ref_text = self._to_phones(ref_text)
        input_text = self._to_phones(input_text)
        
        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )
        
        output = self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=settings.temperature,
            top_k=settings.top_k,
            stop=["<|SPEECH_GENERATION_END|>"],
        )
        output_str = output["choices"][0]["text"]
        
        return output_str