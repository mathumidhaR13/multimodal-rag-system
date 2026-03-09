from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from typing import Optional
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMGenerator:
    """
    Handles answer generation using a local LLM (FLAN-T5).

    Model: google/flan-t5-base (default)
    - Instruction-tuned T5 model
    - Great for question answering with context
    - Runs fully locally — no API key needed
    - Downloads automatically on first use (~1GB)

    Pipeline:
        Context + Question
            ↓
        Build Prompt
            ↓
        Tokenize
            ↓
        Generate Answer Tokens
            ↓
        Decode → Answer String
    """

    _instance = None  # Singleton — load model only once

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info(
            f"🧠 Loading LLM: {settings.LLM_MODEL}"
        )
        logger.info(
            "⏳ First run will download model (~1GB). Please wait..."
        )

        try:
            # Detect device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"💻 Using device: {self.device}")

            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(
                settings.LLM_MODEL
            )

            # Load model
            self.model = T5ForConditionalGeneration.from_pretrained(
                settings.LLM_MODEL
            )

            # Move model to device
            self.model.to(self.device)
            self.model.eval()  # Set to inference mode

            self._initialized = True

            logger.info(
                f"✅ LLM loaded successfully | "
                f"Model: {settings.LLM_MODEL} | "
                f"Device: {self.device}"
            )

        except Exception as e:
            logger.error(f"❌ Failed to load LLM: {e}")
            raise

    def build_prompt(
        self,
        question: str,
        context: str
    ) -> str:
        """
        Build a structured prompt for FLAN-T5.

        FLAN-T5 works best with explicit instruction-style prompts.
        The prompt guides the model to answer ONLY from context.

        Args:
            question: User's question
            context:  Retrieved document chunks as context

        Returns:
            Formatted prompt string
        """
        prompt = (
            f"Answer the question based on the context below.\n"
            f"If the answer cannot be found in the context, "
            f"say 'I don't have enough information to answer this.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        logger.debug(
            f"  📝 Prompt built | Length: {len(prompt)} chars"
        )

        return prompt

    def generate(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        num_beams: int = 4,
        early_stopping: bool = True,
    ) -> str:
        """
        Generate an answer from the LLM given a question and context.

        Args:
            question:       User's question
            context:        Retrieved context from RAG pipeline
            max_new_tokens: Max tokens in generated answer
            temperature:    Sampling temperature (lower = more focused)
            num_beams:      Beam search width (higher = better quality)
            early_stopping: Stop when all beams reach EOS token

        Returns:
            Generated answer string
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")

        if not context.strip():
            logger.warning(
                "⚠️  Empty context provided — "
                "LLM will answer from general knowledge"
            )

        # Build prompt
        prompt = self.build_prompt(question, context)

        logger.info(
            f"🧠 Generating answer | "
            f"Question: '{question[:60]}...'"
        )

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding=True
            ).to(self.device)

            logger.debug(
                f"  Tokenized | Input tokens: {inputs['input_ids'].shape[1]}"
            )

            # Generate answer
            with torch.no_grad():
                output_tokens = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    early_stopping=early_stopping,
                    temperature=temperature,
                    do_sample=False,  # Greedy/beam search (deterministic)
                    no_repeat_ngram_size=3,  # Avoid repetition
                )

            # Decode output tokens to string
            answer = self.tokenizer.decode(
                output_tokens[0],
                skip_special_tokens=True
            ).strip()

            logger.info(
                f"✅ Answer generated | "
                f"Length: {len(answer)} chars"
            )

            return answer if answer else "I could not generate an answer."

        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise

    def answer(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 256
    ) -> str:
        """
        Simplified public interface for answer generation.

        Args:
            question:       User's question
            context:        Context from RAG retriever
            max_new_tokens: Max tokens to generate

        Returns:
            Answer string
        """
        return self.generate(
            question=question,
            context=context,
            max_new_tokens=max_new_tokens
        )

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return settings.LLM_MODEL