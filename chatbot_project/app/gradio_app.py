"""
Gradio deployment interface — Personalised Transformer Chatbot.

Deployment model:
    sebastiantrbl/DialoGPT-finetuned-daily-dialog

This version loads the Hugging Face Transformer model directly.
It does not require a local Stage-1 checkpoint or Stage-2 UPE/DST
checkpoint.

The interface preserves:
    1. Chat window
    2. User persona/profile panel
    3. Persona-mode toggle
    4. Per-response feedback
    5. Research disclaimer
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
from datetime import datetime

import gradio as gr


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "sebastiantrbl/DialoGPT-finetuned-daily-dialog"


# ============================================================
# SIMPLE DIALOGUE STATE
# ============================================================

class SimpleDialogueState:
    def __init__(self):
        self.current_topic = ""
        self.current_intent = ""
        self.unresolved_needs = []
        self.formality_score = 0.5
        self.topic_interests = {}
        self.emotional_tone = ""
        self.explicit_preferences = []


# ============================================================
# DISCLAIMER
# ============================================================

DISCLAIMER = (
    "⚠️ **Research Prototype Notice**: You are interacting with an academic AI "
    "research system (MSc dissertation project). Responses may contain errors, "
    "inconsistencies, or biases, and this system is **not** designed for sensitive "
    "applications (e.g. medical, legal, or crisis support). No automated decisions "
    "are made based on its outputs. Participation in feedback collection is voluntary "
    "and anonymised (Section 3.9)."
)


# ============================================================
# FEEDBACK
# ============================================================

FEEDBACK_LOG_PATH = "./app/feedback_log.csv"


def log_feedback(participant_code, session_id, turn_idx, rating, comment):

    os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)

    file_exists = os.path.isfile(FEEDBACK_LOG_PATH)

    with open(
        FEEDBACK_LOG_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "participant_code",
                "session_id",
                "turn_idx",
                "rating",
                "comment",
            ])

        writer.writerow([
            datetime.utcnow().isoformat(),
            participant_code,
            session_id,
            turn_idx,
            rating,
            comment,
        ])

    return "Feedback recorded. Thank you."


# ============================================================
# MODEL SYSTEM
# ============================================================

class HuggingFaceChatbot:

    def __init__(self, model_name=MODEL_NAME, device=None):

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print("=" * 60)
        print("Loading deployment model")
        print(f"Model: {model_name}")
        print(f"Device: {device}")
        print("=" * 60)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.to(device)
        self.model.eval()

        # DialoGPT does not normally have a pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Model loaded successfully.")

    def generate_response(
        self,
        persona_sentences,
        raw_history,
        persona_enabled=True,
    ):

        import torch

        # ----------------------------------------------------
        # Build conversation context
        # ----------------------------------------------------

        context_parts = []

        # Add persona information if enabled
        if persona_enabled and persona_sentences:

            persona_text = " ".join(persona_sentences)

            context_parts.append(
                f"Persona: {persona_text}"
            )

        # Add previous conversation
        if raw_history:

            context_parts.extend(raw_history)

        prompt = "\n".join(context_parts)

        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        with torch.no_grad():

            output_ids = self.model.generate(
                **inputs,

                max_new_tokens=80,

                do_sample=True,

                temperature=0.8,

                top_p=0.92,

                top_k=50,

                repetition_penalty=1.1,

                pad_token_id=self.tokenizer.eos_token_id,

            )

        # ----------------------------------------------------
        # Extract only generated response
        # ----------------------------------------------------

        generated_ids = output_ids[
            :, inputs["input_ids"].shape[-1]:
        ]

        response = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        ).strip()

        # Safety fallback
        if not response:
            response = "I'm sorry, I couldn't generate a response."

        return response


# ============================================================
# PORT
# ============================================================

def find_free_port(start_port=7860, max_attempts=10):

    for port in range(
        start_port,
        start_port + max_attempts
    ):

        with socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        ) as sock:

            sock.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_REUSEADDR,
                1
            )

            try:

                sock.bind(
                    ("127.0.0.1", port)
                )

                return port

            except OSError:

                continue

    raise OSError(
        f"No free port found between "
        f"{start_port} and "
        f"{start_port + max_attempts - 1}"
    )


# ============================================================
# GRADIO APPLICATION
# ============================================================

def launch_app(system):

    with gr.Blocks(
        title="Personalised Transformer Chatbot — Research Prototype"
    ) as demo:

        gr.Markdown(
            "## Personalised Transformer Chatbot "
            "(MSc Dissertation Research Prototype)"
        )

        gr.Markdown(DISCLAIMER)

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        state_history = gr.State([])

        state_dialogue_state = gr.State(
            SimpleDialogueState()
        )

        state_persona = gr.State([])

        # ----------------------------------------------------
        # Layout
        # ----------------------------------------------------

        with gr.Row():

            # =================================================
            # CHAT
            # =================================================

            with gr.Column(scale=3):

                gr.Markdown("### 1. Chat window")

                chatbot_ui = gr.Chatbot(
                    label="Conversation",
                    height=420,
                )

                msg_box = gr.Textbox(
                    label="Your message",
                    placeholder=(
                        "Type a message and press Enter..."
                    ),
                )

                with gr.Row():

                    send_btn = gr.Button(
                        "Send",
                        variant="primary",
                    )

                    clear_btn = gr.Button(
                        "Clear conversation"
                    )

                # ------------------------------------------------
                # Feedback
                # ------------------------------------------------

                gr.Markdown("### 4. Feedback")

                with gr.Row():

                    rating = gr.Radio(
                        ["1", "2", "3", "4", "5"],
                        label=(
                            "Rate the last response "
                            "(1=poor, 5=excellent)"
                        ),
                    )

                    feedback_comment = gr.Textbox(
                        label="Optional comment",
                        placeholder=(
                            "Anything you'd like to note?"
                        ),
                    )

                with gr.Row():

                    participant_code = gr.Textbox(
                        label="Participant code",
                        value="P000",
                    )

                    session_id = gr.Textbox(
                        label="Session ID",
                        value="session_1",
                    )

                feedback_btn = gr.Button(
                    "Submit feedback for last response"
                )

                feedback_status = gr.Markdown("")

            # =================================================
            # PROFILE
            # =================================================

            with gr.Column(scale=1):

                gr.Markdown(
                    "### 2. User profile panel"
                )

                persona_input = gr.Textbox(
                    label=(
                        "Your persona sentences "
                        "(one per line)"
                    ),
                    placeholder=(
                        "I enjoy hiking.\n"
                        "I work as a teacher.\n"
                        "I have two dogs."
                    ),
                    lines=5,
                )

                set_persona_btn = gr.Button(
                    "Set / update persona"
                )

                gr.Markdown(
                    "**Inferred preferences (editable)**"
                )

                inferred_topic = gr.Textbox(
                    label="Current topic",
                    interactive=True,
                )

                inferred_formality = gr.Slider(
                    0,
                    1,
                    value=0.5,
                    label=(
                        "Formality "
                        "(0=informal, 1=formal)"
                    ),
                )

                inferred_tone = gr.Textbox(
                    label="Emotional tone",
                    interactive=True,
                )

                gr.Markdown(
                    "### 3. Persona mode"
                )

                persona_toggle = gr.Checkbox(
                    label=(
                        "Enable persona-conditioned "
                        "personalisation"
                    ),
                    value=True,
                )

        # =====================================================
        # HELPERS
        # =====================================================

        def history_to_messages(history):

            messages = []

            for idx, text in enumerate(history):

                messages.append({
                    "role": (
                        "user"
                        if idx % 2 == 0
                        else "assistant"
                    ),
                    "content": text,
                })

            return messages

        # =====================================================
        # RESPONSE
        # =====================================================

        def respond(
            user_message,
            history,
            dialogue_state,
            persona_sentences,
            persona_enabled,
        ):

            if not user_message:

                return (
                    history_to_messages(history),
                    history,
                    dialogue_state,
                    "",
                    0.5,
                    "",
                )

            if history is None:
                history = []

            if dialogue_state is None:
                dialogue_state = (
                    SimpleDialogueState()
                )

            if persona_sentences is None:
                persona_sentences = []

            # ------------------------------------------------
            # Active persona
            # ------------------------------------------------

            active_personas = (
                persona_sentences
                if persona_enabled
                else []
            )

            # ------------------------------------------------
            # Generate
            # ------------------------------------------------

            raw_history = (
                history + [user_message]
            )

            reply = system.generate_response(
                active_personas,
                raw_history,
                persona_enabled=persona_enabled,
            )

            # ------------------------------------------------
            # Simple profile updates
            # ------------------------------------------------

            dialogue_state.current_topic = (
                user_message[:80]
            )

            dialogue_state.emotional_tone = (
                "neutral"
            )

            dialogue_state.formality_score = (
                0.5
            )

            # ------------------------------------------------
            # New history
            # ------------------------------------------------

            new_history = (
                raw_history + [reply]
            )

            display = history_to_messages(
                new_history
            )

            return (
                display,
                new_history,
                dialogue_state,
                dialogue_state.current_topic,
                dialogue_state.formality_score,
                dialogue_state.emotional_tone,
            )

        # =====================================================
        # PERSONA
        # =====================================================

        def set_persona(persona_text):

            if not persona_text:
                return []

            return [
                sentence.strip()
                for sentence in persona_text.split("\n")
                if sentence.strip()
            ]

        # =====================================================
        # CLEAR
        # =====================================================

        def clear_conversation():

            return (
                [],
                [],
                SimpleDialogueState(),
                "",
                0.5,
                "",
            )

        # =====================================================
        # FEEDBACK
        # =====================================================

        def submit_feedback(
            p_code,
            s_id,
            history,
            rating_val,
            comment,
        ):

            turn_idx = (
                len(history) // 2
                if history
                else 0
            )

            return log_feedback(
                p_code,
                s_id,
                turn_idx,
                rating_val,
                comment,
            )

        # =====================================================
        # BUTTON EVENTS
        # =====================================================

        send_btn.click(
            respond,

            inputs=[
                msg_box,
                state_history,
                state_dialogue_state,
                state_persona,
                persona_toggle,
            ],

            outputs=[
                chatbot_ui,
                state_history,
                state_dialogue_state,
                inferred_topic,
                inferred_formality,
                inferred_tone,
            ],
        ).then(
            lambda: "",
            None,
            msg_box,
        )

        msg_box.submit(
            respond,

            inputs=[
                msg_box,
                state_history,
                state_dialogue_state,
                state_persona,
                persona_toggle,
            ],

            outputs=[
                chatbot_ui,
                state_history,
                state_dialogue_state,
                inferred_topic,
                inferred_formality,
                inferred_tone,
            ],
        ).then(
            lambda: "",
            None,
            msg_box,
        )

        set_persona_btn.click(
            set_persona,
            inputs=[persona_input],
            outputs=[state_persona],
        )

        clear_btn.click(
            clear_conversation,

            outputs=[
                chatbot_ui,
                state_history,
                state_dialogue_state,
                inferred_topic,
                inferred_formality,
                inferred_tone,
            ],
        )

        feedback_btn.click(
            submit_feedback,

            inputs=[
                participant_code,
                session_id,
                state_history,
                rating,
                feedback_comment,
            ],

            outputs=[
                feedback_status
            ],
        )

    return demo


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default=MODEL_NAME,
        help="Hugging Face model ID",
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Use Gradio public URL sharing",
    )

    parser.add_argument(
        "--host",
        default=None,
        help="Host/IP for Gradio",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for Gradio",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    system = HuggingFaceChatbot(
        model_name=args.model_name
    )

    # --------------------------------------------------------
    # Launch
    # --------------------------------------------------------

    demo = launch_app(system)

    env_port = int(
        os.environ.get("PORT", "0")
    )

    preferred_port = (
        args.port
        if args.port is not None
        else (
            env_port
            if env_port > 0
            else 0
        )
    )

    candidate_ports = (
        [preferred_port]
        if preferred_port > 0
        else list(range(7860, 7875))
    )

    if args.host:

        server_name = args.host

    elif env_port > 0:

        server_name = os.environ.get(
            "GRADIO_SERVER_NAME",
            "0.0.0.0",
        )

    else:

        server_name = os.environ.get(
            "GRADIO_SERVER_NAME",
            "127.0.0.1",
        )

    last_error = None

    for port in candidate_ports:

        try:

            print(
                f"Launching Gradio on "
                f"http://{server_name}:{port}"
            )

            demo.launch(
                share=args.share,
                server_name=server_name,
                server_port=port,
            )

            break

        except OSError as exc:

            last_error = exc

            print(
                f"Port {port} unavailable; "
                f"trying next port."
            )

    else:

        raise (
            last_error
            or OSError(
                "Unable to launch Gradio"
            )
        )