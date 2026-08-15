"""
Gradio deployment interface — Personalised Transformer Chatbot.

Deployment architecture
-----------------------
The chatbot model is hosted on Hugging Face and accessed through the
Hugging Face Inference API.

Render runs only the Gradio interface.

Required Render environment variables:
    HF_TOKEN = your Hugging Face access token

Optional:
    HF_MODEL = Hugging Face model ID

Default model:
    sebastiantrbl/DialoGPT-finetuned-daily-dialog

The application includes:
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
from datetime import datetime

import gradio as gr
from huggingface_hub import InferenceClient


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.environ.get(
    "HF_MODEL",
    "sebastiantrbl/DialoGPT-finetuned-daily-dialog",
)

HF_TOKEN_ENV_NAME = "HF_TOKEN"

FEEDBACK_LOG_PATH = "./app/feedback_log.csv"


# ============================================================
# SIMPLE DIALOGUE STATE
# ============================================================

class SimpleDialogueState:
    """
    Lightweight dialogue state used by the deployment interface.

    This keeps the interface compatible with the dissertation design
    while the deployed model is accessed through Hugging Face.
    """

    def __init__(self):
        self.current_topic = ""
        self.current_intent = ""
        self.unresolved_needs = []
        self.formality_score = 0.5
        self.topic_interests = {}
        self.emotional_tone = ""
        self.explicit_preferences = []


# ============================================================
# RESEARCH DISCLAIMER
# ============================================================

DISCLAIMER = (
    "⚠️ **Research Prototype Notice**: You are interacting with an academic AI "
    "research system (MSc dissertation project). Responses may contain errors, "
    "inconsistencies, or biases, and this system is **not** designed for sensitive "
    "applications (e.g. medical, legal, or crisis support). No automated decisions "
    "are made based on its outputs. Participation in feedback collection is "
    "voluntary and anonymised (Section 3.9)."
)


# ============================================================
# FEEDBACK LOGGING
# ============================================================

def log_feedback(
    participant_code,
    session_id,
    turn_idx,
    rating,
    comment,
):
    """
    Save participant feedback to CSV.

    The file is stored on the Render instance.
    """

    feedback_directory = os.path.dirname(FEEDBACK_LOG_PATH)

    if feedback_directory:
        os.makedirs(
            feedback_directory,
            exist_ok=True,
        )

    file_exists = os.path.isfile(
        FEEDBACK_LOG_PATH
    )

    with open(
        FEEDBACK_LOG_PATH,
        "a",
        newline="",
        encoding="utf-8",
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
# HUGGING FACE CHATBOT
# ============================================================

class HuggingFaceChatbot:
    """
    Chatbot wrapper around the Hugging Face Inference API.

    The Hugging Face token is NEVER hard-coded.
    It must be supplied through the HF_TOKEN environment variable.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ):

        self.model_name = model_name

        # ----------------------------------------------------
        # Read token securely from environment
        # ----------------------------------------------------

        hf_token = os.environ.get(
            HF_TOKEN_ENV_NAME
        )

        if not hf_token:

            raise RuntimeError(
                "HF_TOKEN is not configured. "
                "Add HF_TOKEN as an environment variable "
                "in the Render service."
            )

        print("=" * 70)
        print("Initialising Hugging Face deployment")
        print(f"Model: {self.model_name}")
        print("Hugging Face token: configured")
        print("=" * 70)

        # ----------------------------------------------------
        # Hugging Face client
        # ----------------------------------------------------

        self.client = InferenceClient(
            api_key=hf_token,
            provider="auto",
        )

        print(
            "Hugging Face InferenceClient initialised successfully."
        )

    # ========================================================
    # GENERATE RESPONSE
    # ========================================================

    def generate_response(
        self,
        persona_sentences,
        raw_history,
        persona_enabled=True,
    ):

        # ----------------------------------------------------
        # Build system instruction
        # ----------------------------------------------------

        if persona_enabled and persona_sentences:

            persona_text = " ".join(
                persona_sentences
            )

            system_instruction = (
                "You are a personalised conversational assistant. "
                "Use the user's persona information when it is relevant "
                "to the conversation. Do not mention the persona explicitly "
                "unless the user asks about it. Respond naturally, helpfully, "
                "and conversationally.\n\n"
                f"User persona: {persona_text}"
            )

        else:

            system_instruction = (
                "You are a friendly conversational assistant. "
                "Respond naturally, helpfully, and appropriately "
                "to the user's messages."
            )

        messages = [
            {
                "role": "system",
                "content": system_instruction,
            }
        ]

        # ----------------------------------------------------
        # Add conversation history
        # ----------------------------------------------------

        for idx, text in enumerate(
            raw_history
        ):

            if not text:
                continue

            if idx % 2 == 0:

                messages.append({
                    "role": "user",
                    "content": str(text),
                })

            else:

                messages.append({
                    "role": "assistant",
                    "content": str(text),
                })

        # ----------------------------------------------------
        # Call Hugging Face
        # ----------------------------------------------------

        try:

            completion = (
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=100,
                    temperature=0.8,
                    top_p=0.92,
                )
            )

            response = (
                completion
                .choices[0]
                .message
                .content
            )

            if response is None:
                response = ""

            response = response.strip()

        except Exception as exc:

            print(
                "Hugging Face inference error:"
            )

            print(
                repr(exc)
            )

            return (
                "I'm sorry, I couldn't generate a response "
                "right now. Please try again."
            )

        # ----------------------------------------------------
        # Empty-response fallback
        # ----------------------------------------------------

        if not response:

            return (
                "I'm sorry, I couldn't generate a response."
            )

        return response


# ============================================================
# GRADIO APPLICATION
# ============================================================

def launch_app(system):

    with gr.Blocks(
        title=(
            "Personalised Transformer Chatbot "
            "— Research Prototype"
        )
    ) as demo:

        # ====================================================
        # HEADER
        # ====================================================

        gr.Markdown(
            "## Personalised Transformer Chatbot "
            "(MSc Dissertation Research Prototype)"
        )

        gr.Markdown(
            DISCLAIMER
        )

        # ====================================================
        # SESSION STATE
        # ====================================================

        state_history = gr.State(
            []
        )

        state_dialogue_state = gr.State(
            SimpleDialogueState()
        )

        state_persona = gr.State(
            []
        )

        # ====================================================
        # MAIN LAYOUT
        # ====================================================

        with gr.Row():

            # =================================================
            # LEFT: CHAT
            # =================================================

            with gr.Column(
                scale=3
            ):

                gr.Markdown(
                    "### 1. Chat window"
                )

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

                # =================================================
                # FEEDBACK
                # =================================================

                gr.Markdown(
                    "### 4. Feedback"
                )

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

                feedback_status = gr.Markdown(
                    ""
                )

            # =================================================
            # RIGHT: PROFILE
            # =================================================

            with gr.Column(
                scale=1
            ):

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

                # =================================================
                # PERSONA MODE
                # =================================================

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

        # ====================================================
        # HISTORY DISPLAY HELPER
        # ====================================================

        def history_to_messages(
            history
        ):

            messages = []

            if not history:
                return messages

            for idx, text in enumerate(
                history
            ):

                messages.append({
                    "role": (
                        "user"
                        if idx % 2 == 0
                        else "assistant"
                    ),
                    "content": str(text),
                })

            return messages

        # ====================================================
        # RESPONSE FUNCTION
        # ====================================================

        def respond(
            user_message,
            history,
            dialogue_state,
            persona_sentences,
            persona_enabled,
        ):

            # ------------------------------------------------
            # Empty message
            # ------------------------------------------------

            if not user_message:

                safe_history = (
                    history
                    if history
                    else []
                )

                return (
                    history_to_messages(
                        safe_history
                    ),
                    safe_history,
                    dialogue_state,
                    "",
                    0.5,
                    "",
                )

            # ------------------------------------------------
            # Initialise state
            # ------------------------------------------------

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
            # Build conversation history
            # ------------------------------------------------

            raw_history = (
                history
                + [user_message]
            )

            # ------------------------------------------------
            # Generate response
            # ------------------------------------------------

            reply = system.generate_response(
                active_personas,
                raw_history,
                persona_enabled=persona_enabled,
            )

            # ------------------------------------------------
            # Update lightweight dialogue state
            # ------------------------------------------------

            dialogue_state.current_topic = (
                str(user_message)[:80]
            )

            dialogue_state.emotional_tone = (
                "neutral"
            )

            dialogue_state.formality_score = (
                0.5
            )

            # ------------------------------------------------
            # Update history
            # ------------------------------------------------

            new_history = (
                raw_history
                + [reply]
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

        # ====================================================
        # PERSONA FUNCTION
        # ====================================================

        def set_persona(
            persona_text
        ):

            if not persona_text:
                return []

            return [
                sentence.strip()
                for sentence in persona_text.split(
                    "\n"
                )
                if sentence.strip()
            ]

        # ====================================================
        # CLEAR CONVERSATION
        # ====================================================

        def clear_conversation():

            return (
                [],
                [],
                SimpleDialogueState(),
                "",
                0.5,
                "",
            )

        # ====================================================
        # FEEDBACK FUNCTION
        # ====================================================

        def submit_feedback(
            p_code,
            s_id,
            history,
            rating_val,
            comment,
        ):

            if not history:
                turn_idx = 0
            else:
                turn_idx = (
                    len(history)
                    // 2
                )

            return log_feedback(
                p_code,
                s_id,
                turn_idx,
                rating_val,
                comment,
            )

        # ====================================================
        # SEND BUTTON
        # ====================================================

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

        # ====================================================
        # ENTER KEY
        # ====================================================

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

        # ====================================================
        # SET PERSONA
        # ====================================================

        set_persona_btn.click(
            set_persona,

            inputs=[
                persona_input
            ],

            outputs=[
                state_persona
            ],
        )

        # ====================================================
        # CLEAR
        # ====================================================

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

        # ====================================================
        # FEEDBACK
        # ====================================================

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

    parser = argparse.ArgumentParser(
        description=(
            "Launch the Personalised Transformer "
            "Chatbot Gradio application."
        )
    )

    parser.add_argument(
        "--model_name",
        default=MODEL_NAME,
        help=(
            "Hugging Face model ID. "
            "Defaults to HF_MODEL or the configured model."
        ),
    )

    parser.add_argument(
        "--host",
        default=None,
        help="Optional host override.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Optional port override.",
    )

    args = parser.parse_args()

    # ========================================================
    # INITIALISE MODEL CLIENT
    # ========================================================

    system = HuggingFaceChatbot(
        model_name=args.model_name
    )

    # ========================================================
    # BUILD APPLICATION
    # ========================================================

    demo = launch_app(
        system
    )

    # ========================================================
    # RENDER CONFIGURATION
    # ========================================================

    # Render supplies PORT automatically.
    # We use 10000 only as a local fallback.

    port = int(
        os.environ.get(
            "PORT",
            args.port or 10000,
        )
    )

    # Render requires the application to listen
    # on 0.0.0.0 rather than 127.0.0.1.

    host = os.environ.get(
        "GRADIO_SERVER_NAME",
        args.host or "0.0.0.0",
    )

    print("=" * 70)
    print("Starting Personalised Transformer Chatbot")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Model: {args.model_name}")
    print("=" * 70)

    # ========================================================
    # START GRADIO
    # ========================================================

    demo.launch(
        server_name=host,
        server_port=port,
        show_error=True,
    )