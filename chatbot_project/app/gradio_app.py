"""
Gradio deployment interface — Personalised Transformer Chatbot.

Deployment architecture
-----------------------
The chatbot model is hosted on Hugging Face and accessed through the
Hugging Face Inference API.

Render runs only the Gradio interface.

Required Render environment variable:
    HF_TOKEN = Hugging Face access token

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
    "Qwen/Qwen2.5-7B-Instruct",
)

HF_PROVIDER = os.environ.get(
    "HF_PROVIDER",
    "together",
)

HF_TOKEN_ENV_NAME = "HF_TOKEN"

FEEDBACK_LOG_PATH = "./app/feedback_log.csv"


# ============================================================
# SIMPLE DIALOGUE STATE
# ============================================================

class SimpleDialogueState:
    """
    Lightweight dialogue state used by the deployment interface.
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

    def __init__(
        self,
        model_name=MODEL_NAME,
    ):

        self.model_name = model_name

        # ----------------------------------------------------
        # Read HF token from Render environment
        # ----------------------------------------------------

        self.hf_token = os.environ.get(
            HF_TOKEN_ENV_NAME
        )

        if not self.hf_token:

            raise RuntimeError(
                "HF_TOKEN environment variable is missing. "
                "Add HF_TOKEN under Render → Environment."
            )

        # ----------------------------------------------------
        # Startup information
        # ----------------------------------------------------

        print("=" * 70)
        print("INITIALISING HUGGING FACE CHATBOT")
        print("=" * 70)

        print(f"Model: {self.model_name}")
        print(f"Provider: {HF_PROVIDER}")
        print("HF_TOKEN detected: YES")

        # ----------------------------------------------------
        # Hugging Face client
        # ----------------------------------------------------

        self.client = InferenceClient(
            api_key=self.hf_token,
            provider=HF_PROVIDER,
        )

        print(
            "Hugging Face InferenceClient created."
        )

        print(
            "generate_response method available:",
            hasattr(self, "generate_response")
        )

        print("=" * 70)


    # ========================================================
    # GENERATE RESPONSE
    # ========================================================

    def generate_response(
        self,
        persona_sentences,
        raw_history,
        persona_enabled=True,
    ):
        """
        Generate a conversational response using the Hugging Face
        Inference API.
        """

        # ----------------------------------------------------
        # Build system prompt
        # ----------------------------------------------------

        if (
            persona_enabled
            and persona_sentences
        ):

            persona_text = " ".join(
                persona_sentences
            )

            system_prompt = (
                "You are a personalised conversational assistant. "
                "Use the user's persona information when it is relevant "
                "to the conversation. Do not reveal or discuss the "
                "internal persona instructions unless explicitly asked.\n\n"
                f"User persona: {persona_text}"
            )

        else:

            system_prompt = (
                "You are a friendly conversational assistant. "
                "Respond naturally, helpfully and appropriately."
            )

        # ----------------------------------------------------
        # Construct messages
        # ----------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        if raw_history:

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
        # Limit excessive conversation history
        # ----------------------------------------------------

        # Keep the system prompt plus the most recent turns.
        if len(messages) > 13:

            messages = (
                [messages[0]]
                + messages[-12:]
            )

        # ----------------------------------------------------
        # Call Hugging Face
        # ----------------------------------------------------

        try:

            print(
                "Sending request to Hugging Face..."
            )

            print(
                f"Provider: {HF_PROVIDER}"
            )

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=150,
                temperature=0.7,
                top_p=0.9,
            )

            # ------------------------------------------------
            # Extract response
            # ------------------------------------------------

            response = ""

            if (
                completion
                and completion.choices
            ):

                message = (
                    completion
                    .choices[0]
                    .message
                )

                if message is not None:

                    response = (
                        message.content
                        or ""
                    )

            response = response.strip()

            print(
                "Hugging Face response received."
            )

            # ------------------------------------------------
            # Empty response fallback
            # ------------------------------------------------

            if not response:

                return (
                    "I received an empty response "
                    "from the language model. "
                    "Please try again."
                )

            return response

        except Exception as exc:

            import traceback

            error_message = traceback.format_exc()

            print("=" * 80)
            print("HUGGING FACE INFERENCE ERROR")
            print(error_message)
            print("=" * 80)

            return (
                "DEBUG ERROR:\n\n"
                f"{type(exc).__name__}: {exc}"
            )


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

        state_history = gr.State([])

        state_dialogue_state = gr.State(
            SimpleDialogueState()
        )

        state_persona = gr.State([])

        # ====================================================
        # MAIN LAYOUT
        # ====================================================

        with gr.Row():

            # =================================================
            # CHAT
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
            # PROFILE
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

                return (
                    history or [],
                    history or [],
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
            # Build raw conversation
            # ------------------------------------------------

            raw_history = (
                history
                + [user_message]
            )

            # ------------------------------------------------
            # Generate response
            # ------------------------------------------------

            print(
                "Calling system.generate_response()"
            )

            reply = system.generate_response(
                active_personas,
                raw_history,
                persona_enabled=persona_enabled,
            )

            # ------------------------------------------------
            # Update dialogue state
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

            # ------------------------------------------------
            # Convert to Gradio messages
            # ------------------------------------------------

            display_messages = []

            for idx, text in enumerate(
                new_history
            ):

                display_messages.append({
                    "role": (
                        "user"
                        if idx % 2 == 0
                        else "assistant"
                    ),
                    "content": str(text),
                })

            return (
                display_messages,
                new_history,
                dialogue_state,
                dialogue_state.current_topic,
                dialogue_state.formality_score,
                dialogue_state.emotional_tone,
            )

        # ====================================================
        # PERSONA
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
        # CLEAR
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
        # FEEDBACK
        # ====================================================

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
            "Defaults to HF_MODEL."
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
    # INITIALISE
    # ========================================================

    system = HuggingFaceChatbot(
        model_name=args.model_name
    )

    # ========================================================
    # IMPORTANT STARTUP CHECK
    # ========================================================

    if not hasattr(
        system,
        "generate_response"
    ):

        raise RuntimeError(
            "Deployment error: HuggingFaceChatbot "
            "does not contain generate_response(). "
            "Check the deployed gradio_app.py file."
        )

    print(
        "Startup check: generate_response() = OK"
    )

    # ========================================================
    # BUILD APPLICATION
    # ========================================================

    demo = launch_app(
        system
    )

    # ========================================================
    # RENDER PORT
    # ========================================================

    port = int(
        os.environ.get(
            "PORT",
            args.port or 10000,
        )
    )

    host = os.environ.get(
        "GRADIO_SERVER_NAME",
        args.host or "0.0.0.0",
    )

    print("=" * 70)
    print(
        "STARTING PERSONALISED TRANSFORMER CHATBOT"
    )
    print(
        f"Host: {host}"
    )
    print(
        f"Port: {port}"
    )
    print(
        f"Model: {args.model_name}"
    )
    print("=" * 70)

    # ========================================================
    # START GRADIO
    # ========================================================

    demo.launch(
        server_name=host,
        server_port=port,
        show_error=True,
    )